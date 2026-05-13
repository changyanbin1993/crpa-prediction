import pandas as pd
import numpy as np
import os
import platform
import importlib
import warnings
warnings.filterwarnings('once')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import stats
from collections import OrderedDict

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_SEED = 42
PLOT_DIR    = './plots'
TABLE_DIR   = './tables'
MODEL_DIR   = './models'

for d in [PLOT_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


class ModelInterpreter:
    """模型解释类"""

    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None

    def load_data(self):
        print("=" * 70)
        print("步骤1: 加载数据")
        print("=" * 70)

        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        self.X_train = train_df.drop('CRPA', axis=1)
        self.y_train = train_df['CRPA']
        self.X_val = val_df.drop('CRPA', axis=1)
        self.y_val = val_df['CRPA']
        self.feature_names = list(self.X_train.columns)

        print(f"训练集: {self.X_train.shape}, 测试集: {self.X_val.shape}")
        return self

    def load_best_model(self, model_name='XGBoost'):
        print(f"\n步骤2: 加载模型 ({model_name})")

        self.best_model_name = model_name

        if model_name == 'XGBoost':
            self.best_model = xgb.Booster()
            self.best_model.load_model(f'{MODEL_DIR}/xgboost_model.json')
        elif model_name == 'LightGBM':
            self.best_model = lgb.Booster(model_file=f'{MODEL_DIR}/lightgbm_model.txt')
        elif model_name == 'CatBoost':
            self.best_model = CatBoostClassifier()
            self.best_model.load_model(f'{MODEL_DIR}/catboost_model.cbm')
        elif model_name in ['Random Forest', 'AdaBoost']:
            fname = f'{MODEL_DIR}/{model_name.lower().replace(" ", "_")}_model.pkl'
            loaded = joblib.load(fname)
            self.best_model = loaded['model'] if isinstance(loaded, dict) else loaded
        print(f"✓ {model_name} 已加载")
        return self

    # ───────────────────────────────────
    # 校准曲线（5模型）
    # ───────────────────────────────────

    def _load_all_predictions(self):
        """加载所有模型的验证集预测概率"""
        preds = OrderedDict()
        loaders = {
            'XGBoost': lambda: xgb.Booster(model_file=f'{MODEL_DIR}/xgboost_model.json').predict(
                xgb.DMatrix(self.X_val)),
            'LightGBM': lambda: lgb.Booster(model_file=f'{MODEL_DIR}/lightgbm_model.txt').predict(
                self.X_val),
        }

        def _sklearn_pred(path):
            loaded = joblib.load(path)
            m = loaded['model'] if isinstance(loaded, dict) else loaded
            return m.predict_proba(self.X_val)[:, 1]

        loaders['Random Forest'] = lambda: _sklearn_pred(f'{MODEL_DIR}/random_forest_model.pkl')
        loaders['AdaBoost'] = lambda: _sklearn_pred(f'{MODEL_DIR}/adaboost_model.pkl')

        def _catboost_pred():
            m = CatBoostClassifier()
            m.load_model(f'{MODEL_DIR}/catboost_model.cbm')
            return m.predict(self.X_val, prediction_type='Probability')[:, 1]
        loaders['CatBoost'] = _catboost_pred

        for name, fn in loaders.items():
            try:
                preds[name] = fn()
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        return preds

    def plot_calibration_curve_all_models(self):
        print("\n" + "=" * 70)
        print("步骤3: 校准曲线（5模型）")
        print("=" * 70)

        preds = self._load_all_predictions()
        colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e',
                  'Random Forest': '#2ca02c', 'CatBoost': '#d62728',
                  'AdaBoost': '#9467bd'}
        brier = {n: brier_score_loss(self.y_val, p) for n, p in preds.items()}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')

        for name, yp in preds.items():
            frac, mean_pred = calibration_curve(self.y_val, yp, n_bins=10, strategy='quantile')
            ax1.plot(mean_pred, frac, 'o-', color=colors.get(name, 'gray'), lw=2,
                     label=f'{name} (Brier={brier[name]:.3f})')

        ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax1.set_title('Calibration Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(alpha=0.3)

        for name, yp in preds.items():
            ax2.hist(yp, bins=20, alpha=0.5, label=name,
                     color=colors.get(name, 'gray'), edgecolor='black', lw=0.5)
        ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Predicted Probability Distributions', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/07_calibration_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ {PLOT_DIR}/07_calibration_curves.png")
        plt.close()
        return self

    # ───────────────────────────────────
    # [额外] Hosmer-Lemeshow 检验
    # ───────────────────────────────────

    def hosmer_lemeshow_test(self):
        """[额外] 校准拟合优度检验"""
        print("\n" + "=" * 70)
        print("[额外] Hosmer-Lemeshow 检验")
        print("=" * 70)

        preds = self._load_all_predictions()

        def _hl(y_true, y_prob, n_groups=10):
            order = np.argsort(y_prob)
            ys, ps = np.array(y_true)[order], y_prob[order]
            groups = np.array_split(np.arange(len(ps)), n_groups)
            chi2_val = 0.0
            for g in groups:
                n_g = len(g)
                o1, e1 = ys[g].sum(), ps[g].sum()
                o0, e0 = n_g - o1, n_g - e1
                if e1 > 0: chi2_val += (o1 - e1) ** 2 / e1
                if e0 > 0: chi2_val += (o0 - e0) ** 2 / e0
            dof = n_groups - 2
            return chi2_val, dof, 1 - stats.chi2.cdf(chi2_val, dof)

        rows = []
        for name, yp in preds.items():
            c, d, p = _hl(self.y_val.values, yp)
            interp = 'Good fit' if p >= 0.05 else 'Poor fit'
            print(f"  {name:18s}: χ²={c:.3f}, df={d}, P={p:.4f} ({interp})")
            rows.append({'Model': name, 'HL_Chi2': f"{c:.3f}",
                         'df': d, 'P_value': f"{p:.4f}", 'Interpretation': interp})

        pd.DataFrame(rows).to_csv(f'{TABLE_DIR}/Suppl_Hosmer_Lemeshow.csv', index=False)
        print(f"\n✓ {TABLE_DIR}/Suppl_Hosmer_Lemeshow.csv")
        return self

    # ───────────────────────────────────
    # DCA（标题已修正）
    # ───────────────────────────────────

    def plot_decision_curve_analysis(self):
        print("\n" + "=" * 70)
        print("步骤4: 决策曲线分析 (DCA)")
        print("=" * 70)

        preds = self._load_all_predictions()
        thresholds = np.arange(0.01, 0.99, 0.01)
        n = len(self.y_val)
        prevalence = self.y_val.mean()
        nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)

        colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e',
                  'Random Forest': '#2ca02c', 'CatBoost': '#d62728',
                  'AdaBoost': '#9467bd'}

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, nb_all, 'k:', lw=2, label='Treat All')
        plt.axhline(y=0, color='gray', lw=2, label='Treat None')

        for name, yp in preds.items():
            nb = []
            for t in thresholds:
                yc = (yp >= t).astype(int)
                tp = ((yc == 1) & (self.y_val == 1)).sum()
                fp = ((yc == 1) & (self.y_val == 0)).sum()
                nb.append((tp / n) - (fp / n) * (t / (1 - t)))
            plt.plot(thresholds, nb, color=colors.get(name, 'gray'), lw=2.5, label=name)

        plt.xlim([0, 1])
        plt.ylim([-0.1, max(prevalence, 0.3) + 0.05])
        plt.xlabel('Threshold Probability', fontsize=14, fontweight='bold')
        plt.ylabel('Net Benefit', fontsize=14, fontweight='bold')
        # [SE-3/SE-1] 修正标题: 去掉 "CRPA Infection Prediction"
        plt.title('Decision Curve Analysis\n'
                  'Prediction of Carbapenem Resistance in Invasive P. aeruginosa',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/08_decision_curve_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ {PLOT_DIR}/08_decision_curve_analysis.png")
        plt.close()
        return self

    # ───────────────────────────────────
    # SHAP 全局分析（标注 log-odds）
    # ───────────────────────────────────

    def shap_global_analysis(self):
        """[SE-3] SHAP全局分析 — 标注 log-odds scale"""
        print("\n" + "=" * 70)
        print(f"步骤5: SHAP全局分析 ({self.best_model_name})")
        print("=" * 70)

        try:
            import shap

            if self.best_model_name == 'XGBoost':
                def xgb_predict(X):
                    return self.best_model.predict(
                        xgb.DMatrix(X, feature_names=self.feature_names))

                background = self.X_train.sample(n=min(100, len(self.X_train)),
                                                  random_state=RANDOM_SEED)
                print("  使用Kernel SHAP (兼容XGBoost 2.0+)...")
                explainer = shap.KernelExplainer(xgb_predict, background)

                n_samples = min(100, len(self.X_val))
                X_sample = self.X_val.iloc[:n_samples]
                print(f"  计算SHAP值 ({n_samples}个样本)...")
                shap_values = explainer.shap_values(X_sample, nsamples=100)

            elif self.best_model_name in ['Random Forest', 'AdaBoost']:
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(self.X_val)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                X_sample = self.X_val

            elif self.best_model_name == 'LightGBM':
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(self.X_val)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                X_sample = self.X_val

            elif self.best_model_name == 'CatBoost':
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(self.X_val)
                X_sample = self.X_val

            print("✓ SHAP值计算完成")

            # ── Summary Plot ──
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample,
                              feature_names=self.feature_names,
                              max_display=20, show=False)
            # [SE-3] 标注 log-odds scale
            plt.title(f'SHAP Summary Plot — {self.best_model_name}\n'
                      f'(SHAP values on log-odds scale)',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{PLOT_DIR}/09_shap_summary.png', dpi=300, bbox_inches='tight')
            print(f"✓ {PLOT_DIR}/09_shap_summary.png")
            plt.close()

            # ── Bar Plot ──
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample,
                              feature_names=self.feature_names,
                              plot_type='bar', max_display=20, show=False)
            plt.title(f'SHAP Feature Importance — {self.best_model_name}\n'
                      f'mean(|SHAP value|) on log-odds scale',
                      fontsize=14, fontweight='bold')
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(f'{PLOT_DIR}/10_shap_bar.png', dpi=300, bbox_inches='tight')
            print(f"✓ {PLOT_DIR}/10_shap_bar.png")
            plt.close()

            # 保存重要性
            mean_abs = np.abs(shap_values).mean(axis=0)
            if mean_abs.ndim > 1:
                mean_abs = mean_abs.flatten()
            imp_df = pd.DataFrame({'Feature': self.feature_names,
                                   'Mean_Abs_SHAP': mean_abs}
                                  ).sort_values('Mean_Abs_SHAP', ascending=False)
            imp_df.to_csv('./shap_feature_importance.csv', index=False)
            print("✓ shap_feature_importance.csv")

            self._shap_explainer = explainer
            self._shap_values = shap_values
            self._X_sample = X_sample

        except ImportError:
            print("⚠ SHAP未安装")
        except Exception as e:
            print(f"⚠ SHAP分析出错: {e}")
            import traceback; traceback.print_exc()

        return self

    # ───────────────────────────────────
    # SHAP 个案分析
    # ───────────────────────────────────

    def shap_individual_cases(self, n_cases=2):
        print(f"\n步骤6: SHAP个案解释 ({n_cases}个案例)")

        try:
            import shap

            if self.best_model_name == 'XGBoost':
                dval = xgb.DMatrix(self.X_val, feature_names=self.feature_names)
                y_pred = self.best_model.predict(dval)
            elif self.best_model_name == 'LightGBM':
                y_pred = self.best_model.predict(self.X_val)
            elif self.best_model_name in ['Random Forest', 'AdaBoost']:
                y_pred = self.best_model.predict_proba(self.X_val)[:, 1]
            elif self.best_model_name == 'CatBoost':
                y_pred = self.best_model.predict(self.X_val, prediction_type='Probability')[:, 1]

            if not hasattr(self, '_shap_values'):
                print("⚠ 请先运行 shap_global_analysis()")
                return self

            sample_size = len(self._X_sample)
            y_sub = self.y_val.iloc[:sample_size]
            yp_sub = y_pred[:sample_size]

            # 找 TP 和 TN
            tp_list = np.where((yp_sub > 0.5) & (y_sub == 1))[0]
            if len(tp_list) == 0:
                tp_list = np.where((yp_sub > 0.3) & (y_sub == 1))[0]
            tp_idx = tp_list[0] if len(tp_list) > 0 else 0

            tn_list = np.where((yp_sub < 0.3) & (y_sub == 0))[0]
            tn_idx = tn_list[0] if len(tn_list) > 0 else min(1, sample_size - 1)

            explainer = self._shap_explainer
            if hasattr(explainer, 'expected_value'):
                ev = explainer.expected_value
                base_value = float(ev[0]) if isinstance(ev, np.ndarray) and len(ev) > 0 else float(ev)
            else:
                base_value = 0.5

            for case_idx, label in [(tp_idx, 'True Positive (High Risk)'),
                                     (tn_idx, 'True Negative (Low Risk)')]:
                if case_idx >= len(self._shap_values):
                    continue

                sv = self._shap_values[case_idx]
                fv = self._X_sample.iloc[case_idx].values

                print(f"\n  {label}: actual={y_sub.iloc[case_idx]}, "
                      f"predicted={yp_sub[case_idx]:.2%}")

                max_display = 15
                order = np.argsort(np.abs(sv))[::-1][:max_display]
                order = order[np.argsort(sv[order])[::-1]]

                fig, ax = plt.subplots(figsize=(11.2, max(7.2, len(order) * 0.54)))
                ypos = np.arange(len(order))[::-1]
                bar_colors = ['#ff0051' if v > 0 else '#008bfb' for v in sv[order]]
                bars = ax.barh(ypos, sv[order], color=bar_colors, height=0.7, alpha=0.85)

                # 动态计算 xlim，保证数值标签不超出边界
                max_abs = max(np.abs(sv[order]).max(), 0.01)
                ax.set_xlim(-max_abs * 1.26, max_abs * 1.26)

                for i, (bar, val) in enumerate(zip(bars, sv[order])):
                    txt = f"{val:+.3f}"
                    bar_len = abs(val)
                    if bar_len > max_abs * 0.18:
                        # 长条：白色文字放在条内
                        xp = val * 0.55
                        ha = 'center'
                        col = 'white'
                    else:
                        # 短条：黑色文字放在条外，偏移量相对 xlim
                        offset = max_abs * 0.06
                        xp = val + offset if val >= 0 else val - offset
                        ha = 'left' if val >= 0 else 'right'
                        col = '#222222'
                    ax.text(xp, bar.get_y() + bar.get_height() / 2, txt,
                            va='center', ha=ha, fontsize=9, fontweight='bold', color=col)

                y_labels = []
                for v, idx in zip(fv[order], order):
                    fn = self.feature_names[idx]
                    v_str = f"{int(v)}" if isinstance(v, float) and v == int(v) else f"{v:.3g}"
                    y_labels.append(f"{v_str} = {fn}")

                ax.set_yticks(ypos)
                ax.set_yticklabels(y_labels, fontsize=11)
                ax.axvline(x=0, color='#888888', lw=1)

                final_val = base_value + np.sum(sv)
                ax.set_xlabel('SHAP Value (log-odds scale)', fontsize=12, fontweight='bold')
                ax.set_title(f'SHAP Waterfall — {label}\n'
                             f'E[f(X)]={base_value:.3f} → f(x)={final_val:.3f}',
                             fontsize=13, fontweight='bold')

                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor='#ff0051', alpha=0.85, label='↑ Risk'),
                                   Patch(facecolor='#008bfb', alpha=0.85, label='↓ Risk')],
                          loc='lower right', fontsize=10)
                ax.grid(axis='x', alpha=0.3)
                fig.subplots_adjust(left=0.34, right=0.98, top=0.90, bottom=0.10)

                safe = label.replace(" ", "_").replace("(", "").replace(")", "").lower()
                plt.savefig(
                    f'{PLOT_DIR}/11_shap_case_{safe}.png',
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.03
                )
                print(f"  ✓ {PLOT_DIR}/11_shap_case_{safe}.png")
                plt.close()

        except Exception as e:
            print(f"⚠ 个案分析出错: {e}")
            import traceback; traceback.print_exc()

        return self

    # ═══════════════════════════════════════
    # [SE-7] 敏感性分析 — 纳入被排除的高缺失变量
    # ═══════════════════════════════════════

    def sensitivity_analysis_missing_vars(self, raw_data_path='./CRPA.csv'):
        """[SE-7] 将之前排除的高缺失变量通过多重填补纳入，重新训练并比较"""
        print("\n" + "=" * 70)
        print("[SE-7] 敏感性分析 — 纳入被排除的高缺失变量")
        print("=" * 70)

        if not os.path.exists(raw_data_path):
            print(f"  ⚠ {raw_data_path} 不存在，跳过敏感性分析")
            print(f"    请将原始数据文件放到当前目录后重跑")
            return self

        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
        import xgboost as xgb_pkg

        raw_df = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        print(f"  原始数据: {raw_df.shape}")

        exclude_cols = ['Anchor Year Group', 'CRPA']
        feature_cols = [c for c in raw_df.columns if c not in exclude_cols]
        num_cols = raw_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        missing_pct = (raw_df[num_cols].isnull().sum() / len(raw_df) * 100)
        high_missing = missing_pct[missing_pct > 10]

        if len(high_missing) == 0:
            print("  ✓ 没有>10%缺失的变量")
            return self

        print(f"\n  被排除变量:")
        for v, p in high_missing.items():
            print(f"    {v}: {p:.1f}%")

        y = raw_df['CRPA'].values
        spw = (y == 0).sum() / (y == 1).sum()
        model_cls = xgb_pkg.XGBClassifier(
            objective='binary:logistic', max_depth=6, learning_rate=0.05,
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, use_label_encoder=False,
            random_state=RANDOM_SEED, verbosity=0)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

        # A: 主分析（排除高缺失变量）
        primary_cols = [c for c in num_cols if c not in high_missing.index]
        X_primary = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(raw_df[primary_cols]),
            columns=primary_cols)
        s1 = cross_val_score(model_cls, X_primary, y, cv=cv,
                              scoring='roc_auc', n_jobs=-1)
        print(f"\n  主分析 (排除>10%缺失):    CV AUROC = {s1.mean():.4f} ± {s1.std():.4f}")

        # B: MICE全变量
        X_full_imp = pd.DataFrame(
            IterativeImputer(max_iter=20, random_state=RANDOM_SEED).fit_transform(
                raw_df[num_cols]),
            columns=num_cols)
        s2 = cross_val_score(model_cls, X_full_imp, y, cv=cv,
                              scoring='roc_auc', n_jobs=-1)
        print(f"  方法A MICE全变量:          CV AUROC = {s2.mean():.4f} ± {s2.std():.4f}")

        # C: 缺失指示变量
        X_ind = raw_df[num_cols].copy()
        for v in high_missing.index:
            if v in X_ind.columns:
                X_ind[f'{v}_missing'] = X_ind[v].isna().astype(int)
        X_ind_imp = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(X_ind),
            columns=X_ind.columns)
        s3 = cross_val_score(model_cls, X_ind_imp, y, cv=cv,
                              scoring='roc_auc', n_jobs=-1)
        print(f"  方法B 缺失指示变量:        CV AUROC = {s3.mean():.4f} ± {s3.std():.4f}")

        result = pd.DataFrame([
            {'Analysis': 'Primary (exclude >10% missing)',
             'N_features': len(primary_cols),
             'CV_AUROC_mean': f"{s1.mean():.4f}", 'CV_AUROC_sd': f"{s1.std():.4f}"},
            {'Analysis': 'Sensitivity A: MICE (all vars)',
             'N_features': len(num_cols),
             'CV_AUROC_mean': f"{s2.mean():.4f}", 'CV_AUROC_sd': f"{s2.std():.4f}"},
            {'Analysis': 'Sensitivity B: Missingness indicators',
             'N_features': len(X_ind.columns),
             'CV_AUROC_mean': f"{s3.mean():.4f}", 'CV_AUROC_sd': f"{s3.std():.4f}"},
        ])
        result.to_csv(f'{TABLE_DIR}/Suppl_Sensitivity_MissingVars.csv', index=False)
        print(f"\n✓ {TABLE_DIR}/Suppl_Sensitivity_MissingVars.csv")

        return self

    # ═══════════════════════════════════════
    # [SE-9] 完整可重复性汇总
    # ═══════════════════════════════════════

    def reproducibility_summary(self):
        """[SE-9] 汇总所有可重复性信息"""
        print("\n" + "=" * 70)
        print("[SE-9] 可重复性信息汇总")
        print("=" * 70)

        info = OrderedDict()
        info['Python'] = platform.python_version()
        info['Platform'] = platform.platform()

        for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost',
                     'lightgbm', 'catboost', 'shap', 'matplotlib']:
            try:
                mod = importlib.import_module(pkg)
                info[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                info[pkg] = 'not installed'

        info['random_seed'] = str(RANDOM_SEED)

        print("\n  软件环境:")
        for k, v in info.items():
            print(f"    {k:18s}: {v}")

        env_df = pd.DataFrame(list(info.items()), columns=['Item', 'Value'])
        env_df.to_csv(f'{TABLE_DIR}/Suppl_Software_Versions.csv', index=False)

        # XGBoost超参数
        hp_path = f'{MODEL_DIR}/xgboost_hyperparameters.json'
        if os.path.exists(hp_path):
            import json
            with open(hp_path) as f:
                params = json.load(f)
            hp_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            hp_df.to_csv(f'{TABLE_DIR}/Suppl_XGBoost_Hyperparameters.csv', index=False)
            print(f"\n  XGBoost超参数:")
            for k, v in params.items():
                print(f"    {k:25s}: {v}")

        # CatBoost超参数
        cb_hp_path = f'{MODEL_DIR}/catboost_hyperparameters.json'
        if os.path.exists(cb_hp_path):
            import json
            with open(cb_hp_path) as f:
                cb_params = json.load(f)
            cb_hp_df = pd.DataFrame(list(cb_params.items()), columns=['Parameter', 'Value'])
            cb_hp_df.to_csv(f'{TABLE_DIR}/Suppl_CatBoost_Hyperparameters.csv', index=False)
            print(f"\n  CatBoost超参数:")
            for k, v in cb_params.items():
                print(f"    {k:25s}: {v}")

        print(f"\n✓ {TABLE_DIR}/Suppl_Software_Versions.csv")
        print(f"✓ {TABLE_DIR}/Suppl_XGBoost_Hyperparameters.csv")
        print(f"✓ {TABLE_DIR}/Suppl_CatBoost_Hyperparameters.csv")

        return self


def main():
    print("\n")
    print("=" * 70)
    print("   CRPA碳青霉烯耐药预测 - 模型解释 + 补充分析 (修订版)")
    print("=" * 70)
    print("  新增: HL检验, 敏感性分析 [SE-7], SHAP标注 [SE-3], 可重复性 [SE-9]")
    print("\n")

    train_path = './final_train_data.csv'
    val_path = './final_val_data.csv'

    # 自动选取测试集 AUROC 最优模型
    try:
        perf_df = pd.read_csv('./model_performance.csv')
        best_model_name = perf_df.sort_values('AUC', ascending=False).iloc[0]['Model']
        print(f"✓ 自动选取最优模型: {best_model_name} "
              f"(AUROC={perf_df.sort_values('AUC', ascending=False).iloc[0]['AUC']:.4f})")
    except Exception:
        best_model_name = 'XGBoost'
        print(f"⚠ 未找到 model_performance.csv，默认使用 {best_model_name}")

    interpreter = ModelInterpreter(train_path, val_path)

    (interpreter
     .load_data()
     .load_best_model(model_name=best_model_name)
     .plot_calibration_curve_all_models()
     .hosmer_lemeshow_test()                        # [额外]
     .plot_decision_curve_analysis()
     .shap_global_analysis()
     .shap_individual_cases(n_cases=2)
     # .sensitivity_analysis_missing_vars()         # [SE-7] 已移除：见问题5
     .reproducibility_summary())                    # [SE-9]

    print("\n")
    print("=" * 70)
    print("  模型解释 + 补充分析全部完成！")
    print("=" * 70)
    print(f"""
  生成的图表 ({PLOT_DIR}/):
    07_calibration_curves.png         校准曲线
    08_decision_curve_analysis.png    DCA（标题已修正）
    09_shap_summary.png               SHAP摘要（标注log-odds）
    10_shap_bar.png                   SHAP条形图
    11_shap_case_*.png                SHAP个案

  生成的表格 ({TABLE_DIR}/):
    Suppl_Hosmer_Lemeshow.csv         HL检验
    Suppl_Software_Versions.csv       软件版本 [SE-9]
    Suppl_XGBoost_Hyperparameters.csv XGBoost超参数 [SE-9]
    Suppl_CatBoost_Hyperparameters.csv CatBoost超参数 [SE-9]
    """)


if __name__ == '__main__':
    main()

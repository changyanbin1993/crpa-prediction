#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRPA Prediction Model - Six ML Models Training + Additional Evaluation per Reviewer Requirements

"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('once')

from collections import OrderedDict
from scipy import stats

from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, classification_report,
                             confusion_matrix, brier_score_loss, log_loss,
                             precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
PLOT_DIR    = './plots'
TABLE_DIR   = './tables'
MODEL_DIR   = './models'

for d in [PLOT_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


class ModelTrainer:
    """Model Trainer Class"""

    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.models = {}
        self.predictions = {}
        self.results = []
        self.best_model_name = None

    def load_data(self):
        print("=" * 70)
        print("Step 1: Load Data")
        print("=" * 70)

        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)

        self.X_train = train_df.drop('CRPA', axis=1)
        self.y_train = train_df['CRPA']
        self.X_val = val_df.drop('CRPA', axis=1)
        self.y_val = val_df['CRPA']
        self.feature_names = list(self.X_train.columns)

        self.scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_val.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Training setCRPA positive rate: {self.y_train.mean():.2%}")
        print(f"Test setCRPA positive rate: {self.y_val.mean():.2%}")
        print(f"scale_pos_weight: {self.scale_pos_weight:.2f}")

        return self

    # ───────────────────────────────────
    # 六种模型训练（与原始基本一致）
    # ───────────────────────────────────

    def train_lasso(self):
        print("\n" + "=" * 70)
        print("Model 1: L1-Logistic Regression (LASSO)")
        print("=" * 70)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)

        lr = LogisticRegressionCV(
            penalty='l1', solver='saga', Cs=100,
            cv=5, max_iter=10000,
            random_state=RANDOM_SEED,
            class_weight='balanced',
            scoring='roc_auc',
            n_jobs=-1
        )
        lr.fit(X_train_scaled, self.y_train)
        print(f"✓ Training complete, Optimal C: {lr.C_[0]:.6f}")

        y_pred_proba = lr.predict_proba(X_val_scaled)[:, 1]

        self.models['LASSO'] = {'model': lr, 'scaler': scaler}
        self.predictions['LASSO'] = y_pred_proba
        return self

    def train_random_forest(self):
        print("\n" + "=" * 70)
        print("Model 2: Random Forest")
        print("=" * 70)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        print(f"✓ Training complete")

        self.models['Random Forest'] = {'model': rf}
        self.predictions['Random Forest'] = rf.predict_proba(self.X_val)[:, 1]
        return self

    def train_adaboost(self):
        print("\n" + "=" * 70)
        print("Model 3: AdaBoost")
        print("=" * 70)

        base = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
        ada = AdaBoostClassifier(estimator=base, n_estimators=200,
                                  learning_rate=1.0, algorithm='SAMME',
                                  random_state=RANDOM_SEED)
        ada.fit(self.X_train, self.y_train)
        print(f"✓ Training complete")

        self.models['AdaBoost'] = {'model': ada}
        self.predictions['AdaBoost'] = ada.predict_proba(self.X_val)[:, 1]
        return self

    def train_xgboost(self, use_gpu=True):
        print("\n" + "=" * 70)
        print(f"Model 4: XGBoost")
        print("=" * 70)

        device = 'cpu'
        if use_gpu:
            try:
                import subprocess
                if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0:
                    device = 'cuda'
                    print("✓ GPU detected")
            except Exception:
                pass

        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'scale_pos_weight': self.scale_pos_weight,
            'tree_method': 'hist', 'device': device, 'seed': RANDOM_SEED
        }

        # ──────────────────────────────────────────
        # [SE-8/SE-9] 保存超参数到 JSON
        # ──────────────────────────────────────────
        with open(f'{MODEL_DIR}/xgboost_hyperparameters.json', 'w') as f:
            json.dump(params, f, indent=2, default=str)
        print(f"✓ 超参数saved: {MODEL_DIR}/xgboost_hyperparameters.json")

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        # Step 1: 内部分割找最优迭代次数（避免Test set泄漏）
        X_inner, X_es, y_inner, y_es = train_test_split(
            self.X_train, self.y_train, test_size=0.2,
            random_state=RANDOM_SEED, stratify=self.y_train)
        dtrain_inner = xgb.DMatrix(X_inner, label=y_inner)
        des = xgb.DMatrix(X_es, label=y_es)
        model_es = xgb.train(params, dtrain_inner, num_boost_round=1000,
                              evals=[(des, 'eval')], early_stopping_rounds=50,
                              verbose_eval=False)
        best_iter = model_es.best_iteration
        print(f"  Internal validation best iteration: {best_iter}")

        # Step 2: 全Training set重训
        model = xgb.train(params, dtrain, num_boost_round=best_iter + 1,
                           verbose_eval=False)
        print(f"✓ Training complete, using {best_iter + 1} iterations")

        self.models['XGBoost'] = {'model': model}
        self.predictions['XGBoost'] = model.predict(dval)
        return self

    def train_lightgbm(self, use_gpu=True):
        print("\n" + "=" * 70)
        print("Model 5: LightGBM")
        print("=" * 70)

        device = 'cpu'
        if use_gpu:
            try:
                import subprocess
                if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0:
                    device = 'gpu'
            except Exception:
                pass

        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'scale_pos_weight': self.scale_pos_weight, 'device': device,
            'verbose': -1, 'seed': RANDOM_SEED
        }

        # Step 1: 内部分割找最优迭代次数
        X_inner, X_es, y_inner, y_es = train_test_split(
            self.X_train, self.y_train, test_size=0.2,
            random_state=RANDOM_SEED, stratify=self.y_train)
        train_data = lgb.Dataset(X_inner, label=y_inner)
        valid_data = lgb.Dataset(X_es, label=y_es, reference=train_data)

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        model_es = lgb.train(params, train_data, num_boost_round=1000,
                              valid_sets=[valid_data], callbacks=callbacks)
        best_iter = model_es.best_iteration
        print(f"  Internal validation best iteration: {best_iter}")

        # Step 2: 全Training set重训
        full_data = lgb.Dataset(self.X_train, label=self.y_train)
        model = lgb.train(params, full_data, num_boost_round=best_iter)
        print(f"✓ Training complete, using {best_iter} iterations")

        self.models['LightGBM'] = {'model': model}
        self.predictions['LightGBM'] = model.predict(self.X_val,
                                                       num_iteration=model.best_iteration)
        return self

    def train_catboost(self, use_gpu=True):
        print("\n" + "=" * 70)
        print("Model 6: CatBoost")
        print("=" * 70)

        task_type = 'CPU'
        if use_gpu:
            try:
                import subprocess
                if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0:
                    task_type = 'GPU'
            except Exception:
                pass

        cb_params = dict(
            depth=6, learning_rate=0.05, l2_leaf_reg=3,
            scale_pos_weight=self.scale_pos_weight, task_type=task_type,
            devices='0', random_seed=RANDOM_SEED, eval_metric='AUC')

        # Step 1: 内部分割找最优迭代次数
        X_inner, X_es, y_inner, y_es = train_test_split(
            self.X_train, self.y_train, test_size=0.2,
            random_state=RANDOM_SEED, stratify=self.y_train)
        model_es = CatBoostClassifier(iterations=1000, early_stopping_rounds=50,
                                       verbose=0, **cb_params)
        model_es.fit(X_inner, y_inner, eval_set=(X_es, y_es), plot=False)
        best_iter = model_es.best_iteration_
        print(f"  Internal validation best iteration: {best_iter}")

        # Step 2: 全Training set重训
        cb = CatBoostClassifier(iterations=best_iter, verbose=100, **cb_params)
        cb.fit(self.X_train, self.y_train, plot=False)
        print(f"✓ Training complete, using {best_iter} iterations")

        # [SE-8/SE-9] 保存CatBoost超参数到JSON
        catboost_params = {**cb_params, 'iterations': best_iter,
                          'best_iteration_from_es': best_iter}
        with open(f'{MODEL_DIR}/catboost_hyperparameters.json', 'w') as f:
            json.dump(catboost_params, f, indent=2, default=str)
        print(f"✓ 超参数saved: {MODEL_DIR}/catboost_hyperparameters.json")

        self.models['CatBoost'] = {'model': cb}
        self.predictions['CatBoost'] = cb.predict_proba(self.X_val)[:, 1]
        return self

    # ───────────────────────────────────
    # 模型评估（with 95% CI）
    # ───────────────────────────────────

    def bootstrap_ci(self, y_true, y_pred_proba, threshold, n_bootstrap=1000):
        n = len(y_true)
        metrics_boot = {k: [] for k in
                        ['auc', 'sensitivity', 'specificity', 'precision',
                         'npv', 'f1', 'brier', 'logloss']}
        rng = np.random.default_rng(RANDOM_SEED)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            yt = y_true.iloc[idx].values if hasattr(y_true, 'iloc') else y_true[idx]
            yp = y_pred_proba[idx]
            yc = (yp >= threshold).astype(int)
            if len(np.unique(yt)) < 2:
                continue
            try:
                tn = ((yt == 0) & (yc == 0)).sum()
                fp = ((yt == 0) & (yc == 1)).sum()
                fn = ((yt == 1) & (yc == 0)).sum()
                tp = ((yt == 1) & (yc == 1)).sum()
                sens = tp / (tp + fn) if (tp + fn) else 0
                spec = tn / (tn + fp) if (tn + fp) else 0
                prec = tp / (tp + fp) if (tp + fp) else 0
                npv = tn / (tn + fn) if (tn + fn) else 0
                f1v = 2 * prec * sens / (prec + sens) if (prec + sens) else 0

                metrics_boot['auc'].append(roc_auc_score(yt, yp))
                metrics_boot['sensitivity'].append(sens)
                metrics_boot['specificity'].append(spec)
                metrics_boot['precision'].append(prec)
                metrics_boot['npv'].append(npv)
                metrics_boot['f1'].append(f1v)
                metrics_boot['brier'].append(brier_score_loss(yt, yp))
                metrics_boot['logloss'].append(log_loss(yt, np.clip(yp, 1e-15, 1 - 1e-15)))
            except Exception:
                continue

        ci = {}
        for k, v in metrics_boot.items():
            if v:
                ci[k] = (np.percentile(v, 2.5), np.percentile(v, 97.5))
            else:
                ci[k] = (np.nan, np.nan)
        return ci

    def evaluate_all_models(self, n_bootstrap=1000):
        print("\n" + "=" * 70)
        print("Step 2: Evaluate All Models（with 95% CI）")
        print("=" * 70)

        for model_name, y_pred_proba in self.predictions.items():
            print(f"\n--- {model_name} ---")
            auc = roc_auc_score(self.y_val, y_pred_proba)
            fpr, tpr, thresholds = roc_curve(self.y_val, y_pred_proba)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            y_pred = (y_pred_proba >= best_thresh).astype(int)

            tn = ((self.y_val == 0) & (y_pred == 0)).sum()
            fp = ((self.y_val == 0) & (y_pred == 1)).sum()
            fn = ((self.y_val == 1) & (y_pred == 0)).sum()
            tp = ((self.y_val == 1) & (y_pred == 1)).sum()

            prec = precision_score(self.y_val, y_pred)
            sens = recall_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            spec = tn / (tn + fp) if (tn + fp) else 0
            npv = tn / (tn + fn) if (tn + fn) else 0
            brier = brier_score_loss(self.y_val, y_pred_proba)
            ll = log_loss(self.y_val, y_pred_proba)

            ci = self.bootstrap_ci(self.y_val, y_pred_proba, best_thresh, n_bootstrap)

            print(f"AUROC: {auc:.4f} ({ci['auc'][0]:.4f}-{ci['auc'][1]:.4f})")
            print(f"Threshold: {best_thresh:.4f}")
            print(f"Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
            print(f"PPV: {prec:.4f}, NPV: {npv:.4f}")
            print(f"F1: {f1:.4f}, Brier: {brier:.4f}")

            self.results.append({
                'Model': model_name, 'AUC': auc,
                'AUC_95CI': f"({ci['auc'][0]:.4f}-{ci['auc'][1]:.4f})",
                'Sensitivity': sens,
                'Sensitivity_95CI': f"({ci['sensitivity'][0]:.4f}-{ci['sensitivity'][1]:.4f})",
                'Specificity': spec,
                'Specificity_95CI': f"({ci['specificity'][0]:.4f}-{ci['specificity'][1]:.4f})",
                'Precision': prec,
                'Precision_95CI': f"({ci['precision'][0]:.4f}-{ci['precision'][1]:.4f})",
                'NPV': npv,
                'NPV_95CI': f"({ci['npv'][0]:.4f}-{ci['npv'][1]:.4f})",
                'F1': f1,
                'F1_95CI': f"({ci['f1'][0]:.4f}-{ci['f1'][1]:.4f})",
                'Brier': brier,
                'Brier_95CI': f"({ci['brier'][0]:.4f}-{ci['brier'][1]:.4f})",
                'Threshold': best_thresh
            })

        results_df = pd.DataFrame(self.results).sort_values('AUC', ascending=False)
        results_df.to_csv('./model_performance.csv', index=False)
        print(f"\n✓ 性能表saved: ./model_performance.csv")

        self.best_model_name = results_df.iloc[0]['Model']
        print(f"✓ Best model: {self.best_model_name} (AUROC={results_df.iloc[0]['AUC']:.4f})")
        return self

    def plot_roc_curves(self):
        print("\nPlotting ROC curves...")
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for (name, yp), c in zip(self.predictions.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_val, yp)
            auc = roc_auc_score(self.y_val, yp)
            plt.plot(fpr, tpr, color=c, lw=2, label=f'{name} (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('1 - Specificity', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.title('ROC Curves — Six ML Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/06_roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ {PLOT_DIR}/06_roc_curves.png")
        plt.close()
        return self

    # ═══════════════════════════════════════
    # [SE-5] Precision-Recall 曲线 + AUPRC
    # ═══════════════════════════════════════

    def plot_pr_curves(self):
        """[SE-5] 绘制PR curves → Supplementary Figure S4"""
        print("\n" + "=" * 70)
        print("[SE-5] Precision-Recall 曲线 + AUPRC")
        print("=" * 70)

        prevalence = self.y_val.mean()
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        rng_pr = np.random.default_rng(RANDOM_SEED)
        auprc_rows = []
        for (name, yp), c in zip(self.predictions.items(), colors):
            pr, rc, _ = precision_recall_curve(self.y_val, yp)
            auprc = average_precision_score(self.y_val, yp)

            boots = []
            for _ in range(N_BOOTSTRAP):
                idx = rng_pr.choice(len(self.y_val), len(self.y_val), replace=True)
                try:
                    boots.append(average_precision_score(self.y_val.iloc[idx], yp[idx]))
                except Exception:
                    continue
            ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

            ax.plot(rc, pr, color=c, lw=2,
                    label=f'{name} (AUPRC={auprc:.3f}, {ci_lo:.3f}-{ci_hi:.3f})')
            auprc_rows.append({'Model': name, 'AUPRC': f"{auprc:.4f}",
                               '95CI_lower': f"{ci_lo:.4f}", '95CI_upper': f"{ci_hi:.4f}"})

        ax.axhline(y=prevalence, color='gray', ls='--', lw=1.5,
                   label=f'Baseline (prev={prevalence:.3f})')
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/Suppl_Fig_S4_PR_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ {PLOT_DIR}/Suppl_Fig_S4_PR_curves.png")
        plt.close()

        pd.DataFrame(auprc_rows).to_csv(f'{TABLE_DIR}/Suppl_AUPRC.csv', index=False)
        print(f"✓ {TABLE_DIR}/Suppl_AUPRC.csv")
        return self

    # ═══════════════════════════════════════
    # [SE-4/SE-8] 交叉验证
    # ═══════════════════════════════════════

    def cross_validation_analysis(self, n_splits=10, n_repeats=5):
        """[SE-4/SE-8] 10折重复交叉验证"""
        print("\n" + "=" * 70)
        print(f"[SE-4/SE-8] {n_splits}折重复交叉验证 ({n_repeats}次重复)")
        print("=" * 70)

        full_X = pd.concat([self.X_train, self.X_val], ignore_index=True)
        full_y = pd.concat([self.y_train, self.y_val], ignore_index=True)
        spw = (full_y == 0).sum() / (full_y == 1).sum()

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                      random_state=RANDOM_SEED)

        model_defs = OrderedDict({
            'XGBoost': xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
                use_label_encoder=False, random_state=RANDOM_SEED, verbosity=0),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced',
                random_state=RANDOM_SEED, n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED),
                n_estimators=200, algorithm='SAMME', random_state=RANDOM_SEED),
            'LASSO': Pipeline([('scaler', StandardScaler()),
                               ('lr', LogisticRegression(penalty='l1', solver='saga',
                                                          max_iter=10000,
                                                          class_weight='balanced',
                                                          random_state=RANDOM_SEED))]),
        })
        try:
            model_defs['LightGBM'] = lgb.LGBMClassifier(
                objective='binary', num_leaves=31, max_depth=6, learning_rate=0.05,
                n_estimators=300, scale_pos_weight=spw, verbose=-1,
                random_state=RANDOM_SEED)
        except Exception:
            pass
        try:
            model_defs['CatBoost'] = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                scale_pos_weight=spw, random_seed=RANDOM_SEED, verbose=0)
        except Exception:
            pass

        cv_results = {}
        for name, model in model_defs.items():
            print(f"  {name}...", end=' ', flush=True)
            scores = cross_val_score(model, full_X, full_y, cv=cv,
                                      scoring='roc_auc', n_jobs=-1)
            cv_results[name] = scores
            print(f"AUROC = {scores.mean():.4f} ± {scores.std():.4f}")

        # 保存
        rows = []
        for name, scores in cv_results.items():
            rows.append({'Model': name,
                         'Mean_AUROC': f"{scores.mean():.4f}",
                         'SD': f"{scores.std():.4f}",
                         'Min': f"{scores.min():.4f}",
                         'Max': f"{scores.max():.4f}"})
        pd.DataFrame(rows).to_csv(f'{TABLE_DIR}/Suppl_CV_Summary.csv', index=False)
        print(f"\n✓ {TABLE_DIR}/Suppl_CV_Summary.csv")

        # 箱线图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([cv_results[n] for n in cv_results], labels=list(cv_results.keys()),
                   patch_artist=True, showmeans=True)
        ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax.set_title(f'Repeated {n_splits}-Fold CV ({n_repeats} repeats)',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/S_cv_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"✓ {PLOT_DIR}/S_cv_boxplot.png")
        plt.close()

        return self

    # ═══════════════════════════════════════
    # [SE-6] 多阈值性能表
    # ═══════════════════════════════════════

    def threshold_performance_table(self):
        """[SE-6] 不同阈值下Best model性能 → Suppl Table S5"""
        print("\n" + "=" * 70)
        print("[SE-6] 多阈值性能指标表")
        print("=" * 70)

        yp = self.predictions.get(self.best_model_name)
        if yp is None:
            print(f"  ✗ {self.best_model_name}未找到"); return self

        fpr, tpr, roc_t = roc_curve(self.y_val, yp)
        best_t = roc_t[np.argmax(tpr - fpr)]

        thresholds = sorted(set([0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                                  0.50, round(best_t, 3), 0.55, 0.60,
                                  0.65, 0.70, 0.75, 0.80]))
        rows = []
        for t in thresholds:
            yc = (yp >= t).astype(int)
            tp = ((self.y_val == 1) & (yc == 1)).sum()
            tn = ((self.y_val == 0) & (yc == 0)).sum()
            fp = ((self.y_val == 0) & (yc == 1)).sum()
            fn = ((self.y_val == 1) & (yc == 0)).sum()
            sens = tp / (tp + fn) if (tp + fn) else 0
            spec = tn / (tn + fp) if (tn + fp) else 0
            ppv = tp / (tp + fp) if (tp + fp) else 0
            npv = tn / (tn + fn) if (tn + fn) else 0
            f1v = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0
            note = '← Youden' if abs(t - best_t) < 0.005 else ''
            rows.append({'Threshold': f"{t:.3f}", 'Sensitivity': f"{sens:.3f}",
                         'Specificity': f"{spec:.3f}", 'PPV': f"{ppv:.3f}",
                         'NPV': f"{npv:.3f}", 'F1': f"{f1v:.3f}", 'Note': note})

        df = pd.DataFrame(rows)
        df.to_csv(f'{TABLE_DIR}/Suppl_Table_S5_Thresholds.csv', index=False)
        print(df.to_string(index=False))
        print(f"\n✓ {TABLE_DIR}/Suppl_Table_S5_Thresholds.csv")
        return self

    # ═══════════════════════════════════════
    # [R1-3] 不同 prevalence 下 PPV/NPV
    # ═══════════════════════════════════════

    def prevalence_adjusted_ppv_npv(self):
        """[R1-3] 基于 Bayes 定理重算不同 prevalence 下预测值"""
        print("\n" + "=" * 70)
        print("[R1-3] 不同 prevalence 下 PPV/NPV (Bayes 定理)")
        print("=" * 70)

        yp = self.predictions.get(self.best_model_name)
        if yp is None:
            print(f"  ✗ {self.best_model_name}未找到"); return self

        fpr, tpr, t = roc_curve(self.y_val, yp)
        idx = np.argmax(tpr - fpr)
        sens, spec = tpr[idx], 1 - fpr[idx]
        study_prev = self.y_val.mean()

        prevs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                 round(study_prev, 3), 0.40, 0.50]
        rows = []
        for p in prevs:
            ppv = (sens * p) / (sens * p + (1 - spec) * (1 - p))
            npv = (spec * (1 - p)) / (spec * (1 - p) + (1 - sens) * p)
            note = '← this study' if abs(p - study_prev) < 0.01 else ''
            rows.append({'Prevalence': f"{p:.1%}", 'PPV': f"{ppv:.3f}",
                         'NPV': f"{npv:.3f}", 'Note': note})

        df = pd.DataFrame(rows)
        df.to_csv(f'{TABLE_DIR}/Suppl_Table_S6_Prevalence.csv', index=False)
        print(df.to_string(index=False))
        print(f"\n✓ {TABLE_DIR}/Suppl_Table_S6_Prevalence.csv")
        return self

    # ═══════════════════════════════════════
    # [R1-5] DeLong 检验
    # ═══════════════════════════════════════

    def delong_test(self):
        """[R1-5] DeLong 检验比较各模型 AUROC"""
        print("\n" + "=" * 70)
        print("[R1-5] DeLong 检验（模型间 AUROC 比较）")
        print("=" * 70)

        ref = self.best_model_name
        if ref not in self.predictions:
            ref = list(self.predictions.keys())[0]

        aucs = {n: roc_auc_score(self.y_val, p) for n, p in self.predictions.items()}
        y_arr = self.y_val.values

        rows = []
        for name in self.predictions:
            if name == ref:
                continue
            z, p = self._delong(y_arr, self.predictions[ref], self.predictions[name])
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'NS'))
            print(f"  {ref} vs {name:18s}: ΔAUC={aucs[ref]-aucs[name]:+.4f}, "
                  f"Z={z:.3f}, P={p:.4f} {sig}")
            rows.append({
                'Comparison': f"{ref} vs {name}",
                'AUC_ref': f"{aucs[ref]:.4f}", 'AUC_other': f"{aucs[name]:.4f}",
                'Delta_AUC': f"{aucs[ref]-aucs[name]:+.4f}",
                'Z': f"{z:.3f}", 'P_value': f"{p:.4f}", 'Sig': sig})

        pd.DataFrame(rows).to_csv(f'{TABLE_DIR}/Suppl_Table_S7_DeLong.csv', index=False)
        print(f"\n✓ {TABLE_DIR}/Suppl_Table_S7_DeLong.csv")
        return self

    @staticmethod
    def _delong(y_true, score1, score2):
        """DeLong test (Sun & Xu 2014 fast algorithm)"""
        n1 = (y_true == 1).sum()
        n0 = (y_true == 0).sum()

        def midrank(x):
            J = np.argsort(x)
            Z = x[J]
            N = len(x)
            T = np.zeros(N, dtype=float)
            i = 0
            while i < N:
                j = i
                while j < N and Z[j] == Z[i]:
                    j += 1
                for k in range(i, j):
                    T[k] = 0.5 * (i + j - 1)
                i = j
            T2 = np.empty(N, dtype=float)
            T2[J] = T + 1
            return T2

        pos1, neg1 = score1[y_true == 1], score1[y_true == 0]
        pos2, neg2 = score2[y_true == 1], score2[y_true == 0]

        all1 = np.concatenate([pos1, neg1])
        all2 = np.concatenate([pos2, neg2])
        mr1, mr2 = midrank(all1), midrank(all2)

        v01_1 = (mr1[:n1] - np.arange(1, n1 + 1)) / n0
        v01_2 = (mr2[:n1] - np.arange(1, n1 + 1)) / n0
        v10_1 = 1.0 - (mr1[n1:] - np.arange(1, n0 + 1)) / n1
        v10_2 = 1.0 - (mr2[n1:] - np.arange(1, n0 + 1)) / n1

        s01_1 = np.var(v01_1, ddof=1) if n1 > 1 else 0
        s01_2 = np.var(v01_2, ddof=1) if n1 > 1 else 0
        s10_1 = np.var(v10_1, ddof=1) if n0 > 1 else 0
        s10_2 = np.var(v10_2, ddof=1) if n0 > 1 else 0
        cov01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] if n1 > 1 else 0
        cov10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] if n0 > 1 else 0

        var_diff = (s01_1 + s01_2 - 2 * cov01) / n1 + (s10_1 + s10_2 - 2 * cov10) / n0
        if var_diff <= 0:
            return 0.0, 1.0

        auc1 = (mr1[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc2 = (mr2[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

        z = (auc1 - auc2) / np.sqrt(var_diff)
        return z, 2 * stats.norm.sf(abs(z))

    # ───────────────────────────────────
    # 保存模型
    # ───────────────────────────────────

    def save_models(self):
        print("\n" + "=" * 70)
        print("Save All Models")
        print("=" * 70)

        # [修复] 保存LASSO模型
        if 'LASSO' in self.models:
            joblib.dump(self.models['LASSO'], f'{MODEL_DIR}/lasso_model.pkl')
            print(f"✓ LASSO → {MODEL_DIR}/lasso_model.pkl")

        for name, d in self.models.items():
            if name == 'LASSO':
                continue
            m = d['model']
            if name == 'XGBoost':
                m.save_model(f'{MODEL_DIR}/xgboost_model.json')
                print(f"✓ XGBoost → {MODEL_DIR}/xgboost_model.json")
            elif name == 'LightGBM':
                m.save_model(f'{MODEL_DIR}/lightgbm_model.txt')
                print(f"✓ LightGBM → {MODEL_DIR}/lightgbm_model.txt")
            elif name == 'CatBoost':
                m.save_model(f'{MODEL_DIR}/catboost_model.cbm')
                print(f"✓ CatBoost → {MODEL_DIR}/catboost_model.cbm")
            else:
                path = f'{MODEL_DIR}/{name.lower().replace(" ", "_")}_model.pkl'
                joblib.dump(d, path)
                print(f"✓ {name} → {path}")

        return self


def main():
    print("\n")
    print("=" * 70)
    print("   CRPA Prediction Model - Training + Additional Evaluation (Revised)")
    print("=" * 70)
    print("  Added: CV [SE-4], PR curves [SE-5],")
    print("        Threshold table [SE-6], Prevalence [R1-3], DeLong [R1-5]")
    print("\n")

    train_path = './final_train_data.csv'
    val_path = './final_val_data.csv'
    use_gpu = True

    trainer = ModelTrainer(train_path, val_path)
    trainer.load_data()

    # 训练6种模型
    (trainer
     .train_lasso()
     .train_random_forest()
     .train_adaboost()
     .train_xgboost(use_gpu=use_gpu)
     .train_lightgbm(use_gpu=use_gpu)
     .train_catboost(use_gpu=use_gpu))

    # 原始评估
    (trainer
     .evaluate_all_models(n_bootstrap=N_BOOTSTRAP)
     .plot_roc_curves()
     .save_models())

    # ── 审稿人要求的补充分析 ──
    (trainer
     .plot_pr_curves()                    # [SE-5]
     .cross_validation_analysis()         # [SE-4/SE-8]
     .threshold_performance_table()       # [SE-6]
     .prevalence_adjusted_ppv_npv()       # [R1-3]
     .delong_test())                      # [R1-5]

    print("\n")
    print("=" * 70)
    print("  Model Training + Additional Evaluation Complete!")
    print("=" * 70)
    print(f"\n  Supplementary materials output directory: {TABLE_DIR}/")
    print("  Next step: Run 04_model_interpretation.py")
    print("\n")


if __name__ == '__main__':
    main()

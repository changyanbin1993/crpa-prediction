import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('once')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

RANDOM_SEED = 42


class FeatureSelector:
    """特征选择类"""

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.selected_features = None

    def load_data(self):
        """加载01脚本已分割好的训练集和测试集"""
        print("=" * 70)
        print("步骤1: 加载预处理后的训练/测试数据")
        print("=" * 70)

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        self.X_train = train_df.drop('CRPA', axis=1)
        self.y_train = train_df['CRPA']
        self.X_test = test_df.drop('CRPA', axis=1)
        self.y_test = test_df['CRPA']
        self.feature_names = list(self.X_train.columns)

        print(f"训练集: {self.X_train.shape}")
        print(f"测试集: {self.X_test.shape}")
        print(f"特征数: {len(self.feature_names)}")
        print(f"训练集CRPA阳性率: {self.y_train.mean():.2%}")
        print(f"测试集CRPA阳性率: {self.y_test.mean():.2%}")

        return self

    def lasso_selection(self, n_alphas=100):
        """方法A: LASSO特征选择（候选池1）"""
        print("\n" + "=" * 70)
        print("步骤2A: LASSO特征选择 [候选池1]")
        print("=" * 70)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)

        alphas = np.logspace(-4, 1, n_alphas)
        lasso = LassoCV(
            alphas=alphas, cv=10, max_iter=10000,
            random_state=RANDOM_SEED, n_jobs=-1
        )
        lasso.fit(X_train_scaled, self.y_train)

        print(f"\n最优alpha: {lasso.alpha_:.6f}")

        coef = pd.DataFrame({
            '特征': self.feature_names,
            '系数': lasso.coef_
        })
        lasso_features = coef[coef['系数'] != 0].sort_values(
            '系数', key=abs, ascending=False)

        print(f"LASSO选择的特征数: {len(lasso_features)}")
        print("\nLASSO特征及系数（前20）:")
        print(lasso_features.head(20))

        # ── 可视化 ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        mean_mse = lasso.mse_path_.mean(axis=1)
        std_mse = lasso.mse_path_.std(axis=1)
        axes[0].semilogx(lasso.alphas_, mean_mse, 'b-', lw=2, label='Mean CV MSE')
        axes[0].fill_between(lasso.alphas_, mean_mse - std_mse, mean_mse + std_mse,
                             alpha=0.2, color='blue')
        axes[0].axvline(lasso.alpha_, color='red', linestyle='--', lw=2,
                        label=f'Optimal α = {lasso.alpha_:.4f}')
        axes[0].set_xlabel('Alpha', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MSE', fontsize=12, fontweight='bold')
        axes[0].set_title('LASSO Cross-Validation (10-fold)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        from sklearn.linear_model import lasso_path
        alphas_path, coefs_path, _ = lasso_path(
            X_train_scaled, self.y_train,
            alphas=np.logspace(-4, 1, 50), max_iter=10000)
        final_coef_abs = np.abs(lasso.coef_)
        top_idx = np.argsort(final_coef_abs)[-20:][::-1]
        for idx in top_idx:
            if final_coef_abs[idx] > 0:
                axes[1].semilogx(alphas_path, coefs_path[idx, :], lw=1.5, alpha=0.7)
        axes[1].axvline(lasso.alpha_, color='red', linestyle='--', lw=2)
        axes[1].set_xlabel('Alpha', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Coefficients', fontsize=12, fontweight='bold')
        axes[1].set_title('LASSO Coefficient Path', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', lw=0.5)

        top20 = lasso_features.head(20)
        colors = ['green' if x > 0 else 'red' for x in top20['系数']]
        axes[2].barh(range(len(top20)), top20['系数'], color=colors, alpha=0.7)
        axes[2].set_yticks(range(len(top20)))
        axes[2].set_yticklabels(top20['特征'], fontsize=9)
        axes[2].set_xlabel('LASSO Coefficient', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Selected Features (n={len(lasso_features)})',
                          fontsize=14, fontweight='bold')
        axes[2].axvline(x=0, color='black', lw=1)
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('./plots/02a_lasso_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ 保存图表: ./plots/02a_lasso_analysis.png")
        plt.close()

        return lasso_features['特征'].tolist()

    def rfe_selection(self, min_features=5):
        """方法B: RFE特征选择（候选池2）"""
        print("\n" + "=" * 70)
        print("步骤2B: RFE特征选择 [候选池2]（使用Random Forest）")
        print("=" * 70)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')

        rfecv = RFECV(
            estimator=rf, step=1, cv=5,
            scoring='roc_auc', min_features_to_select=min_features, n_jobs=-1)

        print("训练RFE模型...")
        rfecv.fit(self.X_train, self.y_train)

        print(f"\n✓ RFE训练完成")
        print(f"最优特征数: {rfecv.n_features_}")
        print(f"最优交叉验证AUC: {rfecv.cv_results_['mean_test_score'].max():.4f}")

        rfe_features = [feat for feat, sel in zip(self.feature_names, rfecv.support_) if sel]
        print(f"\nRFE选择的特征:")
        for i, feat in enumerate(rfe_features, 1):
            print(f"  {i}. {feat}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(min_features, len(rfecv.cv_results_['mean_test_score']) + min_features),
                 rfecv.cv_results_['mean_test_score'], 'o-', lw=2, markersize=5)
        plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', lw=2,
                    label=f'Optimal n={rfecv.n_features_}')
        plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
        plt.ylabel('Cross-Validation AUC', fontsize=12, fontweight='bold')
        plt.title('RFE Cross-Validation Curve (5-fold)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./plots/03_rfe_curve.png', dpi=300, bbox_inches='tight')
        print("✓ 保存图表: ./plots/03_rfe_curve.png")
        plt.close()

        return rfe_features

    def shap_selection(self, top_k=30):
        """
        方法C: SHAP特征重要性排名
        ─────────────────────────────────────
        SHAP-rank 作为三个并列候选池之一（与 LASSO、RFE 并列），
        三池取并集后进入 Bootstrap 稳定性过滤。
        """
        print("\n" + "=" * 70)
        print("步骤2C: SHAP特征重要性排名 [第三候选池]")
        print("=" * 70)
        print("  SHAP-rank 与 LASSO、RFE 并列作为候选池，")
        print("         三池并集进入 Bootstrap 稳定性过滤。")

        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'verbose': -1, 'seed': RANDOM_SEED
        }

        print("\n训练LightGBM模型...")
        model = lgb.train(
            params, train_data, num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

        try:
            import shap
            print("计算SHAP值...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_train)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                '特征': self.feature_names,
                'Mean_Abs_SHAP': mean_abs_shap
            }).sort_values('Mean_Abs_SHAP', ascending=False)

            print(f"\nSHAP特征重要性（前20）:")
            print(shap_importance.head(20))

            shap_features = shap_importance.head(top_k)['特征'].tolist()
            print(f"\nSHAP排名前{top_k}特征数: {len(shap_features)}")

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[:500], self.X_train.iloc[:500],
                              max_display=20, show=False)
            plt.tight_layout()
            plt.savefig('./plots/04_shap_summary_feature_selection.png',
                        dpi=300, bbox_inches='tight')
            print("✓ 保存图表: ./plots/04_shap_summary_feature_selection.png")
            plt.close()

            return shap_features

        except ImportError:
            print("\n⚠ 未安装SHAP库，跳过SHAP验证")
            return []

    def bootstrap_stability(self, n_bootstrap=100, stability_threshold=0.93):
        """
        步骤3: Bootstrap稳定性验证
        ─────────────────────────────────
        [SE-3] 三种方法并列作为候选池，取并集后进入 Bootstrap 过滤
        """
        print("\n" + "=" * 70)
        print(f"步骤3: Bootstrap稳定性验证（{n_bootstrap}次重采样）")
        print("=" * 70)

        # 获取三种方法的特征
        lasso_features = self.lasso_selection()
        rfe_features = self.rfe_selection()
        shap_features = self.shap_selection()

        # ──────────────────────────────────────────
        # 特征选择方法交叠分析
        # ──────────────────────────────────────────
        lasso_set = set(lasso_features)
        rfe_set = set(rfe_features)
        shap_set = set(shap_features)

        print(f"\n{'='*50}")
        print(f"  特征选择方法交叠分析")
        print(f"{'='*50}")
        print(f"  LASSO（候选池1）选出:        {len(lasso_set)} 个")
        print(f"  RFE（候选池2）选出:          {len(rfe_set)} 个")
        print(f"  SHAP（第三候选池）排名前列:  {len(shap_set)} 个")
        print(f"  ─────────────────────────────────")
        print(f"  LASSO ∩ RFE（双主力交集）:   {len(lasso_set & rfe_set)} 个")
        print(f"  三者交集:                    {len(lasso_set & rfe_set & shap_set)} 个")
        print(f"  三者并集（候选池）:          {len(lasso_set | rfe_set | shap_set)} 个")

        # 保存交叠表
        all_feats = sorted(lasso_set | rfe_set | shap_set)
        overlap_df = pd.DataFrame({
            'Feature': all_feats,
            'LASSO': [int(f in lasso_set) for f in all_feats],
            'RFE': [int(f in rfe_set) for f in all_feats],
            'SHAP': [int(f in shap_set) for f in all_feats],
        })
        overlap_df['N_Methods'] = overlap_df[['LASSO', 'RFE', 'SHAP']].sum(axis=1)
        overlap_df = overlap_df.sort_values('N_Methods', ascending=False)
        overlap_df.to_csv('./feature_selection_overlap.csv', index=False)
        print(f"\n✓ 特征交叠分析已保存: ./feature_selection_overlap.csv")

        # 合并候选特征
        candidate_features = list(set(lasso_features + rfe_features + shap_features))
        candidate_features = [f for f in candidate_features if f in self.feature_names]
        print(f"\n候选特征总数: {len(candidate_features)}")

        X_train_candidates = self.X_train[candidate_features]

        # Bootstrap重采样
        feature_selection_counts = Counter()
        print(f"\n开始Bootstrap重采样（{n_bootstrap}次）...")

        rng = np.random.default_rng(RANDOM_SEED)
        for i in range(n_bootstrap):
            if (i + 1) % 20 == 0:
                print(f"  完成 {i+1}/{n_bootstrap} 次重采样")

            indices = rng.choice(len(X_train_candidates),
                                 size=len(X_train_candidates), replace=True)
            X_boot = X_train_candidates.iloc[indices]
            y_boot = self.y_train.iloc[indices]

            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)

            lasso_boot = LassoCV(
                alphas=np.logspace(-4, 1, 50), cv=3,
                max_iter=5000, random_state=RANDOM_SEED + i)
            lasso_boot.fit(X_boot_scaled, y_boot)

            coef = lasso_boot.coef_
            selected = [feat for feat, c in zip(candidate_features, coef)
                        if c != 0]
            feature_selection_counts.update(selected)

        feature_stability = pd.DataFrame({
            '特征': list(feature_selection_counts.keys()),
            '选择次数': list(feature_selection_counts.values()),
            '选择频率': [c / n_bootstrap for c in feature_selection_counts.values()]
        }).sort_values('选择频率', ascending=False)

        print(f"\n特征稳定性统计（前20）:")
        print(feature_stability.head(20))

        stable_features = feature_stability[
            feature_stability['选择频率'] >= stability_threshold]['特征'].tolist()

        print(f"\n稳定特征数（选择频率 >= {stability_threshold}): {len(stable_features)}")

        if len(stable_features) == 0:
            raise ValueError(
                f"未找到选择频率 >= {stability_threshold:.2%} 的特征，请检查Bootstrap结果或调整阈值。"
            )

        # 保存稳定性结果
        feature_stability.to_csv('./feature_stability_results.csv', index=False)
        print(f"✓ 稳定性结果已保存: ./feature_stability_results.csv")

        # 绘图
        plt.figure(figsize=(12, 8))
        top20 = feature_stability.head(20)
        colors_bar = ['#2ca02c' if f >= stability_threshold else '#d62728'
                      for f in top20['选择频率']]
        plt.barh(range(len(top20)), top20['选择频率'], color=colors_bar)
        plt.yticks(range(len(top20)), top20['特征'])
        plt.xlabel('Selection Frequency', fontsize=12, fontweight='bold')
        plt.title(f'Feature Stability (Bootstrap={n_bootstrap}, threshold={stability_threshold})',
                  fontsize=14, fontweight='bold')
        plt.axvline(x=stability_threshold, color='r', linestyle='--',
                    lw=2, label=f'Threshold={stability_threshold}')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('./plots/05_feature_stability.png', dpi=300, bbox_inches='tight')
        print("✓ 保存图表: ./plots/05_feature_stability.png")
        plt.close()

        self.selected_features = stable_features
        return self

    def save_results(self):
        """保存特征选择结果"""
        print("\n" + "=" * 70)
        print("步骤4: 保存特征选择结果")
        print("=" * 70)

        with open('./selected_features.txt', 'w') as f:
            f.write("# 最终选择的特征\n")
            f.write(f"# 总数: {len(self.selected_features)}\n")
            f.write(f"# 方法: LASSO ∪ RFE ∪ SHAP-top-30 → Bootstrap稳定性过滤(≥0.93)\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i}. {feat}\n")
        print(f"\n✓ 特征列表已保存: ./selected_features.txt")

        # 保存最终数据集
        final_X_train = self.X_train[self.selected_features]
        final_X_test = self.X_test[self.selected_features]

        final_train = final_X_train.copy()
        final_train['CRPA'] = self.y_train.values
        final_test = final_X_test.copy()
        final_test['CRPA'] = self.y_test.values

        final_train.to_csv('./final_train_data.csv', index=False)
        final_test.to_csv('./final_val_data.csv', index=False)

        print(f"✓ 训练集已保存: ./final_train_data.csv")
        print(f"✓ 测试集已保存: ./final_val_data.csv")
        print(f"\n最终特征数: {len(self.selected_features)}")
        print(f"训练集: {final_train.shape}, 测试集: {final_test.shape}")

        # EPV计算 [SE-8]
        n_events = int(self.y_train.sum()) + int(self.y_test.sum())
        epv = n_events / len(self.selected_features)
        print(f"\n[SE-8] Events-Per-Variable (EPV) = {n_events} / {len(self.selected_features)} = {epv:.1f}")
        if epv >= 10:
            print(f"  ✓ EPV >= 10，满足常规最低要求")
        else:
            print(f"  ⚠ EPV < 10，过拟合风险较高")

        return self


def main():
    print("\n")
    print("=" * 70)
    print("   CRPA碳青霉烯耐药预测 - 特征选择 (修订版)")
    print("=" * 70)
    print("  LASSO ∪ RFE ∪ SHAP-top-30 → Bootstrap稳定性过滤")
    print("\n")

    # 读取01脚本输出的训练集/测试集
    train_path = './processed_train.csv'
    test_path = './processed_test.csv'

    selector = FeatureSelector(train_path, test_path)

    (selector
     .load_data()
     .bootstrap_stability(n_bootstrap=100, stability_threshold=0.93)
     .save_results())

    print("\n")
    print("=" * 70)
    print("  特征选择完成！")
    print("=" * 70)
    print("\n下一步: 运行 03_model_training.py 进行模型训练")
    print("\n")


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import platform
import warnings
warnings.filterwarnings('once')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import f_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────
# 全局配置
# ────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.2

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        # [SE-7] 新增: 记录被排除变量信息
        self.excluded_vars_info = None

    def load_data(self):
        """加载数据"""
        print("=" * 70)
        print("步骤1: 加载数据")
        print("=" * 70)

        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"数据形状: {self.df.shape}")
        print(f"\n目标变量 (CRPA) 分布:")
        print(self.df['CRPA'].value_counts())
        print(f"CRPA阳性率: {self.df['CRPA'].mean():.2%}")

        return self

    def check_data_quality(self):
        """检查数据质量"""
        print("\n" + "=" * 70)
        print("步骤2: 数据质量检查")
        print("=" * 70)

        # 缺失值统计
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            '缺失数量': missing[missing > 0],
            '缺失比例(%)': missing_pct[missing > 0]
        }).sort_values('缺失比例(%)', ascending=False)

        print("\n缺失值统计:")
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("✓ 无缺失值")

        # [SE-7] 新增: 保存完整缺失率表
        exclude_cols = ['Anchor Year Group', 'CRPA']
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        all_missing = pd.DataFrame({
            'Variable': feature_cols,
            'N_Missing': [self.df[c].isnull().sum() for c in feature_cols],
            'Missing_Rate(%)': [(self.df[c].isnull().sum() / len(self.df) * 100)
                                for c in feature_cols]
        }).sort_values('Missing_Rate(%)', ascending=False)
        all_missing.to_csv('./missing_rate_summary.csv', index=False)
        print(f"\n✓ 完整缺失率统计已保存: ./missing_rate_summary.csv")

        print("\n数值型特征描述性统计:")
        print(self.df.describe().T)

        return self

    def remove_high_missing_features(self, threshold=10):
        """删除缺失比例>threshold%的变量"""
        print("\n" + "=" * 70)
        print(f"步骤3: 删除缺失比例>{threshold}%的变量")
        print("=" * 70)

        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        exclude_cols = ['Anchor Year Group', 'CRPA']

        high_missing_features = []
        for col in self.df.columns:
            if col not in exclude_cols and missing_pct[col] > threshold:
                high_missing_features.append(col)

        if len(high_missing_features) > 0:
            print(f"\n⚠ 发现 {len(high_missing_features)} 个高缺失率变量 (>{threshold}%):")
            for feat in high_missing_features:
                print(f"  - {feat}: 缺失率 = {missing_pct[feat]:.2f}%")

            # ────────────────────────────────────────
            # [SE-7] 新增: 保存被排除变量信息到文件
            # ────────────────────────────────────────
            self.excluded_vars_info = pd.DataFrame({
                'Variable': high_missing_features,
                'Missing_Rate(%)': [round(missing_pct[f], 2) for f in high_missing_features]
            }).sort_values('Missing_Rate(%)', ascending=False)
            self.excluded_vars_info.to_csv('./excluded_high_missing_vars.csv', index=False)
            print(f"\n✓ 被排除变量信息已保存: ./excluded_high_missing_vars.csv")
            print("  （用于 Response Letter 中回复审稿意见 SE-7）")

            self.df = self.df.drop(columns=high_missing_features)
            print(f"\n✓ 已删除 {len(high_missing_features)} 个高缺失率变量")
            print(f"✓ 剩余变量数: {self.df.shape[1]}")
        else:
            print(f"\n✓ 没有缺失率>{threshold}%的变量")

        return self

    def split_and_impute(self):
        """
        [SE-2] 修改: 先分割训练集/测试集，再用训练集中位数填补
        ──────────────────────────────────────────────────────
        原始流程: 全量填补 → 再分割（存在数据泄漏风险）
        修改后:   先分割 → 仅用训练集拟合Imputer → 分别填补
        """
        print("\n" + "=" * 70)
        print("步骤4: 分割数据 → 中位数填补（防止数据泄漏）")
        print("=" * 70)

        # 分离特征和目标
        y = self.df['CRPA'].values
        exclude_cols = ['Anchor Year Group', 'CRPA']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        X_raw = self.df[feature_cols].copy()
        X_raw.columns = [col.strip() for col in X_raw.columns]

        print(f"\n特征数量: {X_raw.shape[1]}")
        print(f"样本数量: {X_raw.shape[0]}")

        # ──────────────────────────────────────────
        # 关键修改: 先分割再填补
        # ──────────────────────────────────────────
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        print(f"\n训练集: {X_train_raw.shape} (CRPA阳性率: {self.y_train.mean():.2%})")
        print(f"测试集: {X_test_raw.shape} (CRPA阳性率: {self.y_test.mean():.2%})")

        # 仅用训练集拟合中位数
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_raw)

        # 分别转换
        self.X_train = pd.DataFrame(
            imputer.transform(X_train_raw),
            columns=X_raw.columns, index=X_train_raw.index
        )
        self.X_test = pd.DataFrame(
            imputer.transform(X_test_raw),
            columns=X_raw.columns, index=X_test_raw.index
        )
        self.feature_names = list(self.X_train.columns)

        print(f"\n✓ 中位数仅从训练集计算（n={len(X_train_raw)}），防止数据泄漏")
        print(f"✓ 训练集和测试集分别填补完成")
        print(f"✓ 剩余缺失值 (训练): {self.X_train.isnull().sum().sum()}")
        print(f"✓ 剩余缺失值 (测试): {self.X_test.isnull().sum().sum()}")

        return self

    def check_collinearity(self, threshold=10):
        """检查多重共线性 - 仅在训练集上计算VIF"""
        print("\n" + "=" * 70)
        print(f"步骤5: 检查多重共线性（VIF阈值: {threshold}）")
        print("=" * 70)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.X_train.columns)

        vif_data = pd.DataFrame()
        vif_data["特征"] = self.X_train.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i)
                           for i in range(len(self.X_train.columns))]
        vif_data = vif_data.sort_values('VIF', ascending=False)

        print("\nVIF统计（前20）:")
        print(vif_data.head(20))

        high_vif = vif_data[vif_data['VIF'] > threshold]

        if len(high_vif) > 0:
            print(f"\n⚠ 发现 {len(high_vif)} 个高VIF特征 (VIF > {threshold}):")
            print(high_vif)

            features_to_drop = high_vif['特征'].tolist()
            self.X_train = self.X_train.drop(columns=features_to_drop)
            self.X_test = self.X_test.drop(columns=features_to_drop)
            self.feature_names = list(self.X_train.columns)

            print(f"\n✓ 已移除高VIF特征")
            print(f"✓ 剩余特征数: {len(self.feature_names)}")
        else:
            print(f"\n✓ 所有特征VIF < {threshold}")

        return self

    def check_low_variance(self, threshold=0.01):
        """检查并移除低方差特征（仅基于训练集）"""
        print("\n" + "=" * 70)
        print(f"步骤6: 检查低方差特征（阈值: {threshold}）")
        print("=" * 70)

        variances = self.X_train.var()
        low_var_features = variances[variances < threshold].index.tolist()

        if len(low_var_features) > 0:
            print(f"\n⚠ 发现 {len(low_var_features)} 个低方差特征:")
            for feat in low_var_features:
                print(f"  - {feat}: 方差 = {variances[feat]:.6f}")

            self.X_train = self.X_train.drop(columns=low_var_features)
            self.X_test = self.X_test.drop(columns=low_var_features)
            self.feature_names = list(self.X_train.columns)

            print(f"\n✓ 已移除低方差特征")
            print(f"✓ 剩余特征数: {len(self.feature_names)}")
        else:
            print(f"\n✓ 所有特征方差 > {threshold}")

        return self

    def univariate_feature_selection(self, k=50, p_threshold=0.1):
        """单变量特征筛选（粗筛）— 仅基于训练集"""
        print("\n" + "=" * 70)
        print(f"步骤7: 单变量特征筛选（保留前{k}个或p<{p_threshold}）")
        print("=" * 70)

        f_scores, p_values = f_classif(self.X_train, self.y_train)

        feature_scores = pd.DataFrame({
            '特征': self.feature_names,
            'F-score': f_scores,
            'p-value': p_values
        }).sort_values('F-score', ascending=False)

        print("\n特征重要性排序（前20）:")
        print(feature_scores.head(20))

        if len(self.feature_names) > k:
            selected_topk = feature_scores.head(k)['特征'].tolist()
            selected_pval = feature_scores[feature_scores['p-value'] < p_threshold]['特征'].tolist()
            selected_features = list(set(selected_topk) | set(selected_pval))

            print(f"\n✓ 从 {len(self.feature_names)} 个特征中筛选出 {len(selected_features)} 个候选特征")

            self.X_train = self.X_train[selected_features]
            self.X_test = self.X_test[selected_features]
            self.feature_names = selected_features
        else:
            print(f"\n✓ 特征数量({len(self.feature_names)})未超过阈值({k})，保留所有特征")

        return self

    def save_processed_data(self, train_path='./processed_train.csv',
                            test_path='./processed_test.csv'):
        """保存处理后的训练集和测试集"""
        print("\n" + "=" * 70)
        print("步骤8: 保存处理后的数据")
        print("=" * 70)

        train_df = self.X_train.copy()
        train_df['CRPA'] = self.y_train
        train_df.to_csv(train_path, index=False)

        test_df = self.X_test.copy()
        test_df['CRPA'] = self.y_test
        test_df.to_csv(test_path, index=False)

        print(f"\n✓ 训练集已保存至: {train_path}")
        print(f"✓ 测试集已保存至: {test_path}")
        print(f"✓ 最终特征数: {len(self.feature_names)}")
        print(f"✓ 训练集样本: {len(train_df)}, 测试集样本: {len(test_df)}")

        # 保存特征名称
        with open('./processed_data_features.txt', 'w') as f:
            for feat in self.feature_names:
                f.write(f"{feat}\n")
        print(f"✓ 特征名称已保存至: ./processed_data_features.txt")

        return self

    # ──────────────────────────────────────────
    # [SE-9] 新增: 记录软件环境信息
    # ──────────────────────────────────────────
    def log_environment(self):
        """记录软件版本和环境信息"""
        print("\n" + "=" * 70)
        print("步骤9: 记录软件环境 [SE-9]")
        print("=" * 70)

        import importlib
        info = {}
        info['Python'] = platform.python_version()
        info['Platform'] = platform.platform()
        info['Random_Seed'] = str(RANDOM_SEED)

        for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost',
                     'lightgbm', 'catboost', 'shap', 'matplotlib', 'seaborn',
                     'statsmodels']:
            try:
                mod = importlib.import_module(pkg)
                info[pkg] = getattr(mod, '__version__', 'unknown')
            except ImportError:
                info[pkg] = 'not installed'

        env_df = pd.DataFrame(list(info.items()), columns=['Package', 'Version'])
        env_df.to_csv('./software_environment.csv', index=False)

        print("\n  软件环境:")
        for _, row in env_df.iterrows():
            print(f"    {row['Package']:18s}: {row['Version']}")
        print(f"\n✓ 已保存至: ./software_environment.csv")

        return self

    def plot_summary(self, output_dir='./plots'):
        """绘制数据摘要图"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("步骤10: 生成可视化图表")
        print("=" * 70)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        pd.concat([pd.Series(self.y_train), pd.Series(self.y_test)]).value_counts().plot(
            kind='bar', ax=axes[0])
        axes[0].set_title('CRPA Distribution (Full Dataset)')
        axes[0].set_xlabel('CRPA Status')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['CSPA (0)', 'CRPA (1)'], rotation=0)

        pd.concat([pd.Series(self.y_train), pd.Series(self.y_test)]).value_counts(
            normalize=True).plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
        axes[1].set_title('CRPA Proportion')
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ 保存图表: {output_dir}/01_target_distribution.png")
        plt.close()

        print("\n✓ 所有图表生成完成")
        return self


def main():
    print("\n")
    print("=" * 70)
    print("   CRPA碳青霉烯耐药预测 - 数据预处理 (修订版)")
    print("=" * 70)
    print("  修订: 先分割再填补 [SE-2], 记录排除变量 [SE-7], 环境记录 [SE-9]")
    print("\n")

    data_path = './CRPA.csv'

    preprocessor = DataPreprocessor(data_path)

    (preprocessor
     .load_data()
     .check_data_quality()
     .remove_high_missing_features(threshold=10)
     .split_and_impute()                           # [SE-2] 先分割再填补
     .check_collinearity(threshold=10)
     .check_low_variance(threshold=0.01)
     .univariate_feature_selection(k=50, p_threshold=0.1)
     .save_processed_data('./processed_train.csv', './processed_test.csv')
     .log_environment()                            # [SE-9] 记录环境
     .plot_summary('./plots'))

    print("\n")
    print("=" * 70)
    print("  预处理完成！")
    print("=" * 70)
    print("\n下一步: 运行 02_feature_selection.py 进行特征筛选")
    print("\n")


if __name__ == '__main__':
    main()

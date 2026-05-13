import pandas as pd
import numpy as np
import os
import json
import platform
import importlib
import warnings
warnings.filterwarnings('once')

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, log_loss, f1_score)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════
RANDOM_SEED        = 42
TEST_SIZE          = 0.2
N_BOOTSTRAP_CI     = 1000
HEADLINE_MODEL     = 'XGBoost'
CATHETER_LABEL     = 'CATHETER TIP-IV'

INPUT_DATA         = './CRPA.csv'
MAIN_TRAIN_CSV     = './final_train_data.csv'
MAIN_VAL_CSV       = './final_val_data.csv'
MAIN_PERF_CSV      = './model_performance.csv'
MAIN_SHAP_CSV      = './shap_feature_importance.csv'

OUTPUT_DIR         = './sensitivity_catheter_tip'
PLOT_DIR           = f'{OUTPUT_DIR}/plots'
TABLE_DIR          = f'{OUTPUT_DIR}/tables'
MODEL_DIR          = f'{OUTPUT_DIR}/models'

for d in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _gpu_available():
    try:
        import subprocess
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except Exception:
        return False


def _bootstrap_ci(y_true, y_pred_proba, threshold, n_boot=N_BOOTSTRAP_CI):
    """Bootstrap percentile CI (与主管线 03 一致)"""
    n = len(y_true)
    keys = ['auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'brier', 'auprc']
    boot = {k: [] for k in keys}
    rng = np.random.default_rng(RANDOM_SEED)
    y_arr = np.asarray(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_arr[idx]; yp = y_pred_proba[idx]
        yc = (yp >= threshold).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        try:
            tn = ((yt == 0) & (yc == 0)).sum()
            fp = ((yt == 0) & (yc == 1)).sum()
            fn = ((yt == 1) & (yc == 0)).sum()
            tp = ((yt == 1) & (yc == 1)).sum()
            boot['auc'].append(roc_auc_score(yt, yp))
            boot['sensitivity'].append(tp / (tp + fn) if (tp + fn) else 0)
            boot['specificity'].append(tn / (tn + fp) if (tn + fp) else 0)
            boot['ppv'].append(tp / (tp + fp) if (tp + fp) else 0)
            boot['npv'].append(tn / (tn + fn) if (tn + fn) else 0)
            prec = tp / (tp + fp) if (tp + fp) else 0
            sens = tp / (tp + fn) if (tp + fn) else 0
            boot['f1'].append(2 * prec * sens / (prec + sens) if (prec + sens) else 0)
            boot['brier'].append(brier_score_loss(yt, yp))
            boot['auprc'].append(average_precision_score(yt, yp))
        except Exception:
            continue
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
            for k, v in boot.items()}


def _evaluate(name, y_true, y_pred):
    """评估模型 (Youden 最优阈值, 与主管线一致)"""
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    best_t = thr[np.argmax(tpr - fpr)]
    yc = (y_pred >= best_t).astype(int)
    tn = ((y_true == 0) & (yc == 0)).sum()
    fp = ((y_true == 0) & (yc == 1)).sum()
    fn = ((y_true == 1) & (yc == 0)).sum()
    tp = ((y_true == 1) & (yc == 1)).sum()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    f1 = f1_score(y_true, yc)
    brier = brier_score_loss(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    ci = _bootstrap_ci(y_true, y_pred, best_t)
    return {
        'Model': name, 'AUC': auc, 'AUC_CI': ci['auc'],
        'AUPRC': auprc, 'AUPRC_CI': ci['auprc'],
        'Sensitivity': sens, 'Sens_CI': ci['sensitivity'],
        'Specificity': spec, 'Spec_CI': ci['specificity'],
        'PPV': ppv, 'PPV_CI': ci['ppv'],
        'NPV': npv, 'NPV_CI': ci['npv'],
        'F1': f1, 'F1_CI': ci['f1'],
        'Brier': brier, 'Brier_CI': ci['brier'],
        'Threshold': best_t}


# ═══════════════════════════════════════════════════════════════════
# 步骤 1: 数据加载 + 方案 A 分割
# ═══════════════════════════════════════════════════════════════════

def step1_load_and_split():
    """方案 A: 用主分析的 train/test split, 再从中剔除 catheter-tip"""
    print("\n" + "=" * 70)
    print("步骤 1: 数据加载 + 方案 A 分割 (固定主分析分割, 剔除 catheter-tip)")
    print("=" * 70)

    df = pd.read_csv(INPUT_DATA, encoding='utf-8-sig')
    print(f"  输入文件: {INPUT_DATA}")
    print(f"  数据形状: {df.shape}")
    print(f"  CRPA 阳性率: {df['CRPA'].mean():.2%}")

    is_catheter = (df['Specimen Type'] == CATHETER_LABEL)
    n_catheter = is_catheter.sum()
    print(f"\n  Catheter-tip 样本: {n_catheter}/{len(df)} ({n_catheter/len(df):.1%})")
    print(f"  Catheter-tip CRPA+: {df.loc[is_catheter, 'CRPA'].sum()}/{n_catheter}")

    # 读取主分析的 15 个最终特征
    main_train = pd.read_csv(MAIN_TRAIN_CSV)
    features = [c for c in main_train.columns if c != 'CRPA']
    print(f"\n  主分析最终特征 ({len(features)} 个): {features}")

    # 重建主分析的 train/test split
    exclude_cols = ['Anchor Year Group', 'CRPA', 'Specimen Type']
    feature_cols_all = [c for c in df.columns if c not in exclude_cols]
    y = df['CRPA'].values
    X = df[feature_cols_all]

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    train_idx = X_train_full.index.tolist()
    test_idx = X_test_full.index.tolist()

    catheter_in_train = is_catheter.iloc[train_idx].sum()
    catheter_in_test = is_catheter.iloc[test_idx].sum()
    print(f"\n  主分析分割中 catheter-tip 分布:")
    print(f"    训练集: {catheter_in_train}/{len(train_idx)}")
    print(f"    测试集: {catheter_in_test}/{len(test_idx)}")

    # 方案 A: 从训练/测试集中各自剔除 catheter-tip
    train_clean = [i for i in train_idx if not is_catheter.iloc[i]]
    test_clean = [i for i in test_idx if not is_catheter.iloc[i]]

    X_train = df[features].iloc[train_clean].reset_index(drop=True)
    X_test = df[features].iloc[test_clean].reset_index(drop=True)
    y_train = df['CRPA'].iloc[train_clean].values
    y_test = df['CRPA'].iloc[test_clean].values

    # 中位数填补 (与主管线一致: fit on train, transform both)
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=features)

    print(f"\n  方案 A 剔除后:")
    print(f"    训练集: {len(X_train)} (CRPA+: {y_train.sum()}, 率: {y_train.mean():.2%})")
    print(f"    测试集: {len(X_test)} (CRPA+: {y_test.sum()}, 率: {y_test.mean():.2%})")
    print(f"    总计: {len(X_train)+len(X_test)} (= 567 - {n_catheter} = {567-n_catheter})")

    return X_train, X_test, y_train, y_test, features


# ═══════════════════════════════════════════════════════════════════
# 步骤 2: 6 个模型训练 + 评估
# ═══════════════════════════════════════════════════════════════════
# PLACEHOLDER_STEP2

def step2_train_and_evaluate(X_train, X_test, y_train, y_test, features):
    """训练 6 个模型并评估 (超参与主管线一致)"""
    print("\n" + "=" * 70)
    print("步骤 2: 模型训练 + 评估 (6 个模型, 超参与主管线一致)")
    print("=" * 70)

    y_tr = pd.Series(y_train)
    y_te = pd.Series(y_test)
    spw = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    device_xgb = 'cuda' if _gpu_available() else 'cpu'
    device_lgb = 'gpu' if _gpu_available() else 'cpu'
    task_type_cb = 'GPU' if _gpu_available() else 'CPU'

    models = {}
    preds = {}

    # [1] LASSO
    print("\n  [1/6] LASSO ...")
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)
    lr = LogisticRegressionCV(
        penalty='l1', solver='saga', Cs=100, cv=5, max_iter=10000,
        random_state=RANDOM_SEED, class_weight='balanced',
        scoring='roc_auc', n_jobs=-1)
    lr.fit(Xs_tr, y_train)
    models['LASSO'] = lr
    preds['LASSO'] = lr.predict_proba(Xs_te)[:, 1]

    # [2] Random Forest
    print("  [2/6] Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    preds['Random Forest'] = rf.predict_proba(X_test)[:, 1]

    # [3] AdaBoost
    print("  [3/6] AdaBoost ...")
    base = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_SEED)
    ada = AdaBoostClassifier(estimator=base, n_estimators=200,
                              learning_rate=1.0, algorithm='SAMME',
                              random_state=RANDOM_SEED)
    ada.fit(X_train, y_train)
    models['AdaBoost'] = ada
    preds['AdaBoost'] = ada.predict_proba(X_test)[:, 1]

    # [4] XGBoost
    print("  [4/6] XGBoost ...")
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'scale_pos_weight': spw,
        'tree_method': 'hist', 'device': device_xgb, 'seed': RANDOM_SEED}
    X_in, X_es, y_in, y_es = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    d_in = xgb.DMatrix(X_in, label=y_in)
    d_es = xgb.DMatrix(X_es, label=y_es)
    d_tr = xgb.DMatrix(X_train, label=y_train)
    d_te = xgb.DMatrix(X_test, label=y_test)
    model_es = xgb.train(xgb_params, d_in, num_boost_round=1000,
                          evals=[(d_es, 'eval')],
                          early_stopping_rounds=50, verbose_eval=False)
    best_iter = model_es.best_iteration
    xgb_model = xgb.train(xgb_params, d_tr, num_boost_round=best_iter + 1, verbose_eval=False)
    models['XGBoost'] = xgb_model
    preds['XGBoost'] = xgb_model.predict(d_te)
    with open(f'{MODEL_DIR}/xgboost_hyperparameters.json', 'w') as f:
        json.dump({**xgb_params, 'best_iteration': best_iter + 1}, f, indent=2, default=str)

    # [5] LightGBM
    print("  [5/6] LightGBM ...")
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'scale_pos_weight': spw, 'device': device_lgb,
        'verbose': -1, 'seed': RANDOM_SEED}
    X_in_l, X_es_l, y_in_l, y_es_l = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    td = lgb.Dataset(X_in_l, label=y_in_l)
    vd = lgb.Dataset(X_es_l, label=y_es_l, reference=td)
    lgb_es = lgb.train(lgb_params, td, num_boost_round=1000,
                        valid_sets=[vd],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    best_iter_l = lgb_es.best_iteration
    lgb_full = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train),
                          num_boost_round=best_iter_l)
    models['LightGBM'] = lgb_full
    preds['LightGBM'] = lgb_full.predict(X_test, num_iteration=best_iter_l)

    # [6] CatBoost
    print("  [6/6] CatBoost ...")
    cb_p = dict(depth=6, learning_rate=0.05, l2_leaf_reg=3,
                scale_pos_weight=spw, task_type=task_type_cb,
                devices='0', random_seed=RANDOM_SEED, eval_metric='AUC')
    X_in_c, X_es_c, y_in_c, y_es_c = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train)
    cb_es = CatBoostClassifier(iterations=1000, early_stopping_rounds=50, verbose=0, **cb_p)
    cb_es.fit(X_in_c, y_in_c, eval_set=(X_es_c, y_es_c), plot=False)
    best_iter_c = cb_es.best_iteration_
    cb = CatBoostClassifier(iterations=best_iter_c, verbose=0, **cb_p)
    cb.fit(X_train, y_train, plot=False)
    models['CatBoost'] = cb
    preds['CatBoost'] = cb.predict_proba(X_test)[:, 1]

    # 评估
    print("\n  ✓ 6 个模型训练完成, 开始评估...")
    results = []
    for name, yp in preds.items():
        r = _evaluate(name, y_te.values, yp)
        results.append(r)
        print(f"    {name:14s}  AUROC={r['AUC']:.4f} "
              f"({r['AUC_CI'][0]:.3f}-{r['AUC_CI'][1]:.3f})  "
              f"AUPRC={r['AUPRC']:.4f}  Brier={r['Brier']:.4f}")

    perf = pd.DataFrame(results)
    perf['AUC_95CI'] = perf['AUC_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['Sens_95CI'] = perf['Sens_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['Spec_95CI'] = perf['Spec_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['PPV_95CI'] = perf['PPV_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['NPV_95CI'] = perf['NPV_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['F1_95CI'] = perf['F1_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['Brier_95CI'] = perf['Brier_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")
    perf['AUPRC_95CI'] = perf['AUPRC_CI'].apply(lambda t: f"({t[0]:.4f}-{t[1]:.4f})")

    perf_out = perf.drop(columns=['AUC_CI', 'Sens_CI', 'Spec_CI', 'PPV_CI',
                                   'NPV_CI', 'F1_CI', 'Brier_CI', 'AUPRC_CI']
                         ).sort_values('AUC', ascending=False)
    perf_out.to_csv(f'{OUTPUT_DIR}/sensitivity_model_performance.csv', index=False)
    print(f"\n  ✓ 性能表: {OUTPUT_DIR}/sensitivity_model_performance.csv")

    xgb_model.save_model(f'{MODEL_DIR}/xgboost_model.json')
    return models, preds, perf_out, features


# ═══════════════════════════════════════════════════════════════════
# 步骤 3: SHAP 分析 (XGBoost, pred_contribs)
# ═══════════════════════════════════════════════════════════════════
# PLACEHOLDER_STEP3

def step3_shap_analysis(xgb_model, X_test, feature_names):
    """SHAP 全局分析 (XGBoost pred_contribs, 绕过 shap 库兼容性问题)"""
    print("\n" + "=" * 70)
    print("步骤 3: SHAP 全局分析 (XGBoost pred_contribs)")
    print("=" * 70)

    d_te = xgb.DMatrix(X_test, feature_names=feature_names)
    contribs = xgb_model.predict(d_te, pred_contribs=True)
    shap_values = contribs[:, :-1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({'Feature': feature_names, 'Mean_Abs_SHAP': mean_abs}
                       ).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    imp['Rank'] = imp.index + 1
    imp.to_csv(f'{OUTPUT_DIR}/sensitivity_shap_importance.csv', index=False)
    print(f"  ✓ SHAP 重要性: {OUTPUT_DIR}/sensitivity_shap_importance.csv")

    # SHAP bar plot
    try:
        import shap
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type='bar', max_display=15, show=False)
        plt.title('SHAP Importance — Sensitivity (Catheter-tip Excluded)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/sens_shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    return imp


# ═══════════════════════════════════════════════════════════════════
# 步骤 4: 构建 Supplementary Table S15
# ═══════════════════════════════════════════════════════════════════

def step4_build_s15(sens_perf, sens_shap, n_train_clean, n_test_clean):
    """构建 S15 并列对比表"""
    print("\n" + "=" * 70)
    print("步骤 4: 构建 Supplementary Table S15 (主分析 vs 敏感性分析)")
    print("=" * 70)

    # 加载主分析结果
    main_perf = pd.read_csv(MAIN_PERF_CSV)
    col_map = {'Sensitivity_95CI': 'Sens_95CI', 'Specificity_95CI': 'Spec_95CI',
               'Precision_95CI': 'PPV_95CI', 'Precision': 'PPV'}
    main_perf = main_perf.rename(columns=col_map)
    if 'AUPRC' not in main_perf.columns:
        main_perf['AUPRC'] = np.nan
    print(f"  ✓ 主分析性能表已加载: {MAIN_PERF_CSV}")

    main_shap = pd.read_csv(MAIN_SHAP_CSV)
    if 'Rank' not in main_shap.columns:
        main_shap = main_shap.sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        main_shap['Rank'] = main_shap.index + 1
    print(f"  ✓ 主分析 SHAP 表已加载: {MAIN_SHAP_CSV}")

    # 转为 dict 方便查找
    sens_lookup = {r['Model']: r.to_dict() for _, r in sens_perf.iterrows()}
    main_lookup = {r['Model']: r.to_dict() for _, r in main_perf.iterrows()}

    # 4A: 6 模型 AUROC 并列
    all_models = ['LASSO', 'Random Forest', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost']
    rows_all = []
    for m in all_models:
        ms = sens_lookup.get(m, {})
        mm = main_lookup.get(m, {})
        rows_all.append({
            'Model': m,
            'Main_AUROC': f"{mm.get('AUC', np.nan):.4f}" if mm else 'NA',
            'Main_AUROC_95CI': mm.get('AUC_95CI', 'NA') if mm else 'NA',
            'Sens_AUROC': f"{ms.get('AUC', np.nan):.4f}" if ms else 'NA',
            'Sens_AUROC_95CI': ms.get('AUC_95CI', 'NA') if ms else 'NA',
            'Delta_AUROC': f"{ms['AUC'] - mm['AUC']:+.4f}" if ms and mm else 'NA',
        })
    df_all = pd.DataFrame(rows_all)
    df_all.to_csv(f'{TABLE_DIR}/Suppl_Table_S15_All_Models.csv', index=False)
    print(f"\n  ✓ {TABLE_DIR}/Suppl_Table_S15_All_Models.csv")
    print(df_all.to_string(index=False))

    # 4B: Headline 模型指标并列
    ms = sens_lookup.get(HEADLINE_MODEL, {})
    mm = main_lookup.get(HEADLINE_MODEL, {})
    n_sens = n_train_clean + n_test_clean

    def _fmt(d, key, ci_key):
        if not d:
            return 'NA'
        v = d.get(key, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'NA'
        s = f"{v:.4f}"
        if ci_key and ci_key in d and isinstance(d[ci_key], str):
            s += f" {d[ci_key]}"
        return s

    metrics = [('AUROC (95% CI)', 'AUC', 'AUC_95CI'),
               ('AUPRC (95% CI)', 'AUPRC', 'AUPRC_95CI'),
               ('Sensitivity', 'Sensitivity', 'Sens_95CI'),
               ('Specificity', 'Specificity', 'Spec_95CI'),
               ('NPV', 'NPV', 'NPV_95CI'),
               ('Brier', 'Brier', 'Brier_95CI')]
    rows_head = []
    for label, key, ci_key in metrics:
        rows_head.append({
            'Metric': label,
            f'Main analysis (n = 567)': _fmt(mm, key, ci_key),
            f'Catheter-tip excluded (n = {n_sens})': _fmt(ms, key, ci_key),
        })
    rows_head.append({
        'Metric': 'Final features (n)',
        f'Main analysis (n = 567)': '15',
        f'Catheter-tip excluded (n = {n_sens})': '15 (same)'})
    df_head = pd.DataFrame(rows_head)
    df_head.to_csv(f'{TABLE_DIR}/Suppl_Table_S15_CatheterTip_Sensitivity.csv', index=False)
    print(f"\n  ✓ {TABLE_DIR}/Suppl_Table_S15_CatheterTip_Sensitivity.csv")
    print(df_head.to_string(index=False))

    # 4C: SHAP 排名对比
    main_top = main_shap.head(15)[['Feature', 'Rank']].rename(columns={'Rank': 'Main_Rank'})
    sens_top = sens_shap.head(15)[['Feature', 'Rank']].rename(columns={'Rank': 'Sens_Rank'})
    merged = pd.merge(main_top, sens_top, on='Feature', how='outer')
    merged['In_Both_Top15'] = (merged['Main_Rank'].notna() & merged['Sens_Rank'].notna()).astype(int)
    merged = merged.sort_values(['In_Both_Top15', 'Main_Rank'], ascending=[False, True])
    merged.to_csv(f'{TABLE_DIR}/Suppl_Table_S15_SHAP_Rank.csv', index=False)

    common = merged[merged['In_Both_Top15'] == 1]
    n_overlap = len(common)
    if n_overlap >= 2:
        rho, p = stats.spearmanr(common['Main_Rank'], common['Sens_Rank'])
    else:
        rho, p = np.nan, np.nan

    print(f"\n  ✓ {TABLE_DIR}/Suppl_Table_S15_SHAP_Rank.csv")
    print(f"    Top-15 重叠数: {n_overlap} / 15")
    if not np.isnan(rho):
        print(f"    Spearman ρ = {rho:.3f} (p = {p:.4f})")
    print(merged.to_string(index=False))

    # 4D: SHAP 对比柱图
    try:
        top15_features = sorted(
            set(main_shap.head(15)['Feature']) | set(sens_shap.head(15)['Feature']))
        main_dict = dict(zip(main_shap['Feature'], main_shap['Mean_Abs_SHAP']))
        sens_dict = dict(zip(sens_shap['Feature'], sens_shap['Mean_Abs_SHAP']))
        main_vals = [main_dict.get(f, 0) for f in top15_features]
        sens_vals = [sens_dict.get(f, 0) for f in top15_features]

        fig, ax = plt.subplots(figsize=(12, max(6, len(top15_features) * 0.4)))
        ypos = np.arange(len(top15_features))
        ax.barh(ypos - 0.2, main_vals, height=0.4, color='#1f77b4', label='Main (n=567)')
        ax.barh(ypos + 0.2, sens_vals, height=0.4, color='#ff7f0e',
                 label=f'Sensitivity (n={n_sens}, catheter-tip excluded)')
        ax.set_yticks(ypos); ax.set_yticklabels(top15_features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean(|SHAP value|)', fontweight='bold')
        title = (f'SHAP Feature Importance: Main vs Sensitivity ({HEADLINE_MODEL})\n'
                 f'Top-15 overlap = {n_overlap}/15')
        if not np.isnan(rho):
            title += f'  Spearman ρ = {rho:.3f}'
        ax.set_title(title, fontweight='bold')
        ax.legend(); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/Suppl_Fig_S15_SHAP_Compare.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {PLOT_DIR}/Suppl_Fig_S15_SHAP_Compare.png")
    except Exception as e:
        print(f"  ⚠ SHAP 对比图绘制失败: {e}")


# ═══════════════════════════════════════════════════════════════════
# 环境记录
# ═══════════════════════════════════════════════════════════════════

def log_environment():
    info = {'Python': platform.python_version(), 'Platform': platform.platform(),
            'Random_Seed': str(RANDOM_SEED), 'Script': '05_sensitivity_catheter_tip.py',
            'Method': 'Plan A (fixed split, remove catheter-tip from both sets)'}
    for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost',
                'lightgbm', 'catboost', 'shap', 'matplotlib']:
        try:
            mod = importlib.import_module(pkg)
            info[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            info[pkg] = 'not installed'
    pd.DataFrame(list(info.items()), columns=['Package', 'Version']).to_csv(
        f'{OUTPUT_DIR}/sensitivity_environment.csv', index=False)


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  CRPA 预测 - Catheter-tip 排除敏感性分析 (方案 A: 固定分割)")
    print("=" * 70)
    print("  方案 A: 保持主分析的 train/test 分割不变,")
    print("          从训练集和测试集中各自剔除 catheter-tip 样本,")
    print("          使用主分析确定的 15 个特征, 重新训练 6 个模型。")
    print("  优势: ΔAUROC 纯粹反映 catheter-tip 的影响, 无随机分割噪声。")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print("=" * 70)

    # Step 1
    X_train, X_test, y_train, y_test, features = step1_load_and_split()

    # Step 2
    models, preds, sens_perf, _ = step2_train_and_evaluate(
        X_train, X_test, y_train, y_test, features)

    # Step 3
    sens_shap = step3_shap_analysis(models['XGBoost'], X_test, features)

    # Step 4
    step4_build_s15(sens_perf, sens_shap, len(X_train), len(X_test))

    # 环境记录
    log_environment()

    print("\n" + "=" * 70)
    print("  敏感性分析完成!")
    print("=" * 70)
    print(f"""
  核心交付:
    {TABLE_DIR}/Suppl_Table_S15_CatheterTip_Sensitivity.csv
    {TABLE_DIR}/Suppl_Table_S15_All_Models.csv
    {TABLE_DIR}/Suppl_Table_S15_SHAP_Rank.csv
    {PLOT_DIR}/Suppl_Fig_S15_SHAP_Compare.png
""")


if __name__ == '__main__':
    main()

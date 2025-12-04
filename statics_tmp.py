import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "mmlu_role_entropy_analysis.csv"


def comprehensive_optimization():
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print("Error: CSV file not found. Please run the data generation script first.")
        return

    # === 1. 基础特征清洗 ===
    # 确保没有无穷大或 NaN
    base_cols = ['avg_entropy', 'perplexity', 'volatility', 'avg_margin']
    for col in base_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=base_cols + ['is_correct']).reset_index(drop=True)

    # === 2. 特征工程大爆发 (Feature Explosion) ===
    print("Generating comprehensive features...")

    # (A) 预处理：Log 变换 (对长尾分布的 PPL 很有用)
    df['log_ppl'] = np.log(df['perplexity'] + 1e-6)

    # (B) 相对特征 (Relative Features) - 对每个 Task 内部进行比较
    # 这是一个非常强的归纳偏置：绝对值不重要，"比别人好"才重要
    grouped = df.groupby('sample_id')

    targets = ['avg_entropy', 'perplexity', 'volatility', 'avg_margin', 'log_ppl']

    relative_features = []

    for col in targets:
        # 1. Rank (排名): 1, 2, 3...
        # pct=True 可以转为百分比排名 (0.0 - 1.0)，消除 Task 候选项数量不同的影响
        col_rank = f"{col}_rank"
        df[col_rank] = grouped[col].rank(ascending=(col != 'avg_margin'), pct=True)
        # 注意：Margin 是越大越好，其他通常是越小越好，这里统一一下方向其实无所谓，树模型能学到
        # 但为了人类理解，我们保持原始数值的 Rank

        # 2. Z-Score (标准分): (x - mean) / std
        col_z = f"{col}_z"
        df[col_z] = grouped[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))

        # 3. Diff form Mean (与均值的差): x - mean
        col_diff = f"{col}_diff"
        df[col_diff] = df[col] - grouped[col].transform('mean')

        # 4. Distance to Best (与组内最优值的距离)
        # 假设我们不知道谁是 best，但可以算离 min/max 的距离
        col_dist_min = f"{col}_dist_min"
        df[col_dist_min] = df[col] - grouped[col].transform('min')

        relative_features.extend([col_rank, col_z, col_diff, col_dist_min])

    # (C) 交互特征 (Interaction Features)
    # 结合不同维度的信息
    df['feat_margin_div_ppl'] = df['avg_margin'] / (df['perplexity'] + 1e-6)
    df['feat_ent_div_ppl'] = df['avg_entropy'] / (df['perplexity'] + 1e-6)
    # "幻觉分数": 非常自信(High Margin) 但 读不懂(High PPL)
    df['feat_hallucination'] = df['avg_margin'] * df['log_ppl']
    # "纠结分数": 波动大(High Vol) 且 熵高(High Ent)
    df['feat_confusion'] = df['volatility'] * df['avg_entropy']

    interaction_features = ['feat_margin_div_ppl', 'feat_ent_div_ppl', 'feat_hallucination', 'feat_confusion']

    # === 3. 训练与评估 ===
    # 汇总所有特征
    all_features = base_cols + ['log_ppl'] + relative_features + interaction_features
    target = 'is_correct'

    print(f"Total Features used: {len(all_features)}")

    # 再次清洗 (防止 Z-score 产生 NaN)
    df = df.dropna(subset=all_features + [target]).reset_index(drop=True)

    X = df[all_features]
    y = df[target]
    groups = df['sample_id']

    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE RF (GroupKFold)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    df['cv_pred_score'] = 0.0
    feature_importances = np.zeros(len(all_features))

    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # 使用稍微深一点的树，因为特征多了
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=4,
                                    max_features='sqrt',  # 强制随机采样特征，防止 PPL 一家独大
                                    class_weight='balanced',
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # 预测
        df.loc[test_idx, 'cv_pred_score'] = rf.predict_proba(X_test)[:, 1]
        feature_importances += rf.feature_importances_
        fold += 1

    feature_importances /= 5

    # === 4. 结果展示 ===
    # (1) 特征重要性 Top 15
    print("\nTop 15 Feature Importances:")
    fi_series = pd.Series(feature_importances, index=all_features).sort_values(ascending=False)
    print(fi_series.head(15))

    # (2) 策略准确率对比
    strategies = {
        "Random Selection": lambda x: np.random.rand(len(x)),
        "Min Perplexity": lambda x: -x['perplexity'],  # 基准
        "Max Margin": lambda x: x['avg_margin'],
        "RF (All Features)": lambda x: x['cv_pred_score']  # 我们的全量模型
    }

    results = []
    oracle_acc = df.groupby('sample_id')[target].max().mean()

    for name, func in strategies.items():
        correct = 0
        total = 0
        for _, group in df.groupby('sample_id'):
            # 选分数最高的
            scores = func(group)
            if isinstance(scores, pd.Series):
                best_idx = scores.idxmax()
            else:
                best_idx = group.index[np.argmax(scores)]

            if group.loc[best_idx, target] == 1:
                correct += 1
            total += 1
        results.append({"Strategy": name, "Accuracy": correct / total})

    res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    # 计算提升
    base_acc = res_df.loc[res_df['Strategy'] == "Random Selection", "Accuracy"].values[0]
    res_df['Lift'] = (res_df['Accuracy'] - base_acc) / base_acc

    print("\nFinal Strategy Performance:")
    print(res_df)
    print(f"\nOracle Bound: {oracle_acc:.4f}")

    # 保存特征重要性供分析
    fi_series.to_csv("feature_importances.csv")
    print("Feature importances saved to 'feature_importances.csv'")


if __name__ == "__main__":
    comprehensive_optimization()
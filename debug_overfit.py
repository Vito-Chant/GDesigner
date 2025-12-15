import os
import glob
import pickle
import numpy as np
import pandas as pd
import random
import warnings
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import multiprocessing

# ============================
# 配置区域
# ============================
VAL_DIR = "validation_data_v2"  # 确保指向包含 batch_x.pkl 的目录

# 挖掘策略配置
N_SESSIONS = 1000  # 运行轮数
MAX_GENS_PER_SESSION = 200
PATIENCE = 15  # 早停耐心值
POPULATION_SIZE = 500  # 种群大小
TOURNAMENT_SIZE = 8
N_JOBS = -1  # 并行核心数

# ============================
# 0. 顶层算子定义
# ============================
warnings.filterwarnings("ignore")


def op_add(x, y): return x + y


def op_sub(x, y): return x - y


def op_mul(x, y): return x * y


def op_div(x, y): return x / (y + 1e-6)


def op_abs(x): return np.abs(x)


def op_neg(x): return -x


def op_sq(x): return x ** 2


def op_sqrt(x): return np.sqrt(np.abs(x))


def op_max(x, y): return np.maximum(x, y)  # 新增算子


def op_min(x, y): return np.minimum(x, y)  # 新增算子


OPS = {
    'add': op_add, 'sub': op_sub, 'mul': op_mul, 'div': op_div,
    'abs': op_abs, 'neg': op_neg, 'sq': op_sq, 'sqrt': op_sqrt,
    'max': op_max, 'min': op_min
}

OP_METADATA = {
    'add': (2, "({} + {})"), 'sub': (2, "({} - {})"),
    'mul': (2, "({} * {})"), 'div': (2, "({} / {})"),
    'abs': (1, "abs({})"), 'neg': (1, "-({})"),
    'sq': (1, "({}**2)"), 'sqrt': (1, "sqrt(|{}|)"),
    'max': (2, "max({}, {})"), 'min': (2, "min({}, {})")
}


# ============================
# 1. 全景特征提取 (Feature Engineering Level 1-4)
# ============================

def calc_slope(y):
    """一阶趋势 (速度)"""
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    x_mean = (n - 1) / 2.0
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / (denominator + 1e-9)


def calc_curvature(y):
    """二阶趋势 (加速度): 前半段斜率 vs 后半段斜率"""
    n = len(y)
    if n < 4: return 0.0
    mid = n // 2
    slope1 = calc_slope(y[:mid])
    slope2 = calc_slope(y[mid:])
    return slope2 - slope1  # 正值表示加速上升，负值表示加速下降


def get_quantiles(arr):
    """返回 Q25, Q50(Median), Q75"""
    if len(arr) == 0: return 0, 0, 0
    return np.percentile(arr, [25, 50, 75])


def process_single_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            batch = pickle.load(f)
        results = []
        for item in batch:
            # === 数据完整性检查 ===
            feat = item.get('features')
            if feat is None or feat.shape[0] < 5: continue

            # 获取 GT Logprobs
            gt_logprobs = item.get('gt_logprobs')
            has_gt = (gt_logprobs is not None and len(gt_logprobs) == feat.shape[0])

            # 基础序列提取
            logprobs = feat[:, 0]  # Top-1 Logprobs
            mask = logprobs > -99
            if not mask.any(): continue

            lp_vals = logprobs[mask]
            n_seq = len(lp_vals)
            prob_vals = np.exp(lp_vals)

            # Label
            row = {'label': float(item['is_correct'])}

            # ==========================================
            # Group A: 核心置信度 (Confidence)
            # ==========================================
            # 1. 基础统计
            row['lp_mean'] = np.mean(lp_vals)
            row['lp_std'] = np.std(lp_vals)
            row['lp_sum'] = np.sum(lp_vals)  # 联合概率

            # 2. 分位数与离散度 (Robust Stats)
            q25, q50, q75 = get_quantiles(lp_vals)
            row['lp_q25'] = q25
            row['lp_median'] = q50
            row['lp_iqr'] = q75 - q25  # 四分位距，比 std 更抗噪

            # 3. 极值与木桶效应
            row['lp_min'] = np.min(lp_vals)
            # 计数特征: 有多少个 token 极其不自信 (-2.3 ≈ 10% prob)
            row['lp_low_conf_count'] = np.sum(lp_vals < -2.3)
            row['lp_low_conf_ratio'] = row['lp_low_conf_count'] / n_seq

            # 4. 时序动态
            row['lp_slope'] = calc_slope(lp_vals)  # 线性趋势
            row['lp_curve'] = calc_curvature(lp_vals)  # 加速度
            row['lp_gap_fl'] = lp_vals[-1] - lp_vals[0]

            # 5. 分段统计 (前半段 vs 后半段)
            mid = n_seq // 2
            row['lp_mean_first'] = np.mean(lp_vals[:mid])
            row['lp_mean_last'] = np.mean(lp_vals[mid:])
            row['lp_ratio_fl'] = row['lp_mean_last'] / (row['lp_mean_first'] - 1e-9)  # 避免除0

            # ==========================================
            # Group B: 不确定性与熵 (Entropy / Uncertainty)
            # ==========================================
            if feat.shape[1] > 1:
                probs_full = np.exp(feat[mask])
                p_sum = np.sum(probs_full, axis=1, keepdims=True) + 1e-10
                norm_p = probs_full / p_sum
                ents = -np.sum(norm_p * np.log(norm_p + 1e-10), axis=1)

                row['ent_mean'] = np.mean(ents)
                row['ent_std'] = np.std(ents)
                row['ent_max'] = np.max(ents)
                row['ent_slope'] = calc_slope(ents)

                # 变异系数 (Hesitation)
                row['prob_cv'] = np.std(prob_vals) / (np.mean(prob_vals) + 1e-9)

                # 高级交互: 熵与置信度的背离
                # 理想情况: 置信度高(logprob大)时熵应低。
                # 如果相关性异常，说明模型"盲目自信"或"混乱"
                if n_seq > 2:
                    # 简化版相关性 (Covariance)
                    cov = np.cov(lp_vals, ents)[0, 1]
                    row['cov_lp_ent'] = cov
                else:
                    row['cov_lp_ent'] = 0.0
            else:
                for c in ['ent_mean', 'ent_std', 'ent_max', 'ent_slope', 'prob_cv', 'cov_lp_ent']:
                    row[c] = 0.0

            # ==========================================
            # Group C: 惊诧度与真实性 (Ground Truth / Surprise)
            # ==========================================
            if has_gt:
                gt_vals = gt_logprobs[mask]
                surprise_vals = lp_vals - gt_vals  # Top1 - GT (always >= 0)

                # GT 基础 (Reading Comprehension)
                row['gt_mean'] = np.mean(gt_vals)
                row['gt_slope'] = calc_slope(gt_vals)

                # Surprise 基础 (Alignment)
                row['surp_mean'] = np.mean(surprise_vals)
                row['surp_max'] = np.max(surprise_vals)
                row['surp_std'] = np.std(surprise_vals)

                # 交互特征:
                # 1. 惊诧度占比: 错误有多少是来源于"意外"?
                row['surp_ratio'] = np.sum(surprise_vals) / (np.abs(np.sum(lp_vals)) + 1e-9)

                # 2. 结尾惊诧度: 临门一脚是否出错?
                row['surp_last'] = surprise_vals[-1]

                # 3. 惊诧度聚集性: 最大的惊诧是否发生在后半段?
                if n_seq > 1:
                    max_idx = np.argmax(surprise_vals)
                    row['surp_max_pos'] = max_idx / n_seq  # 0.0 ~ 1.0 (越接近1越危险)
                else:
                    row['surp_max_pos'] = 0.5
            else:
                for c in ['gt_mean', 'gt_slope', 'surp_mean', 'surp_max',
                          'surp_std', 'surp_ratio', 'surp_last', 'surp_max_pos']:
                    row[c] = 0.0

            results.append(row)
        return results
    except Exception as e:
        return []


def load_data_once(val_dir):
    print(f"正在加载数据并计算全景特征 (v3) (CPU Cores: {multiprocessing.cpu_count()})...")
    if not os.path.exists(val_dir): return pd.DataFrame()
    pkl_files = glob.glob(os.path.join(val_dir, "*.pkl"))

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_pkl, pkl_files), total=len(pkl_files)))

    all_rows = [r for res in results for r in res]
    df = pd.DataFrame(all_rows)
    df.fillna(0, inplace=True)
    return df


# ============================
# 2. 遗传规划逻辑 (GP Engine)
# ============================
class Individual:
    def __init__(self, expr_tree=None):
        self.expr_tree = expr_tree
        self.auc = 0.0
        self.formula_str = ""

    def __str__(self):
        return self.formula_str if self.formula_str else self._str_node(self.expr_tree)

    def _str_node(self, node):
        if isinstance(node, str): return node
        op_name = node[0]
        if op_name not in OP_METADATA: return "Error"
        fmt = OP_METADATA[op_name][1]
        args = [self._str_node(child) for child in node[1:]]
        return fmt.format(*args)


def evaluate_worker(expr_tree, df, y_true):
    try:
        def _eval(node, data):
            if isinstance(node, str):
                return data[node]
            elif isinstance(node, tuple):
                func = OPS[node[0]]
                args = [_eval(child, data) for child in node[1:]]
                return func(*args)
            return data.iloc[:, 0] * 0

        scores = _eval(expr_tree, df)
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 极速计算 AUC (避免 sklearn 开销)
        # 这里还是用 sklearn 保证准确，但如果慢可以换 mannwhitneyu
        auc = roc_auc_score(y_true, scores)
        if auc < 0.5: auc = 1 - auc
        return auc
    except:
        return 0.0


def random_tree(features, depth=2):
    if depth == 0 or (depth < 2 and random.random() < 0.2):
        return random.choice(features)
    op = random.choice(list(OP_METADATA.keys()))
    children = [random_tree(features, depth - 1) for _ in range(OP_METADATA[op][0])]
    return (op, *children)


def mutate(individual, features):
    if random.random() < 0.4:
        return Individual(random_tree(features, depth=random.randint(1, 4)))
    return individual


def crossover(ind1, ind2):
    return ind1 if random.random() < 0.5 else ind2


# ============================
# 3. 单轮会话逻辑
# ============================
def run_single_session(session_id, df, y, base_feats):
    new_seed = int(time.time()) + session_id * 1000
    random.seed(new_seed)
    np.random.seed(new_seed)

    print(f"\n>>> 启动第 {session_id + 1}/{N_SESSIONS} 轮挖掘 (Seed: {new_seed})")

    population = [Individual(random_tree(base_feats)) for _ in range(POPULATION_SIZE)]
    best_auc_this_session = 0.0
    no_improv_count = 0
    session_hall_of_fame = []

    for gen in range(MAX_GENS_PER_SESSION):
        trees = [ind.expr_tree for ind in population]
        aucs = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(evaluate_worker)(t, df, y) for t in trees
        )

        for i, ind in enumerate(population):
            ind.auc = aucs[i]
            ind.formula_str = ind._str_node(ind.expr_tree)

        valid_pop = [ind for ind in population if ind.auc > 0]
        valid_pop.sort(key=lambda x: x.auc, reverse=True)

        if not valid_pop:
            population = [Individual(random_tree(base_feats)) for _ in range(POPULATION_SIZE)]
            continue

        top_1 = valid_pop[0]

        if top_1.auc > best_auc_this_session + 0.0002:
            best_auc_this_session = top_1.auc
            no_improv_count = 0
            session_hall_of_fame.append(top_1)
        else:
            no_improv_count += 1

        if gen % 5 == 0:
            print(
                f"   [S{session_id + 1}] Gen {gen:02d} | Best AUC: {top_1.auc:.4f} | Patience: {no_improv_count}/{PATIENCE}")

        if no_improv_count >= PATIENCE:
            print(f"   🛑 [早停] 本轮在 {gen} 代结束。最佳 AUC: {best_auc_this_session:.5f}")
            break

        next_gen = valid_pop[:5]
        while len(next_gen) < POPULATION_SIZE:
            parents = random.sample(valid_pop, min(len(valid_pop), TOURNAMENT_SIZE))
            p1 = max(parents, key=lambda x: x.auc)
            parents = random.sample(valid_pop, min(len(valid_pop), TOURNAMENT_SIZE))
            p2 = max(parents, key=lambda x: x.auc)
            next_gen.append(mutate(crossover(p1, p2), base_feats))
        population = next_gen

    return session_hall_of_fame


# ============================
# 4. 主程序
# ============================
def main():
    # 1. 准备数据
    df = load_data_once(VAL_DIR)
    if len(df) == 0: return print("无数据。请检查目录。")
    y = df['label'].values
    base_feats = [c for c in df.columns if c != 'label']

    print(f"\n✅ 特征池升级完成，共包含 {len(base_feats)} 个全景特征。")
    print(f"示例特征: {random.sample(base_feats, min(5, len(base_feats)))}")

    # 2. 全局容器
    global_hall_of_fame = []

    # 3. 循环运行 Session
    for i in range(N_SESSIONS):
        top_candidates = run_single_session(i, df, y, base_feats)
        global_hall_of_fame.extend(top_candidates)

        # 清理与去重
        global_hall_of_fame.sort(key=lambda x: x.auc, reverse=True)
        unique_hof = []
        seen = set()
        for ind in global_hall_of_fame:
            if ind.formula_str not in seen:
                seen.add(ind.formula_str)
                unique_hof.append(ind)
        global_hall_of_fame = unique_hof[:50]

        if len(global_hall_of_fame) > 0:
            print(f"   >>> 🏆 全局最佳: {global_hall_of_fame[0].auc:.5f} | {global_hall_of_fame[0].formula_str}")

    # 4. 最终结果
    print("\n" + "=" * 80)
    print(f"🏆 所有挖掘结束！Top 15 超级特征")
    print("=" * 80)

    for i, ind in enumerate(global_hall_of_fame[:15]):
        print(f"Rank {i + 1:02d} | AUC: {ind.auc:.5f} | {ind.formula_str}")


if __name__ == "__main__":
    main()
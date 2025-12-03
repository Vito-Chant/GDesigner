import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

# 假设这是你的环境
from GDesigner.llm.chat import VLLMChat
from GDesigner.llm.format import Message


def analyze_segment_stats(logprobs_data: Any, full_text: str, target_text: str) -> Optional[Dict[str, Any]]:
    """
    计算目标文本片段的全套高级指标：熵、困惑度、置信度、波动率、决策裕度。
    采用严格边界检查。
    """
    if not logprobs_data or not target_text:
        return None

    # 1. 定位 Target 在 Full Text 中的绝对起止位置
    start_char_idx = full_text.find(target_text)
    if start_char_idx == -1:
        return None
    end_char_idx = start_char_idx + len(target_text)

    # 2. 兼容性提取数据
    try:
        offsets = logprobs_data.text_offset
        top_logprobs_list = logprobs_data.top_logprobs
        tokens = logprobs_data.tokens
        token_logprobs = logprobs_data.token_logprobs
    except AttributeError:
        offsets = logprobs_data.get('text_offset', [])
        top_logprobs_list = logprobs_data.get('top_logprobs', [])
        tokens = logprobs_data.get('tokens', [])
        token_logprobs = logprobs_data.get('token_logprobs', [])

    # 3. 数据收集容器
    collected_entropies = []
    collected_logprobs = []  # 实际选中的 token 的 logprob
    collected_margins = []  # Top1 - Top2 的差值
    collected_tokens = []  # 对应的 token 字符串

    # 4. 遍历并筛选 Token
    for i, offset in enumerate(offsets):
        token_str = tokens[i]

        # --- 严格边界判断 ---
        token_start = offset
        token_end = token_start + len(token_str)

        # 必须完全落在 target_text 内部，不接受跨越边界的词
        is_strictly_inside = (token_start >= start_char_idx) and (token_end <= end_char_idx)

        if is_strictly_inside:
            # A. 收集基础 Logprob
            lp = token_logprobs[i]
            if lp is None: continue  # 跳过 None (通常是第一个词)
            collected_logprobs.append(lp)
            collected_tokens.append(token_str)

            # B. 处理 Top Logprobs (用于熵和 Margin)
            top_candidates_data = top_logprobs_list[i]
            if not top_candidates_data:
                continue

            # 将 top_candidates 统一转为 list of (token, logprob)
            if isinstance(top_candidates_data, dict):
                candidates = sorted(top_candidates_data.items(), key=lambda x: x[1], reverse=True)
            else:
                # 假设是对象列表，转 tuple
                candidates = []  # 这里需根据实际对象结构调整，通常 dict 是最常见的
                pass

                # --- 计算单点熵 ---
            current_entropy = 0.0
            for _, c_lp in candidates:
                p = np.exp(c_lp)
                if p > 0:
                    current_entropy += -1 * p * c_lp
            collected_entropies.append(current_entropy)

            # --- 计算 Margin (犹豫度) ---
            if len(candidates) >= 2:
                margin = candidates[0][1] - candidates[1][1]
                collected_margins.append(margin)
            else:
                # 如果只有一个候选项（极其罕见），Margin 视为无限大或固定大值
                collected_margins.append(10.0)

    if not collected_logprobs:
        return None

    # 5. 聚合计算指标
    np_logprobs = np.array(collected_logprobs)
    np_entropies = np.array(collected_entropies)

    # 找到最弱的一环 (Min Logprob)
    min_idx = np.argmin(np_logprobs)

    stats = {
        # 序列熵 (平均不确定性)
        "avg_entropy": np.mean(np_entropies),

        # 困惑度 PPL (exp(-mean_logprob)) - 整体顺畅度
        "perplexity": np.exp(-np.mean(np_logprobs)),

        # 最小对数概率 (木桶短板) - 最大风险点
        "min_logprob": np.min(np_logprobs),
        "weakest_token": collected_tokens[min_idx],

        # 波动率 (标准差) - 难度是否均匀
        "volatility": np.std(np_logprobs),

        # 平均决策裕度 (Mean Margin) - 越大越果断
        "avg_margin": np.mean(collected_margins) if collected_margins else 0.0
    }

    return stats


async def test_roles_advanced_metrics():
    # 1. 设置 LLM
    llm = VLLMChat("Qwen/Qwen3-4B-Instruct-2507")

    # 2. 定义 Task (分析目标)
    # 使用一个稍微复杂的 Prompt 以观察不同指标的变化
    task_content = "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?"

    # 3. 定义角色
    roles = {
        "Math Expert": "You are a rigorous mathematician.",
        "Confused Poet": "You are a abstract poet who dislikes numbers.",
        "JSON Bot": "You are a robot that only outputs JSON format."
    }

    print(f"=== Advanced Metrics Analysis on Task Segment ===")
    print(f"Target Text: \"{task_content}\"\n")

    results = []

    for role_name, role_prompt in roles.items():
        messages = [
            Message(role="system", content=role_prompt),
            Message(role="user", content=task_content)
        ]

        # 4. 调用 API
        # logprobs=20 以确保熵计算准确
        full_text, logprobs_data = await llm.acomp(
            messages,
            echo=True,
            logprobs=20,
            max_tokens=1
        )

        # 5. 计算指标
        stats = analyze_segment_stats(logprobs_data, full_text, task_content)

        if stats:
            stats['role'] = role_name
            results.append(stats)

            # 打印单次详情
            print(f"--- Role: {role_name} ---")
            print(f"  System Prompt: {role_prompt[:40]}...")
            print(f"  [Entropy]     Avg: {stats['avg_entropy']:.4f}  (Higher = More 'Choices' considered)")
            print(f"  [Perplexity]  PPL: {stats['perplexity']:.2f}    (Lower = More 'Natural' fit)")
            print(f"  [Confidence]  Min: {stats['min_logprob']:.2f}    (@ Token: '{stats['weakest_token']}')")
            print(f"  [Stability]   Vol: {stats['volatility']:.4f}   (Std Dev of logprobs)")
            print(f"  [Decisiveness] Marg: {stats['avg_margin']:.4f}   (Top1 - Top2 distance)\n")
        else:
            print(f"--- Role: {role_name} ---")
            print("  Failed to extract stats (Target text not found or token alignment issue).\n")

    # 6. 最终对比表格
    print("=" * 105)
    headers = ["Role", "Entropy", "PPL", "Min_LP", "Weakest_Tok", "Volatility", "Margin"]
    print(
        f"{headers[0]:<15} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<12} | {headers[5]:<10} | {headers[6]:<8}")
    print("-" * 105)

    for r in results:
        print(
            f"{r['role']:<15} | {r['avg_entropy']:.4f}   | {r['perplexity']:.2f}     | {r['min_logprob']:.2f}     | {r['weakest_token']:<12} | {r['volatility']:.4f}     | {r['avg_margin']:.4f}")
    print("=" * 105)


if __name__ == "__main__":
    asyncio.run(test_roles_advanced_metrics())


# import asyncio
# import numpy as np
# from typing import List, Dict, Any
#
# # 导入 GDesigner 的相关模块 (假设环境中有)
# from GDesigner.llm.chat import VLLMChat
# from GDesigner.llm.format import Message
#
#
# def calculate_segment_entropy(logprobs_data: Any, full_text: str, target_text: str) -> float:
#     """
#     针对 Completions 接口返回的 logprobs 结构计算特定文本片段的平均香农熵。
#
#     参数:
#         logprobs_data: client.completions.create 返回的 logprobs 对象 (包含 tokens, top_logprobs, text_offset 等列表)
#         full_text: 包含 Prompt 和 Response 的完整文本 (echo=True 返回的内容)
#         target_text: 我们想要分析的那一段文本 (即 Task 内容)
#     """
#     if not logprobs_data or not target_text:
#         return 0.0
#
#     # 1. 在完整文本中定位目标片段的起止位置 (字符级别)
#     # 注意：这里我们查找 Task 文本在 Full Text 中的位置
#     start_char_idx = full_text.find(target_text)
#     if start_char_idx == -1:
#         print(f"Warning: Target text not found in full text response.")
#         return 0.0
#
#     end_char_idx = start_char_idx + len(target_text)
#
#     # 2. 获取 Logprobs 数据列
#     # 注意：Completions 接口返回的 logprobs 是对象，属性是列表
#     # 如果 logprobs_data 是字典取 key，如果是对象取属性，这里做个兼容
#     try:
#         offsets = logprobs_data.text_offset
#         top_logprobs_list = logprobs_data.top_logprobs
#         tokens = logprobs_data.tokens
#     except AttributeError:
#         # 兼容字典访问
#         offsets = logprobs_data.get('text_offset', [])
#         top_logprobs_list = logprobs_data.get('top_logprobs', [])
#         tokens = logprobs_data.get('tokens', [])
#
#     total_entropy = 0.0
#     valid_token_count = 0
#
#     # 3. 遍历所有 Token，筛选出位于目标文本范围内的 Token
#     for i, offset in enumerate(offsets):
#         # token_str = tokens[i]
#         #
#         # # --- [修改点] 严格的边界计算 ---
#         # token_start = offset
#         # token_end = token_start + len(token_str)  # Token 在 full_text 中的结束位置
#         #
#         # # 判断条件：
#         # # 1. Token 的起始位置必须 >= Target 的起始位置
#         # # 2. Token 的结束位置必须 <= Target 的结束位置
#         # is_strictly_inside = (token_start >= start_char_idx) and (token_end <= end_char_idx)
#         #
#         # if is_strictly_inside:
#         #     token_top_candidates = top_logprobs_list[i]
#         #     if not token_top_candidates:
#         #         continue
#
#         # 简单的范围判断：如果 Token 的起始偏移量在目标范围内
#         # (稍微放宽一点边界，确保包含文本主体)
#         if offset >= start_char_idx and offset < end_char_idx:
#
#             # 获取该位置的 Top Logprobs 字典 (例如前 5 个候选项)
#             token_top_candidates = top_logprobs_list[i]
#
#             if not token_top_candidates:
#                 continue
#
#             # --- 计算香农熵 H(x) = - sum(p * log(p)) ---
#             current_entropy = 0.0
#
#             # token_top_candidates 可能是 dict (token -> logprob) 或 list
#             # OpenAI Python SDK v1+ 通常返回 dict
#             iterator = token_top_candidates.items() if isinstance(token_top_candidates, dict) else token_top_candidates
#
#             for token_str, lp in iterator:
#                 p = np.exp(lp)
#                 if p > 0:
#                     current_entropy += -1 * p * lp
#
#             total_entropy += current_entropy
#             valid_token_count += 1
#
#             # Debug: 打印每个 Token 的熵，看看哪里模型最困惑
#             # print(f"  Token: {tokens[i]:<10} | Entropy: {current_entropy:.4f}")
#
#     if valid_token_count == 0:
#         return 0.0
#
#     return total_entropy / valid_token_count
#
#
# async def test_roles_entropy_on_task():
#     # 1. 初始化模型
#     llm = VLLMChat("Qwen/Qwen3-4B-Instruct-2507")
#
#     # 2. 定义 Task (我们要分析这一段文本的序列熵)
#     # task_content = "Solve this equation: x^2 - 5x + 6 = 0. output the roots directly."
#     task_content = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
#
#     # 3. 定义不同的角色
#     roles = {
#         "Math Expert": "You are a rigorous mathematician.",
#         "Confused Poet": "You are a abstract poet who speaks in metaphors and dislikes logic.",
#         "JSON Machine": "You are a robot that only outputs JSON."
#     }
#
#     print(f"--- Analyzing Task Entropy: '{task_content}' ---\n")
#     print(f"{'Role':<20} | {'Avg Entropy':<12} | {'Interpretation'}")
#     print("-" * 60)
#
#     for role_name, role_prompt in roles.items():
#         messages = [
#             Message(role="system", content=role_prompt),
#             Message(role="user", content=task_content)
#         ]
#
#         # 4. 调用 llm.acomp (底层是 completions.create)
#         # 关键参数：
#         # - echo=True: 必须开启，否则无法获取 Prompt 部分的 logprobs
#         # - logprobs=20: 必须足够大！如果设为 1，熵永远接近 0 (因为只有 1 个样本)。设为 5-20 能较好地反映分布的平坦程度。
#         # - max_tokens=1: 我们只关心输入部分的熵，不需要它真的生成回答，设为 1 节省计算资源。
#
#         full_text, logprobs_data = await llm.acomp(
#             messages,
#             echo=True,
#             logprobs=20,  # 重要：计算熵需要概率分布，至少取 Top 5 或 Top 20
#             max_tokens=1
#         )
#
#         if logprobs_data:
#             # 5. 计算 Task 部分的序列熵
#             # 我们只关注 task_content 这段文字对应的 Token 熵
#             avg_entropy = calculate_segment_entropy(logprobs_data, full_text, task_content)
#
#             # 解读：熵越高，说明模型在阅读这个 Task 时，认为下一个词越“不可预测”。
#             # 通常 System Prompt 越符合 Task 的语境，Task 的熵应该越低（PPL 越低）。
#             interpretation = "More Uncertain" if avg_entropy > 1.0 else "Stable"
#
#             print(f"{role_name:<20} | {avg_entropy:.5f}      | {interpretation}")
#         else:
#             print(f"{role_name:<20} | N/A          | Failed to get logprobs")
#
#
# if __name__ == "__main__":
#     asyncio.run(test_roles_entropy_on_task())
#
#     # llm = VLLMChat("Qwen/Qwen3-4B-Instruct-2507")
#     # messages = [{"role": "user", "content": "hello"}]
#     # content, logprobs_data = asyncio.run(llm.acomp(messages, logprobs=1, echo=True))
#     #
#     # print(f"Content: {content}\n")
#     # print(f"Logprobs: {logprobs_data}\n")
#     #
#     # if logprobs_data:
#     #     for item in logprobs_data:
#     #         token = item.token
#     #         logprob = item.logprob
#     #         print(f"Token: {token:<15} Logprob: {logprob:.4f}")

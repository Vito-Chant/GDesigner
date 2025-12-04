import asyncio
import numpy as np
import pandas as pd
import re
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm

# 假设环境设置
from GDesigner.llm.chat import VLLMChat
from GDesigner.llm.format import Message
from dataset.mmlu_dataset import MMLUDataset
from GDesigner.prompt.mmlu_prompt_set import MMLUPromptSet, ROLE_DESCRIPTION

# === 配置项 ===
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_SPLIT = "dev"
LIMIT_SAMPLES = 300  # 样本数
LOGPROBS_K = 20
TEMPERATURE = 0
MAX_TOKENS = 1000

# === [关键新增] 并发设置 ===
CONCURRENCY_LIMIT = 50  # 建议根据显存大小调整。32-64 是比较安全的范围。


# 因为 logprobs=20 比较吃显存，设太大可能会 OOM。

def analyze_segment_stats(logprobs_data: Any, full_text: str, target_text: str) -> Optional[Dict[str, Any]]:
    # ... (保持原本的 analyze_segment_stats 代码不变) ...
    # 为了节省篇幅，这里省略具体实现，请直接保留你原来的代码
    if not logprobs_data or not target_text:
        return None
    start_char_idx = full_text.find(target_text)
    if start_char_idx == -1: return None
    end_char_idx = start_char_idx + len(target_text)
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

    collected_entropies = []
    collected_logprobs = []
    collected_margins = []

    for i, offset in enumerate(offsets):
        token_str = tokens[i]
        token_start = offset
        token_end = token_start + len(token_str)
        is_strictly_inside = (token_start >= start_char_idx) and (token_end <= end_char_idx)

        if is_strictly_inside:
            lp = token_logprobs[i]
            if lp is None: continue
            collected_logprobs.append(lp)
            top_candidates_data = top_logprobs_list[i]
            if not top_candidates_data: continue

            if isinstance(top_candidates_data, dict):
                candidates = sorted(top_candidates_data.items(), key=lambda x: x[1], reverse=True)
            else:
                candidates = []
                pass

            current_entropy = 0.0
            for _, c_lp in candidates:
                p = np.exp(c_lp)
                if p > 0:
                    current_entropy += -1 * p * c_lp
            collected_entropies.append(current_entropy)

            if len(candidates) >= 2:
                margin = candidates[0][1] - candidates[1][1]
                collected_margins.append(margin)
            else:
                collected_margins.append(10.0)

    if not collected_logprobs: return None
    np_logprobs = np.array(collected_logprobs)
    np_entropies = np.array(collected_entropies)
    stats = {
        "avg_entropy": np.mean(np_entropies),
        "perplexity": np.exp(-np.mean(np_logprobs)),
        "volatility": np.std(np_logprobs),
        "avg_margin": np.mean(collected_margins) if collected_margins else 0.0
    }
    return stats


def extract_answer(response_text: str) -> str:
    # ... (保持原本的 extract_answer 代码不变) ...
    match = re.search(r"(?:Answer|answer)(?:\s+is)?\s*[:\s]*([ABCD])\b", response_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"^\s*([ABCD])\b", response_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", response_text)
    if match: return match.group(1).upper()
    return ""


# === [新增] 单个任务的处理函数 ===
async def process_single_entry(
        semaphore: asyncio.Semaphore,
        llm: VLLMChat,
        sample_id: int,
        record: Any,
        role: str,
        role_prompt: str,
        pbar: tqdm
) -> Optional[Dict[str, Any]]:
    # 获取 Semaphore，限制并发数
    async with semaphore:
        try:
            input_data = MMLUDataset.record_to_input(record)
            task_content = input_data['task']
            ground_truth = MMLUDataset.record_to_target_answer(record)

            final_system_prompt = role_prompt + "\nPlease answer with just the letter A, B, C, or D at the start."

            messages = [
                Message(role="system", content=final_system_prompt),
                Message(role="user", content=task_content)
            ]

            # 这里的调用是并发的关键
            full_text, logprobs_data = await llm.acomp(
                messages,
                echo=True,
                logprobs=LOGPROBS_K,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            stats = analyze_segment_stats(logprobs_data, full_text, task_content)

            if stats:
                task_end_idx = full_text.find(task_content) + len(task_content)
                generated_reply = full_text[task_end_idx:].strip()
                prediction = extract_answer(generated_reply)
                is_correct = (prediction == ground_truth)

                return {
                    "sample_id": sample_id,
                    "role": role,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "is_correct": int(is_correct),
                    "avg_entropy": stats["avg_entropy"],
                    "perplexity": stats["perplexity"],
                    "volatility": stats["volatility"],
                    "avg_margin": stats["avg_margin"]
                }

        except Exception as e:
            # 打印简略错误，避免刷屏
            print(f"\n[Error] ID {sample_id} - {role}: {str(e)[:100]}")
            return None
        finally:
            # 更新进度条
            pbar.update(1)


async def run_analysis():
    print(f"Loading Model: {MODEL_NAME}...")
    llm = VLLMChat(MODEL_NAME)

    print(f"Loading Dataset: MMLU ({DATA_SPLIT})...")
    dataset = MMLUDataset(split=DATA_SPLIT)

    roles_dict = ROLE_DESCRIPTION
    role_names = list(roles_dict.keys())

    total_samples = len(dataset)
    if LIMIT_SAMPLES:
        total_samples = min(total_samples, LIMIT_SAMPLES)
        print(f"Limiting to first {total_samples} samples.")

    # 准备任务列表
    tasks = []
    # 创建信号量，控制同时只有 CONCURRENCY_LIMIT 个请求在跑
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 计算总任务数用于进度条
    total_tasks_count = total_samples * len(role_names)
    print(f"Total Tasks: {total_tasks_count} (Concurrency: {CONCURRENCY_LIMIT})")

    # 初始化进度条
    pbar = tqdm(total=total_tasks_count, desc="Processing")

    # 构建所有任务
    for i in range(total_samples):
        record = dataset[i]
        for role in role_names:
            task = process_single_entry(
                semaphore,
                llm,
                i,
                record,
                role,
                roles_dict[role],
                pbar
            )
            tasks.append(task)

    print("Starting Evaluation Loop (Async Batching)...")

    # 执行所有任务
    # asyncio.gather 会等待所有任务完成
    results_with_none = await asyncio.gather(*tasks)

    pbar.close()

    # 过滤掉 None (失败的任务)
    results = [r for r in results_with_none if r is not None]

    # === 统计与分析 ===
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("ANALYSIS REPORT")
    print("=" * 80)

    role_summary = df.groupby("role").agg({
        "is_correct": "mean",
        "avg_entropy": "mean",
        "perplexity": "mean",
        "volatility": "mean",
        "avg_margin": "mean"
    }).sort_values("is_correct", ascending=False)

    print("\n--- Summary by Role ---")
    print(role_summary)

    # ... (保持原本的统计部分代码不变) ...
    # 为了方便，这里只写保存部分
    df.to_csv("mmlu_role_entropy_analysis.csv", index=False)
    print("\nFull results saved to 'mmlu_role_entropy_analysis.csv'")


if __name__ == "__main__":
    asyncio.run(run_analysis())
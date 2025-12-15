import asyncio
import numpy as np
import pandas as pd
import re
import os
import pickle
import torch
import gc
from typing import Dict, Any, Optional, List, Tuple
from tqdm.asyncio import tqdm

# === 假设环境设置 (请确保这些包已安装/路径正确) ===
try:
    from GDesigner.llm.chat import VLLMChat
    from GDesigner.llm.format import Message
    from dataset.mmlu_dataset import MMLUDataset
    from GDesigner.prompt.mmlu_prompt_set import MMLUPromptSet, ROLE_DESCRIPTION
except ImportError as e:
    print("Warning: 缺少项目特定依赖 (GDesigner/dataset)，请确保在正确的项目环境下运行。")
    print(f"Error details: {e}")


    # 为了防止代码直接报错退出，这里定义一些 Mock 类仅供结构参考（实际运行时请删除或忽略）
    class Mock:
        pass


    VLLMChat = Mock
    MMLUDataset = Mock
    MMLUPromptSet = Mock
    ROLE_DESCRIPTION = {"Default": "You are a helpful assistant."}

ROLE_DESCRIPTION["Normal"] = ""

# === 配置项 ===
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_SPLIT = "val"
LIMIT_SAMPLES = 153  # 设为 None 跑全量
LOGPROBS_K = 20  # 必须 >= 20
TEMPERATURE = 0
MAX_TOKENS = 2000

# === 并发与存储设置 ===
CONCURRENCY_LIMIT = 50
SAVE_DIR = "validation_data_v3"
SAVE_EVERY = 100


def extract_logprobs_matrix(logprobs_data: Any, full_text: str, target_text: str) -> Optional[
    Tuple[np.ndarray, np.ndarray]]:
    """
    提取目标文本片段的特征。

    Returns:
        tuple: (
            top_matrix: np.ndarray [Sequence_Length, 20],
            gt_vector: np.ndarray [Sequence_Length]
        )
    """
    if not logprobs_data or not target_text:
        return None

    # 1. 定位 Target (题目) 在 Full Text 中的位置
    start_char_idx = full_text.find(target_text)
    if start_char_idx == -1:
        return None
    end_char_idx = start_char_idx + len(target_text)

    # 2. 兼容性提取 (适配 vLLM / OpenAI 格式)
    try:
        offsets = logprobs_data.text_offset
        top_logprobs_list = logprobs_data.top_logprobs  # list of dict
        tokens = logprobs_data.tokens
        # === 新增：提取 Ground Truth Logprobs ===
        token_logprobs = logprobs_data.token_logprobs
    except AttributeError:
        # 字典访问模式
        offsets = logprobs_data.get('text_offset', [])
        top_logprobs_list = logprobs_data.get('top_logprobs', [])
        tokens = logprobs_data.get('tokens', [])
        token_logprobs = logprobs_data.get('token_logprobs', [])

    collected_top_vectors = []
    collected_gt_scalars = []

    # 3. 遍历 Token
    for i, offset in enumerate(offsets):
        token_str = tokens[i]
        token_start = offset
        token_end = token_start + len(token_str)

        # 严格判断：只收集落在题目范围内的 Token Logprobs
        is_strictly_inside = (token_start >= start_char_idx) and (token_end <= end_char_idx)

        if is_strictly_inside:
            # === Part A: 提取 Top-20 矩阵 ===
            top_candidates_data = top_logprobs_list[i]

            if not top_candidates_data:
                vec = np.full(LOGPROBS_K, -100.0, dtype=np.float32)
            else:
                if isinstance(top_candidates_data, dict):
                    values = list(top_candidates_data.values())
                else:
                    values = [x.logprob for x in top_candidates_data]

                values.sort(reverse=True)  # 降序排列

                if len(values) >= LOGPROBS_K:
                    vec = np.array(values[:LOGPROBS_K], dtype=np.float32)
                else:
                    vec = np.pad(values, (0, LOGPROBS_K - len(values)), 'constant', constant_values=-100.0).astype(
                        np.float32)

            collected_top_vectors.append(vec)

            # === Part B: 提取 Ground Truth Logprob ===
            # token_logprobs[i] 对应的是当前实际 Token 的 Logprob
            gt_val = token_logprobs[i] if i < len(token_logprobs) else -100.0
            collected_gt_scalars.append(gt_val)

    if not collected_top_vectors:
        return None

    # 转换为 Numpy 数组
    # features: [Seq_Len, 20]
    # gt_logprobs: [Seq_Len]
    return np.stack(collected_top_vectors), np.array(collected_gt_scalars, dtype=np.float32)


def extract_answer(response_text: str) -> str:
    match = re.search(r"(?:Answer|answer)(?:\s+is)?\s*[:\s]*([ABCD])\b", response_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"^\s*([ABCD])\b", response_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", response_text)
    if match: return match.group(1).upper()
    return ""


async def process_single_entry(
        semaphore: asyncio.Semaphore,
        llm: VLLMChat,
        sample_id: int,
        record: Any,
        role: str,
        role_prompt: str,
        pbar: tqdm
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        try:
            input_data = MMLUDataset.record_to_input(record)
            task_content = input_data['task']
            ground_truth = MMLUDataset.record_to_target_answer(record)

            final_system_prompt = role_prompt + """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Using the reasoning from other agents as additional advice with critical thinking, can you give an updated answer?
You are strictly prohibited from imitating the analysis process of other agents
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter(for example : A, B, C or D)
"""
            messages = [
                Message(role="system", content=final_system_prompt),
                Message(role="user", content=task_content)
            ]

            full_text, logprobs_data = await llm.acomp(
                messages,
                echo=True,
                logprobs=LOGPROBS_K,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            # === 提取特征 (矩阵 + GT向量) ===
            extraction_result = extract_logprobs_matrix(logprobs_data, full_text, task_content)

            if extraction_result is not None:
                features_matrix, gt_vector = extraction_result

                # 提取答案判断对错
                task_end_idx = full_text.find(task_content) + len(task_content)
                generated_reply = full_text[task_end_idx:].strip()
                prediction = extract_answer(generated_reply)
                is_correct = (prediction == ground_truth)

                return {
                    "sample_id": sample_id,
                    "role": role,
                    "is_correct": int(is_correct),
                    "seq_len": features_matrix.shape[0],
                    "features": features_matrix,  # [L, 20] Top-20 Logprobs
                    "gt_logprobs": gt_vector  # [L]     Actual Token Logprobs
                }

        except Exception as e:
            # print(f"[Error] {sample_id}-{role}: {e}")
            return None
        finally:
            pbar.update(1)


async def run_data_collection():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    print(f"Loading Model: {MODEL_NAME}...")
    try:
        llm = VLLMChat(MODEL_NAME)
    except NameError:
        print("Error: VLLMChat 未定义，请检查导入。")
        return

    print(f"Loading Dataset: MMLU ({DATA_SPLIT})...")
    dataset = MMLUDataset(split=DATA_SPLIT)
    roles_dict = ROLE_DESCRIPTION
    role_names = list(roles_dict.keys())

    total_samples = len(dataset)
    if LIMIT_SAMPLES:
        total_samples = min(total_samples, LIMIT_SAMPLES)
        print(f"Limiting to first {total_samples} samples.")

    # 任务队列
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    total_tasks_count = total_samples * len(role_names)

    print(f"Total Tasks: {total_tasks_count}. Collecting Raw Matrix + GT Data...")
    pbar = tqdm(total=total_tasks_count, desc="Collecting")

    collected_batch = []
    batch_index = 0
    CHUNK_SIZE = 50

    for i in range(0, total_samples, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, total_samples)
        current_chunk_tasks = []

        for sample_idx in range(i, chunk_end):
            record = dataset[sample_idx]
            for role in role_names:
                task = process_single_entry(
                    semaphore, llm, sample_idx, record, role, roles_dict[role], pbar
                )
                current_chunk_tasks.append(task)

        results = await asyncio.gather(*current_chunk_tasks)
        valid_results = [r for r in results if r is not None]
        collected_batch.extend(valid_results)

        # 保存逻辑
        if len(collected_batch) >= SAVE_EVERY:
            save_path = os.path.join(SAVE_DIR, f"batch_{batch_index}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(collected_batch, f)
            print(f"\nSaved {len(collected_batch)} records to {save_path}")

            collected_batch = []
            batch_index += 1
            gc.collect()

    if collected_batch:
        save_path = os.path.join(SAVE_DIR, f"batch_{batch_index}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(collected_batch, f)
        print(f"\nSaved final {len(collected_batch)} records to {save_path}")

    pbar.close()
    print("Data collection complete.")


if __name__ == "__main__":
    asyncio.run(run_data_collection())

"""
多LLM加权投票多智能体系统 V2
支持同构/异构LLM配置、扫描模式、详细数据记录

Features:
- 同构LLM: 指定单个LLM名字和数量
- 异构LLM: 指定LLM名字列表
- 扫描模式: 自动测试1到N个LLM的投票结果
- 详细数据记录: 保存每道题的投票分布、置信度等元数据
- WandB集成: 实时记录和可视化

Usage:
    # 同构LLM (5个相同模型)
    python run_multi_llm_voting_v2.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5

    # 异构LLM (指定不同模型列表)
    python run_multi_llm_voting_v2.py --heterogeneous --llm_names "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"

    # 扫描模式 (测试1到N个LLM)
    python run_multi_llm_voting_v2.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 10 --scan_mode
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import argparse
import time
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from tqdm import tqdm
import math

import weave

# 导入项目依赖
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


# ============================================================================
# 数据类定义 - 用于结构化存储实验数据
# ============================================================================

@dataclass
class AgentVote:
    """单个智能体的投票记录"""
    agent_id: str
    llm_name: str
    weight: float
    raw_response: str
    extracted_answer: str
    response_time: float
    is_correct: bool = False


@dataclass
class QuestionRecord:
    """单道题目的完整记录"""
    question_id: int
    question_text: str
    correct_answer: str
    agent_votes: List[AgentVote] = field(default_factory=list)

    # 投票统计
    final_answer: str = ""
    is_correct: bool = False
    vote_distribution: Dict[str, float] = field(default_factory=dict)  # answer -> weighted score
    raw_vote_counts: Dict[str, int] = field(default_factory=dict)  # answer -> count

    # 一致性指标
    is_unanimous: bool = False
    agreement_ratio: float = 0.0  # 最高票答案的占比
    entropy: float = 0.0  # 投票分布的熵

    # 时间
    total_time: float = 0.0


@dataclass
class ScanResult:
    """扫描模式下某个agent数量的结果"""
    num_agents: int
    agent_ids: List[str]
    accuracy: float
    correct_count: int
    total_count: int
    unanimous_ratio: float
    avg_agreement_ratio: float
    avg_time: float


@dataclass
class ExperimentMetadata:
    """实验元数据"""
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]

    # LLM配置
    llm_configs: List[Tuple[str, float]]
    is_homogeneous: bool

    # 数据集信息
    dataset_name: str
    dataset_split: str
    total_questions: int

    # 结果汇总
    question_records: List[QuestionRecord] = field(default_factory=list)
    scan_results: List[ScanResult] = field(default_factory=list)

    # 性能指标
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time: float = 0.0


# ============================================================================
# 核心类
# ============================================================================

class VotingAgent:
    """单个投票智能体"""

    def __init__(self, agent_id: str, llm_name: str, weight: float = 1.0, temperature=0.7, enable_thinking=True):
        self.agent_id = agent_id
        self.llm_name = llm_name
        self.weight = weight
        self.llm = LLMRegistry.get(llm_name)
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    async def vote(self, question: str) -> Tuple[str, float]:
        """对问题进行投票，返回(原始响应, 响应时间)"""

        system_prompt = """You are an expert at multiple-choice questions.
You will be given a question with 4 options (A, B, C, D).
Only one answer is correct.

IMPORTANT OUTPUT FORMAT:
1. You can use <think>...</think> tags to show your reasoning process
2. After your thinking, you MUST output your final answer in this EXACT format:
   **Answer: X**
   where X is one of A, B, C, or D

Example output format:
<think>
...
</think>

**Answer: B**"""

        user_prompt = f"{question}\n\nRemember: End your response with **Answer: X** where X is your chosen letter."

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        start_time = time.time()
        # response = await self.llm.agen(messages, temperature=self.temperature)
        response = await self.llm.acomp(messages, temperature=self.temperature, enable_thinking=self.enable_thinking)
        elapsed_time = time.time() - start_time

        return response, elapsed_time


class MultiLLMVotingSystemV2:
    """多LLM加权投票系统 V2"""

    def __init__(self, llm_configs: List[Tuple[str, float]], temperature=0.7, enable_thinking=True):
        """
        Args:
            llm_configs: List of (llm_name, weight) tuples
        """
        self.agents: List[VotingAgent] = []

        for idx, (llm_name, weight) in enumerate(llm_configs):
            agent_id = f"agent_{idx}_{llm_name.split('/')[-1]}"
            agent = VotingAgent(agent_id, llm_name, weight, temperature=temperature, enable_thinking=enable_thinking)
            self.agents.append(agent)

        # 归一化权重
        total_weight = sum(agent.weight for agent in self.agents)
        if total_weight > 0:
            for agent in self.agents:
                agent.weight /= total_weight

    def get_subset(self, num_agents: int) -> 'MultiLLMVotingSystemV2':
        """获取前n个agent的子集"""
        subset_configs = [(agent.llm_name, agent.weight) for agent in self.agents[:num_agents]]
        return MultiLLMVotingSystemV2(subset_configs)

    async def vote_on_question(
            self,
            question_id: int,
            question: str,
            correct_answer: str,
            active_agent_ids: Optional[List[str]] = None
    ) -> QuestionRecord:
        """
        对单个问题进行投票

        Args:
            question_id: 问题ID
            question: 问题文本
            correct_answer: 正确答案
            active_agent_ids: 参与投票的agent ID列表，None表示全部参与
        """
        record = QuestionRecord(
            question_id=question_id,
            question_text=question[:500],  # 截断保存
            correct_answer=correct_answer
        )

        start_time = time.time()

        # 确定参与的agents
        active_agents = self.agents
        if active_agent_ids:
            active_agents = [a for a in self.agents if a.agent_id in active_agent_ids]

        # 并发收集投票
        tasks = [agent.vote(question) for agent in active_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        vote_scores = defaultdict(float)
        vote_counts = defaultdict(int)

        for agent, result in zip(active_agents, results):
            if isinstance(result, Exception):
                vote = AgentVote(
                    agent_id=agent.agent_id,
                    llm_name=agent.llm_name,
                    weight=agent.weight,
                    raw_response=f"ERROR: {str(result)}",
                    extracted_answer="ERROR",
                    response_time=0.0,
                    is_correct=False
                )
            else:
                response, resp_time = result
                extracted = self._extract_answer(response)
                vote = AgentVote(
                    agent_id=agent.agent_id,
                    llm_name=agent.llm_name,
                    weight=agent.weight,
                    raw_response=response,
                    extracted_answer=extracted,
                    response_time=resp_time,
                    is_correct=(extracted == correct_answer)
                )

                if extracted not in ["INVALID", "ERROR"]:
                    vote_scores[extracted] += agent.weight
                    vote_counts[extracted] += 1

            record.agent_votes.append(vote)

        # 计算最终答案和统计
        record.vote_distribution = dict(vote_scores)
        record.raw_vote_counts = dict(vote_counts)

        if vote_scores:
            record.final_answer = max(vote_scores.items(), key=lambda x: x[1])[0]
        else:
            record.final_answer = "INVALID"

        record.is_correct = (record.final_answer == correct_answer)

        # 一致性指标
        total_valid_votes = sum(vote_counts.values())
        if total_valid_votes > 0:
            max_votes = max(vote_counts.values())
            record.is_unanimous = (len(vote_counts) == 1)
            record.agreement_ratio = max_votes / total_valid_votes

            # 计算熵
            probs = [c / total_valid_votes for c in vote_counts.values()]
            record.entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)

        record.total_time = time.time() - start_time

        return record

    def _extract_answer(self, response: str) -> str:
        """从回复中提取答案字母"""

        # 策略1: **Answer: X** 格式
        match = re.search(r'\*\*Answer:\s*([A-D])\*\*', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 策略2: Answer: X 格式
        match = re.search(r'(?:Answer|答案):\s*([A-D])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 策略3: </think>后的内容
        think_split = response.split('</think>')
        if len(think_split) > 1:
            after_think = think_split[-1]
            match = re.search(r'(?:^|\s|[.!?\n])\s*([A-D])(?:\s|[.!?,\n]|$)', after_think, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).upper()
            for char in after_think:
                if char.upper() in ['A', 'B', 'C', 'D']:
                    return char.upper()

        # 策略4: 最后一行的独立字母
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if len(line) == 1 and line.upper() in ['A', 'B', 'C', 'D']:
                return line.upper()

        # 策略5: "X is correct" 模式
        matches = re.findall(r'([A-D])\s*(?:is|为|是)\s*(?:correct|right|正确)', response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

        # 策略6: 所有独立字母中的最后一个
        all_letters = re.findall(r'(?:^|\s|[.!?\n])\s*([A-D])(?:\s|[.!?,\n]|$)', response, re.MULTILINE | re.IGNORECASE)
        if all_letters:
            return all_letters[-1].upper()

        # 策略7: 文本中第一个字母
        for char in response:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()

        return "INVALID"


# ============================================================================
# 实验运行器
# ============================================================================

class ExperimentRunner:
    """实验运行器"""

    def __init__(
            self,
            llm_configs: List[Tuple[str, float]],
            dataset,
            is_homogeneous: bool = True,
            scan_mode: bool = False,
            wandb_run=None,
            temperature=0.7,
            enable_thinking=True
    ):
        self.llm_configs = llm_configs
        self.dataset = dataset
        self.is_homogeneous = is_homogeneous
        self.scan_mode = scan_mode
        self.wandb_run = wandb_run

        # 初始化投票系统
        self.voting_system = MultiLLMVotingSystemV2(llm_configs, temperature=temperature,
                                                    enable_thinking=enable_thinking)

        # 实验元数据
        self.experiment_id = time.strftime("%Y%m%d_%H%M%S")
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={},
            llm_configs=llm_configs,
            is_homogeneous=is_homogeneous,
            dataset_name="MMLU",
            dataset_split=dataset.split,
            total_questions=0
        )

    async def run(
            self,
            limit_questions: Optional[int] = None,
            batch_size: int = 4,
            debug_first_n: int = 0
    ) -> ExperimentMetadata:
        """
        运行实验

        Args:
            limit_questions: 限制问题数量
            batch_size: 批处理大小
            debug_first_n: 前N个问题启用调试
        """
        total_questions = min(len(self.dataset), limit_questions) if limit_questions else len(self.dataset)
        self.metadata.total_questions = total_questions

        print(f"\n{'=' * 80}")
        print(f"RUNNING EXPERIMENT: {self.experiment_id}")
        print(f"{'=' * 80}")
        print(f"Total Agents: {len(self.voting_system.agents)}")
        print(f"Total Questions: {total_questions}")
        print(f"Scan Mode: {self.scan_mode}")
        print(f"{'=' * 80}\n")

        # 重置计数器
        Cost.instance().reset()
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()

        start_time = time.time()

        # 收集所有问题的记录
        question_records = []
        num_batches = math.ceil(total_questions / batch_size)

        for batch_idx in tqdm(range(num_batches), desc="Processing"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_questions)

            batch_tasks = []
            for idx in range(start_idx, end_idx):
                record = self.dataset[idx]
                input_dict = self.dataset.record_to_input(record)
                question = input_dict['task']
                correct_answer = self.dataset.record_to_target_answer(record)

                task = self.voting_system.vote_on_question(
                    question_id=idx,
                    question=question,
                    correct_answer=correct_answer
                )
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Error: {result}")
                else:
                    question_records.append(result)

                    # WandB实时记录
                    if self.wandb_run:
                        self.wandb_run.log({
                            "question/is_correct": int(result.is_correct),
                            "question/agreement_ratio": result.agreement_ratio,
                            "question/entropy": result.entropy,
                            "question/is_unanimous": int(result.is_unanimous),
                            "question/time": result.total_time
                        })

            # 打印进度
            if (batch_idx + 1) % 5 == 0:
                correct = sum(1 for r in question_records if r.is_correct)
                print(f"\nProgress: {len(question_records)}/{total_questions}, "
                      f"Accuracy: {correct / len(question_records):.2%}")

        self.metadata.question_records = question_records
        self.metadata.total_time = time.time() - start_time
        self.metadata.total_cost = Cost.instance().value
        self.metadata.total_prompt_tokens = int(PromptTokens.instance().value)
        self.metadata.total_completion_tokens = int(CompletionTokens.instance().value)

        # 扫描模式：计算不同agent数量的结果
        if self.scan_mode:
            self._compute_scan_results()

        # 打印汇总
        self._print_summary()

        # WandB记录汇总
        if self.wandb_run:
            self._log_to_wandb()

        return self.metadata

    def _compute_scan_results(self):
        """计算扫描模式的结果"""
        print("\n" + "=" * 80)
        print("COMPUTING SCAN RESULTS")
        print("=" * 80)

        num_agents = len(self.voting_system.agents)

        for n in range(1, num_agents + 1):
            # 获取前n个agent的ID
            agent_ids = [agent.agent_id for agent in self.voting_system.agents[:n]]

            # 重新计算每道题的结果
            correct_count = 0
            unanimous_count = 0
            total_agreement = 0.0
            total_time = 0.0

            for record in self.metadata.question_records:
                # 筛选前n个agent的投票
                subset_votes = [v for v in record.agent_votes if v.agent_id in agent_ids]

                # 重新计算加权得分
                vote_scores = defaultdict(float)
                vote_counts = defaultdict(int)
                total_weight = sum(v.weight for v in subset_votes)

                for vote in subset_votes:
                    if vote.extracted_answer not in ["INVALID", "ERROR"]:
                        normalized_weight = vote.weight / total_weight if total_weight > 0 else 0
                        vote_scores[vote.extracted_answer] += normalized_weight
                        vote_counts[vote.extracted_answer] += 1

                # 最终答案
                if vote_scores:
                    final_answer = max(vote_scores.items(), key=lambda x: x[1])[0]
                    is_correct = (final_answer == record.correct_answer)
                else:
                    is_correct = False

                correct_count += int(is_correct)

                # 一致性
                total_valid = sum(vote_counts.values())
                if total_valid > 0:
                    unanimous_count += int(len(vote_counts) == 1)
                    total_agreement += max(vote_counts.values()) / total_valid

                total_time += record.total_time

            total_questions = len(self.metadata.question_records)

            scan_result = ScanResult(
                num_agents=n,
                agent_ids=agent_ids,
                accuracy=correct_count / total_questions if total_questions > 0 else 0,
                correct_count=correct_count,
                total_count=total_questions,
                unanimous_ratio=unanimous_count / total_questions if total_questions > 0 else 0,
                avg_agreement_ratio=total_agreement / total_questions if total_questions > 0 else 0,
                avg_time=total_time / total_questions if total_questions > 0 else 0
            )

            self.metadata.scan_results.append(scan_result)

            print(f"  Agents={n}: Accuracy={scan_result.accuracy:.2%} "
                  f"({correct_count}/{total_questions}), "
                  f"Unanimous={scan_result.unanimous_ratio:.1%}")

    def _print_summary(self):
        """打印实验汇总"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        records = self.metadata.question_records
        correct = sum(1 for r in records if r.is_correct)
        unanimous = sum(1 for r in records if r.is_unanimous)

        print(f"\n📊 Overall Performance:")
        print(f"  Accuracy: {correct / len(records):.2%} ({correct}/{len(records)})")
        print(f"  Unanimous Votes: {unanimous / len(records):.1%}")
        print(f"  Avg Agreement: {sum(r.agreement_ratio for r in records) / len(records):.2%}")
        print(f"  Avg Entropy: {sum(r.entropy for r in records) / len(records):.3f}")

        print(f"\n🤖 Per-Agent Performance:")
        agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for record in records:
            for vote in record.agent_votes:
                agent_stats[vote.agent_id]['total'] += 1
                if vote.is_correct:
                    agent_stats[vote.agent_id]['correct'] += 1

        for agent_id in sorted(agent_stats.keys()):
            stats = agent_stats[agent_id]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {agent_id}: {acc:.2%} ({stats['correct']}/{stats['total']})")

        print(f"\n💰 Cost:")
        print(f"  Total Cost: ${self.metadata.total_cost:.4f}")
        print(f"  Prompt Tokens: {self.metadata.total_prompt_tokens / 1000:.1f}k")
        print(f"  Completion Tokens: {self.metadata.total_completion_tokens / 1000:.1f}k")
        print(f"  Total Time: {self.metadata.total_time:.1f}s")

        if self.metadata.scan_results:
            print(f"\n📈 Scan Results (Accuracy by # of Agents):")
            for sr in self.metadata.scan_results:
                bar = "█" * int(sr.accuracy * 20)
                print(f"  {sr.num_agents:2d} agents: {bar:<20} {sr.accuracy:.2%}")

        print("\n" + "=" * 80)

    def _log_to_wandb(self):
        """记录到WandB"""
        records = self.metadata.question_records
        correct = sum(1 for r in records if r.is_correct)

        self.wandb_run.log({
            "summary/accuracy": correct / len(records),
            "summary/unanimous_ratio": sum(1 for r in records if r.is_unanimous) / len(records),
            "summary/avg_agreement": sum(r.agreement_ratio for r in records) / len(records),
            "summary/avg_entropy": sum(r.entropy for r in records) / len(records),
            "summary/total_cost": self.metadata.total_cost,
            "summary/total_time": self.metadata.total_time
        })

        # 每个agent的准确率
        agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for record in records:
            for vote in record.agent_votes:
                agent_stats[vote.agent_id]['total'] += 1
                if vote.is_correct:
                    agent_stats[vote.agent_id]['correct'] += 1

        for agent_id, stats in agent_stats.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            self.wandb_run.log({f"agent/{agent_id}_accuracy": acc})

        # 扫描结果
        if self.metadata.scan_results:
            # 创建表格数据
            scan_data = [[sr.num_agents, sr.accuracy, sr.unanimous_ratio]
                         for sr in self.metadata.scan_results]

            import wandb
            table = wandb.Table(
                data=scan_data,
                columns=["num_agents", "accuracy", "unanimous_ratio"]
            )
            self.wandb_run.log({"scan_results": table})

            # 创建折线图
            for sr in self.metadata.scan_results:
                self.wandb_run.log({
                    "scan/accuracy": sr.accuracy,
                    "scan/num_agents": sr.num_agents
                })

    def save_results(self, output_dir: Path):
        """保存实验结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存JSON汇总
        json_path = output_dir / f"experiment_{self.experiment_id}.json"
        json_data = {
            "experiment_id": self.metadata.experiment_id,
            "timestamp": self.metadata.timestamp,
            "config": self.metadata.config,
            "is_homogeneous": self.metadata.is_homogeneous,
            "llm_configs": self.metadata.llm_configs,
            "total_questions": self.metadata.total_questions,
            "total_cost": self.metadata.total_cost,
            "total_time": self.metadata.total_time,
            "summary": {
                "accuracy": sum(1 for r in self.metadata.question_records if r.is_correct) / len(
                    self.metadata.question_records),
                "unanimous_ratio": sum(1 for r in self.metadata.question_records if r.is_unanimous) / len(
                    self.metadata.question_records),
            },
            "scan_results": [asdict(sr) for sr in self.metadata.scan_results] if self.metadata.scan_results else None
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON summary saved to: {json_path}")

        # 2. 保存完整元数据（pickle）
        pickle_path = output_dir / f"metadata_{self.experiment_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"✓ Full metadata saved to: {pickle_path}")

        return json_path, pickle_path


# ============================================================================
# 配置工具函数
# ============================================================================

def create_homogeneous_config(llm_name: str, num_agents: int, weights: Optional[List[float]] = None) -> List[
    Tuple[str, float]]:
    """创建同构LLM配置"""
    if weights is None:
        weights = [1.0] * num_agents

    if len(weights) != num_agents:
        raise ValueError(f"Weights length ({len(weights)}) must match num_agents ({num_agents})")

    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    return [(llm_name, w) for w in normalized_weights]


def create_heterogeneous_config(llm_names: List[str], weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
    """创建异构LLM配置"""
    num_agents = len(llm_names)

    if weights is None:
        weights = [1.0] * num_agents

    if len(weights) != num_agents:
        raise ValueError(f"Weights length ({len(weights)}) must match number of LLMs ({num_agents})")

    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    return list(zip(llm_names, normalized_weights))


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-LLM Weighted Voting System V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 同构LLM (5个相同模型)
  python run_multi_llm_voting_v2.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5
  
  # 同构LLM + 扫描模式
  python run_multi_llm_voting_v2.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 10 --scan_mode
  
  # 异构LLM
  python run_multi_llm_voting_v2.py --heterogeneous --llm_names "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B"
  
  # 异构LLM + 自定义权重
  python run_multi_llm_voting_v2.py --heterogeneous --llm_names "Qwen/Qwen3-0.6B" "Qwen/Qwen3-4B" --weights 0.3 0.7
        """
    )

    # 配置模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--homogeneous', action='store_true', help='同构LLM模式')
    mode_group.add_argument('--heterogeneous', action='store_true', help='异构LLM模式')

    # 同构模式参数
    parser.add_argument('--llm_name', type=str, help='同构模式下的LLM名称')
    parser.add_argument('--num_agents', type=int, default=3, help='智能体数量')
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--disable_thinking", action='store_true')

    # 异构模式参数
    parser.add_argument('--llm_names', nargs='+', type=str, help='异构模式下的LLM名称列表')

    # 权重
    parser.add_argument('--weights', nargs='+', type=float, default=None, help='自定义权重')

    # 扫描模式
    parser.add_argument('--scan_mode', action='store_true', help='启用扫描模式')

    # 实验参数
    parser.add_argument('--limit', type=int, default=153, help='限制问题数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--debug_first_n', type=int, default=0, help='前N个问题启用调试')

    # 输出
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--no_wandb', action='store_true', help='禁用WandB')
    parser.add_argument('--weave_project', type=str, default='vito_chan/Multi-LLM-Voting-V2', help='Weave项目名')

    return parser.parse_args()


async def main():
    args = parse_args()
    # args.llm_names = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"] * 50
    # args.weights = [61.4, 76.5] * 50

    # 验证参数
    if args.homogeneous and not args.llm_name:
        raise ValueError("同构模式必须指定 --llm_name")
    if args.heterogeneous and not args.llm_names:
        raise ValueError("异构模式必须指定 --llm_names")

    # 创建配置
    if args.homogeneous:
        llm_configs = create_homogeneous_config(args.llm_name, args.num_agents, args.weights)
        is_homogeneous = True
    else:
        llm_configs = create_heterogeneous_config(args.llm_names, args.weights)
        is_homogeneous = False

    print("\n" + "=" * 80)
    print("MULTI-LLM VOTING SYSTEM V2")
    print("=" * 80)
    print(f"Mode: {'Homogeneous' if is_homogeneous else 'Heterogeneous'}")
    print(f"LLM Configs:")
    for llm_name, weight in llm_configs:
        print(f"  - {llm_name}: weight={weight:.3f}")
    print(f"Scan Mode: {args.scan_mode}")
    print("=" * 80 + "\n")

    # 初始化追踪
    weave.init(project_name=args.weave_project)

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project="Multi-LLM-Voting-V2",
            config=vars(args),
            name=f"{'homo' if is_homogeneous else 'hetero'}_{len(llm_configs)}agents_{time.strftime('%H%M%S')}"
        )

    # 加载数据集
    download()
    dataset = MMLUDataset('val')

    # 创建实验运行器
    runner = ExperimentRunner(
        llm_configs=llm_configs,
        dataset=dataset,
        is_homogeneous=is_homogeneous,
        scan_mode=args.scan_mode,
        wandb_run=wandb_run,
        temperature=args.temperature,
        enable_thinking=not args.disable_thinking,
    )

    # 运行实验
    metadata = await runner.run(
        limit_questions=args.limit,
        batch_size=args.batch_size,
        debug_first_n=args.debug_first_n
    )

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else GDesigner_ROOT / "result" / "multi_llm_voting_v2"
    runner.save_results(output_dir)

    print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY\n")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())

"""
多LLM辩论多智能体系统
支持同构/异构LLM配置、多轮辩论、扫描模式、详细数据记录

Features:
- 同构LLM: 指定单个LLM名字和数量
- 异构LLM: 指定LLM名字列表
- 多轮辩论: 每轮agent可以看到其他agent的回答并更新自己的答案
- 当debate_rounds=1时，行为与majority voting一致
- 扫描模式: 自动测试1到N个LLM的投票结果
- 详细数据记录: 保存每道题的投票分布、置信度等元数据
- WandB集成: 实时记录和可视化

Usage:
    # 同构LLM (5个相同模型), 3轮辩论
    python run_multi_llm_debate.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5 --debate_rounds 3

    # 异构LLM + 辩论
    python run_multi_llm_debate.py --heterogeneous --llm_names "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" --debate_rounds 2

    # 单轮 (等同于majority voting)
    python run_multi_llm_debate.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5 --debate_rounds 1
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
class RoundVote:
    """单轮中单个智能体的投票记录"""
    round_num: int
    agent_id: str
    llm_name: str
    weight: float
    raw_response: str
    extracted_answer: str
    response_time: float
    is_correct: bool = False


@dataclass
class AgentDebateHistory:
    """单个智能体在所有轮次的辩论历史"""
    agent_id: str
    llm_name: str
    weight: float
    round_votes: List[RoundVote] = field(default_factory=list)

    @property
    def final_answer(self) -> str:
        if self.round_votes:
            return self.round_votes[-1].extracted_answer
        return "INVALID"

    @property
    def answer_changed(self) -> bool:
        """答案是否在辩论过程中改变"""
        if len(self.round_votes) < 2:
            return False
        first_answer = self.round_votes[0].extracted_answer
        return any(v.extracted_answer != first_answer for v in self.round_votes[1:])


@dataclass
class QuestionRecord:
    """单道题目的完整记录"""
    question_id: int
    question_text: str
    correct_answer: str

    # 辩论历史
    agent_histories: List[AgentDebateHistory] = field(default_factory=list)
    num_debate_rounds: int = 1

    # 最终投票统计
    final_answer: str = ""
    is_correct: bool = False
    vote_distribution: Dict[str, float] = field(default_factory=dict)
    raw_vote_counts: Dict[str, int] = field(default_factory=dict)

    # 每轮的投票分布
    round_vote_distributions: List[Dict[str, float]] = field(default_factory=list)
    round_accuracies: List[float] = field(default_factory=list)

    # 一致性指标
    is_unanimous: bool = False
    agreement_ratio: float = 0.0
    entropy: float = 0.0

    # 辩论动态指标
    answer_change_count: int = 0  # 有多少agent改变了答案
    convergence_round: int = -1   # 在哪一轮达成一致（-1表示未达成）

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
    debate_rounds: int = 1


# ============================================================================
# 核心类
# ============================================================================

class DebateAgent:
    """单个辩论智能体"""

    def __init__(self, agent_id: str, llm_name: str, weight: float = 1.0,
                 temperature: float = 0.7, enable_thinking: bool = True):
        self.agent_id = agent_id
        self.llm_name = llm_name
        self.weight = weight
        self.llm = LLMRegistry.get(llm_name)
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def _get_initial_prompt(self) -> str:
        """获取第一轮的系统提示"""
        return """You are an expert at multiple-choice questions.
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

    def _get_debate_prompt(self) -> str:
        """获取辩论轮的系统提示"""
        return """You are an expert at multiple-choice questions participating in a debate.
You will be given a question with 4 options (A, B, C, D), along with other experts' answers and reasoning.
Only one answer is correct.

Consider other experts' perspectives carefully:
- If their reasoning is convincing, you may change your answer
- If you believe your original answer is correct, defend it with stronger reasoning
- Focus on the logical validity of arguments, not just majority opinion

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

    async def initial_vote(self, question: str) -> Tuple[str, float]:
        """第一轮投票（无其他agent信息）"""
        system_prompt = self._get_initial_prompt()
        user_prompt = f"{question}\n\nRemember: End your response with **Answer: X** where X is your chosen letter."

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        start_time = time.time()
        response = await self.llm.acomp(messages, temperature=self.temperature,
                                         enable_thinking=self.enable_thinking)
        elapsed_time = time.time() - start_time

        return response, elapsed_time

    async def debate_vote(self, question: str, other_responses: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        辩论轮投票（可以看到其他agent的回答）

        Args:
            question: 原始问题
            other_responses: 其他agent的回答列表，每个包含 {'agent_id': str, 'answer': str, 'reasoning': str}
        """
        system_prompt = self._get_debate_prompt()

        # 构建其他专家的回答信息
        others_info = "\n\n--- Other Experts' Responses ---\n"
        for resp in other_responses:
            others_info += f"\n**{resp['agent_id']}** chose **{resp['answer']}**"
            if resp.get('reasoning'):
                # 截取推理部分（避免太长）
                reasoning = resp['reasoning'][:500] + "..." if len(resp['reasoning']) > 500 else resp['reasoning']
                others_info += f":\n{reasoning}\n"
            else:
                others_info += "\n"
        others_info += "\n--- End of Other Experts' Responses ---\n"

        user_prompt = f"""{question}

{others_info}

Now, considering the above perspectives, provide your answer.
Remember: End your response with **Answer: X** where X is your chosen letter."""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        start_time = time.time()
        response = await self.llm.acomp(messages, temperature=self.temperature,
                                         enable_thinking=self.enable_thinking)
        elapsed_time = time.time() - start_time

        return response, elapsed_time


class MultiLLMDebateSystem:
    """多LLM辩论系统"""

    def __init__(self, llm_configs: List[Tuple[str, float]], debate_rounds: int = 1,
                 temperature: float = 0.7, enable_thinking: bool = True):
        """
        Args:
            llm_configs: List of (llm_name, weight) tuples
            debate_rounds: 辩论轮数，1表示无辩论（等同于majority voting）
        """
        self.agents: List[DebateAgent] = []
        self.debate_rounds = debate_rounds

        for idx, (llm_name, weight) in enumerate(llm_configs):
            agent_id = f"agent_{idx}_{llm_name.split('/')[-1]}"
            agent = DebateAgent(agent_id, llm_name, weight,
                               temperature=temperature, enable_thinking=enable_thinking)
            self.agents.append(agent)

        # 归一化权重
        total_weight = sum(agent.weight for agent in self.agents)
        if total_weight > 0:
            for agent in self.agents:
                agent.weight /= total_weight

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

    def _extract_reasoning(self, response: str) -> str:
        """从回复中提取推理部分"""
        # 尝试提取 <think>...</think> 中的内容
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 否则返回 **Answer: 之前的内容
        answer_match = re.search(r'\*\*Answer:', response, re.IGNORECASE)
        if answer_match:
            return response[:answer_match.start()].strip()

        return response[:300]  # 返回前300字符

    async def vote_on_question(
            self,
            question_id: int,
            question: str,
            correct_answer: str,
            active_agent_ids: Optional[List[str]] = None
    ) -> QuestionRecord:
        """
        对单个问题进行辩论投票
        """
        record = QuestionRecord(
            question_id=question_id,
            question_text=question[:500],
            correct_answer=correct_answer,
            num_debate_rounds=self.debate_rounds
        )

        start_time = time.time()

        # 确定参与的agents
        active_agents = self.agents
        if active_agent_ids:
            active_agents = [a for a in self.agents if a.agent_id in active_agent_ids]

        # 初始化每个agent的辩论历史
        agent_histories: Dict[str, AgentDebateHistory] = {}
        for agent in active_agents:
            agent_histories[agent.agent_id] = AgentDebateHistory(
                agent_id=agent.agent_id,
                llm_name=agent.llm_name,
                weight=agent.weight
            )

        # 存储每轮的回答（用于下一轮辩论）
        current_round_responses: Dict[str, Dict[str, str]] = {}

        # 进行多轮辩论
        for round_num in range(1, self.debate_rounds + 1):
            if round_num == 1:
                # 第一轮：独立投票
                tasks = [agent.initial_vote(question) for agent in active_agents]
            else:
                # 后续轮：辩论投票
                tasks = []
                for agent in active_agents:
                    # 收集其他agent的回答
                    other_responses = [
                        current_round_responses[other_id]
                        for other_id in current_round_responses
                        if other_id != agent.agent_id
                    ]
                    tasks.append(agent.debate_vote(question, other_responses))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理本轮结果
            round_vote_scores = defaultdict(float)
            round_vote_counts = defaultdict(int)
            current_round_responses = {}

            for agent, result in zip(active_agents, results):
                if isinstance(result, Exception):
                    vote = RoundVote(
                        round_num=round_num,
                        agent_id=agent.agent_id,
                        llm_name=agent.llm_name,
                        weight=agent.weight,
                        raw_response=f"ERROR: {str(result)}",
                        extracted_answer="ERROR",
                        response_time=0.0,
                        is_correct=False
                    )
                    current_round_responses[agent.agent_id] = {
                        'agent_id': agent.agent_id,
                        'answer': 'ERROR',
                        'reasoning': ''
                    }
                else:
                    response, resp_time = result
                    extracted = self._extract_answer(response)
                    reasoning = self._extract_reasoning(response)

                    vote = RoundVote(
                        round_num=round_num,
                        agent_id=agent.agent_id,
                        llm_name=agent.llm_name,
                        weight=agent.weight,
                        raw_response=response,
                        extracted_answer=extracted,
                        response_time=resp_time,
                        is_correct=(extracted == correct_answer)
                    )

                    current_round_responses[agent.agent_id] = {
                        'agent_id': agent.agent_id,
                        'answer': extracted,
                        'reasoning': reasoning
                    }

                    if extracted not in ["INVALID", "ERROR"]:
                        round_vote_scores[extracted] += agent.weight
                        round_vote_counts[extracted] += 1

                agent_histories[agent.agent_id].round_votes.append(vote)

            # 记录本轮的投票分布
            record.round_vote_distributions.append(dict(round_vote_scores))

            # 计算本轮准确率
            round_correct = sum(1 for agent in active_agents
                               if agent_histories[agent.agent_id].round_votes[-1].is_correct)
            record.round_accuracies.append(round_correct / len(active_agents))

            # 检查是否达成一致
            if len(round_vote_counts) == 1 and record.convergence_round == -1:
                record.convergence_round = round_num

        # 保存agent历史
        record.agent_histories = list(agent_histories.values())

        # 计算最终结果（基于最后一轮）
        final_vote_scores = defaultdict(float)
        final_vote_counts = defaultdict(int)

        for history in record.agent_histories:
            if history.round_votes:
                final_vote = history.round_votes[-1]
                if final_vote.extracted_answer not in ["INVALID", "ERROR"]:
                    final_vote_scores[final_vote.extracted_answer] += history.weight
                    final_vote_counts[final_vote.extracted_answer] += 1

        record.vote_distribution = dict(final_vote_scores)
        record.raw_vote_counts = dict(final_vote_counts)

        if final_vote_scores:
            record.final_answer = max(final_vote_scores.items(), key=lambda x: x[1])[0]
        else:
            record.final_answer = "INVALID"

        record.is_correct = (record.final_answer == correct_answer)

        # 一致性指标
        total_valid_votes = sum(final_vote_counts.values())
        if total_valid_votes > 0:
            max_votes = max(final_vote_counts.values())
            record.is_unanimous = (len(final_vote_counts) == 1)
            record.agreement_ratio = max_votes / total_valid_votes

            # 计算熵
            probs = [c / total_valid_votes for c in final_vote_counts.values()]
            record.entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)

        # 辩论动态指标
        record.answer_change_count = sum(1 for h in record.agent_histories if h.answer_changed)

        record.total_time = time.time() - start_time

        return record


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
            debate_rounds: int = 1,
            scan_mode: bool = False,
            wandb_run=None,
            temperature: float = 0.7,
            enable_thinking: bool = True
    ):
        self.llm_configs = llm_configs
        self.dataset = dataset
        self.is_homogeneous = is_homogeneous
        self.debate_rounds = debate_rounds
        self.scan_mode = scan_mode
        self.wandb_run = wandb_run

        # 初始化辩论系统
        self.debate_system = MultiLLMDebateSystem(
            llm_configs,
            debate_rounds=debate_rounds,
            temperature=temperature,
            enable_thinking=enable_thinking
        )

        # 实验元数据
        self.experiment_id = time.strftime("%Y%m%d_%H%M%S")
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={},
            llm_configs=llm_configs,
            is_homogeneous=is_homogeneous,
            debate_rounds=debate_rounds,
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
        """运行实验"""
        total_questions = min(len(self.dataset), limit_questions) if limit_questions else len(self.dataset)
        self.metadata.total_questions = total_questions

        print(f"\n{'=' * 80}")
        print(f"RUNNING DEBATE EXPERIMENT: {self.experiment_id}")
        print(f"{'=' * 80}")
        print(f"Total Agents: {len(self.debate_system.agents)}")
        print(f"Debate Rounds: {self.debate_rounds}")
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

                task = self.debate_system.vote_on_question(
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
                        log_data = {
                            "question/is_correct": int(result.is_correct),
                            "question/agreement_ratio": result.agreement_ratio,
                            "question/entropy": result.entropy,
                            "question/is_unanimous": int(result.is_unanimous),
                            "question/time": result.total_time,
                            "question/answer_changes": result.answer_change_count,
                        }
                        if result.convergence_round > 0:
                            log_data["question/convergence_round"] = result.convergence_round
                        self.wandb_run.log(log_data)

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

        # 扫描模式
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

        num_agents = len(self.debate_system.agents)

        for n in range(1, num_agents + 1):
            agent_ids = [agent.agent_id for agent in self.debate_system.agents[:n]]

            correct_count = 0
            unanimous_count = 0
            total_agreement = 0.0
            total_time = 0.0

            for record in self.metadata.question_records:
                # 筛选前n个agent的历史
                subset_histories = [h for h in record.agent_histories if h.agent_id in agent_ids]

                # 重新计算加权得分（基于最后一轮）
                vote_scores = defaultdict(float)
                vote_counts = defaultdict(int)
                total_weight = sum(h.weight for h in subset_histories)

                for history in subset_histories:
                    if history.round_votes:
                        final_answer = history.round_votes[-1].extracted_answer
                        if final_answer not in ["INVALID", "ERROR"]:
                            normalized_weight = history.weight / total_weight if total_weight > 0 else 0
                            vote_scores[final_answer] += normalized_weight
                            vote_counts[final_answer] += 1

                if vote_scores:
                    final_answer = max(vote_scores.items(), key=lambda x: x[1])[0]
                    is_correct = (final_answer == record.correct_answer)
                else:
                    is_correct = False

                correct_count += int(is_correct)

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

        # 辩论动态
        if self.debate_rounds > 1:
            print(f"\n🔄 Debate Dynamics:")
            avg_changes = sum(r.answer_change_count for r in records) / len(records)
            print(f"  Avg Answer Changes per Question: {avg_changes:.2f}")

            converged = [r for r in records if r.convergence_round > 0]
            if converged:
                avg_conv_round = sum(r.convergence_round for r in converged) / len(converged)
                print(f"  Questions Reaching Consensus: {len(converged)} ({len(converged)/len(records):.1%})")
                print(f"  Avg Convergence Round: {avg_conv_round:.2f}")

            # 每轮准确率变化
            print(f"\n📈 Accuracy by Round:")
            for round_num in range(self.debate_rounds):
                round_accs = [r.round_accuracies[round_num] for r in records if len(r.round_accuracies) > round_num]
                if round_accs:
                    avg_acc = sum(round_accs) / len(round_accs)
                    print(f"  Round {round_num + 1}: {avg_acc:.2%}")

        print(f"\n🤖 Per-Agent Performance (Final Round):")
        agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'changed': 0})
        for record in records:
            for history in record.agent_histories:
                agent_stats[history.agent_id]['total'] += 1
                if history.round_votes and history.round_votes[-1].is_correct:
                    agent_stats[history.agent_id]['correct'] += 1
                if history.answer_changed:
                    agent_stats[history.agent_id]['changed'] += 1

        for agent_id in sorted(agent_stats.keys()):
            stats = agent_stats[agent_id]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            change_rate = stats['changed'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {agent_id}: {acc:.2%} ({stats['correct']}/{stats['total']}), "
                  f"Changed: {change_rate:.1%}")

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

        log_data = {
            "summary/accuracy": correct / len(records),
            "summary/unanimous_ratio": sum(1 for r in records if r.is_unanimous) / len(records),
            "summary/avg_agreement": sum(r.agreement_ratio for r in records) / len(records),
            "summary/avg_entropy": sum(r.entropy for r in records) / len(records),
            "summary/total_cost": self.metadata.total_cost,
            "summary/total_time": self.metadata.total_time,
            "summary/debate_rounds": self.debate_rounds,
        }

        if self.debate_rounds > 1:
            log_data["summary/avg_answer_changes"] = sum(r.answer_change_count for r in records) / len(records)

        self.wandb_run.log(log_data)

        # 每个agent的准确率
        agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for record in records:
            for history in record.agent_histories:
                agent_stats[history.agent_id]['total'] += 1
                if history.round_votes and history.round_votes[-1].is_correct:
                    agent_stats[history.agent_id]['correct'] += 1

        for agent_id, stats in agent_stats.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            self.wandb_run.log({f"agent/{agent_id}_accuracy": acc})

        # 扫描结果
        if self.metadata.scan_results:
            import wandb
            scan_data = [[sr.num_agents, sr.accuracy, sr.unanimous_ratio]
                         for sr in self.metadata.scan_results]
            table = wandb.Table(
                data=scan_data,
                columns=["num_agents", "accuracy", "unanimous_ratio"]
            )
            self.wandb_run.log({"scan_results": table})

            for sr in self.metadata.scan_results:
                self.wandb_run.log({
                    "scan/accuracy": sr.accuracy,
                    "scan/num_agents": sr.num_agents
                })

    def save_results(self, output_dir: Path):
        """保存实验结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存JSON汇总
        json_path = output_dir / f"debate_experiment_{self.experiment_id}.json"
        json_data = {
            "experiment_id": self.metadata.experiment_id,
            "timestamp": self.metadata.timestamp,
            "config": self.metadata.config,
            "is_homogeneous": self.metadata.is_homogeneous,
            "debate_rounds": self.metadata.debate_rounds,
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
        pickle_path = output_dir / f"debate_metadata_{self.experiment_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"✓ Full metadata saved to: {pickle_path}")

        return json_path, pickle_path


# ============================================================================
# 配置工具函数
# ============================================================================

def create_homogeneous_config(llm_name: str, num_agents: int, weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
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
        description="Multi-LLM Debate System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 同构LLM (5个相同模型), 3轮辩论
  python run_multi_llm_debate.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5 --debate_rounds 3
  
  # 单轮（等同于majority voting）
  python run_multi_llm_debate.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5 --debate_rounds 1
  
  # 异构LLM + 辩论
  python run_multi_llm_debate.py --heterogeneous --llm_names "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" --debate_rounds 2
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

    # 辩论参数
    parser.add_argument('--debate_rounds', type=int, default=1, help='辩论轮数（1=无辩论，等同于majority voting）')

    # 扫描模式
    parser.add_argument('--scan_mode', action='store_true', help='启用扫描模式')

    # 实验参数
    parser.add_argument('--limit', type=int, default=153, help='限制问题数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--debug_first_n', type=int, default=0, help='前N个问题启用调试')

    # 输出
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--no_wandb', action='store_true', help='禁用WandB')
    parser.add_argument('--weave_project', type=str, default='vito_chan/Multi-LLM-Debate', help='Weave项目名')

    return parser.parse_args()


async def main():
    args = parse_args()

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
    print("MULTI-LLM DEBATE SYSTEM")
    print("=" * 80)
    print(f"Mode: {'Homogeneous' if is_homogeneous else 'Heterogeneous'}")
    print(f"Debate Rounds: {args.debate_rounds} {'(equivalent to majority voting)' if args.debate_rounds == 1 else ''}")
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
            project="Multi-LLM-Debate",
            config=vars(args),
            name=f"{'homo' if is_homogeneous else 'hetero'}_{len(llm_configs)}agents_r{args.debate_rounds}_{time.strftime('%H%M%S')}"
        )

    # 加载数据集
    download()
    dataset = MMLUDataset('val')

    # 创建实验运行器
    runner = ExperimentRunner(
        llm_configs=llm_configs,
        dataset=dataset,
        is_homogeneous=is_homogeneous,
        debate_rounds=args.debate_rounds,
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
    output_dir = Path(args.output_dir) if args.output_dir else GDesigner_ROOT / "result" / "multi_llm_debate"
    runner.save_results(output_dir)

    print("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY\n")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    asyncio.run(main())
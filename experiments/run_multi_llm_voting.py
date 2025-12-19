"""
多LLM加权投票多智能体系统
支持不同规模的Qwen模型进行协作推理和投票决策

Usage:
    python experiments/run_multi_llm_voting.py --num_agents 3 --limit 100
    python experiments/run_multi_llm_voting.py --num_agents 6 --weights 0.1 0.15 0.2 0.1 0.15 0.3
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import Counter
import math

import weave

from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


class WeightedVotingAgent:
    """单个投票智能体，基于特定LLM模型"""

    def __init__(self, agent_id: str, llm_name: str, weight: float, domain: str = "mmlu", debug: bool = False):
        self.agent_id = agent_id
        self.llm_name = llm_name
        self.weight = weight
        self.domain = domain
        self.debug = debug  # 调试模式
        self.llm = LLMRegistry.get(llm_name)

        # 导入prompt
        from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
        self.prompt_set = PromptSetRegistry.get(domain)

        print(f"  ✓ Agent [{agent_id}] initialized: {llm_name} (weight={weight:.2f})")

    async def vote(self, question: str) -> str:
        """对问题进行投票（生成答案）"""

        # 优化的prompt：明确要求在思考后输出格式化答案
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
Let me analyze this question...
Option A seems wrong because...
Option B is correct because...
</think>

**Answer: B**"""

        user_prompt = f"{question}\n\nRemember: End your response with **Answer: X** where X is your chosen letter."

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # 调用LLM生成答案
        response = await self.llm.agen(messages, temperature=0.7)

        return response


class MultiLLMVotingSystem:
    """多LLM加权投票系统"""

    def __init__(self,
                 llm_configs: List[Tuple[str, float]],
                 domain: str = "mmlu"):
        """
        Args:
            llm_configs: List of (llm_name, weight) tuples
            domain: 数据集领域
        """
        self.domain = domain
        self.agents = []

        print("\n" + "=" * 80)
        print("INITIALIZING MULTI-LLM VOTING SYSTEM")
        print("=" * 80)

        # 初始化所有智能体
        for idx, (llm_name, weight) in enumerate(llm_configs):
            agent_id = f"agent_{idx}_{llm_name.split('/')[-1]}"
            agent = WeightedVotingAgent(
                agent_id=agent_id,
                llm_name=llm_name,
                weight=weight,
                domain=domain
            )
            self.agents.append(agent)

        # 验证权重总和
        total_weight = sum(agent.weight for agent in self.agents)
        print(f"\n  Total weight: {total_weight:.2f}")

        if abs(total_weight - 1.0) > 0.01:
            print(f"  ⚠️  Warning: Weights don't sum to 1.0, normalizing...")
            for agent in self.agents:
                agent.weight /= total_weight

        print("=" * 80 + "\n")

    async def vote_on_question(self, question: str, debug: bool = False) -> Tuple[str, Dict]:
        """
        对单个问题进行投票

        Args:
            question: 问题文本
            debug: 是否输出调试信息

        Returns:
            (final_answer, voting_details)
        """
        # 并发收集所有智能体的投票
        tasks = [agent.vote(question) for agent in self.agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常和提取答案
        votes = []
        debug_info = []

        for agent, response in zip(self.agents, responses):
            if isinstance(response, Exception):
                print(f"  ⚠️  {agent.agent_id} failed: {response}")
                votes.append(("ERROR", agent.weight))
                debug_info.append({
                    'agent': agent.agent_id,
                    'status': 'error',
                    'error': str(response)
                })
            else:
                # 提取答案
                answer = self._extract_answer(response)
                votes.append((answer, agent.weight))

                # 收集调试信息
                if debug:
                    debug_info.append({
                        'agent': agent.agent_id,
                        'raw_response': response[:300] + '...' if len(response) > 300 else response,
                        'extracted_answer': answer,
                        'weight': agent.weight
                    })

        # 如果启用调试，打印提取过程
        if debug and debug_info:
            print("\n" + "=" * 60)
            print("DEBUG: Answer Extraction Process")
            print("=" * 60)
            for info in debug_info:
                if info.get('status') != 'error':
                    print(f"\n{info['agent']}:")
                    print(f"  Raw Response: {info['raw_response']}")
                    print(f"  Extracted: {info['extracted_answer']}")
                    print(f"  Weight: {info['weight']}")
            print("=" * 60 + "\n")

        # 加权投票
        final_answer, voting_details = self._weighted_vote(votes)

        if debug_info:
            voting_details['debug_info'] = debug_info

        return final_answer, voting_details

    def _extract_answer(self, response: str) -> str:
        """
        从回复中提取答案字母（鲁棒版本）

        策略优先级：
        1. **Answer: X** 格式（最可靠）
        2. 最后一个出现的独立字母（A/B/C/D）
        3. <think>标签后的第一个字母
        4. 整个文本中第一个出现的字母
        """
        import re

        # 策略1：查找 **Answer: X** 格式（最优先）
        answer_pattern = r'\*\*Answer:\s*([A-D])\*\*'
        match = re.search(answer_pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 策略2：查找 Answer: X 格式（无星号）
        answer_pattern_simple = r'(?:Answer|答案):\s*([A-D])'
        match = re.search(answer_pattern_simple, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 策略3：提取 </think> 标签之后的内容
        think_split = response.split('</think>')
        if len(think_split) > 1:
            after_think = think_split[-1]  # 取最后一个 </think> 之后的内容

            # 在 </think> 后查找独立的字母
            # 匹配模式：行首、空格、标点后的单独字母
            letter_pattern = r'(?:^|\s|[.!?\n])\s*([A-D])(?:\s|[.!?,\n]|$)'
            match = re.search(letter_pattern, after_think, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).upper()

            # 如果没找到独立字母，找第一个字母
            for char in after_think:
                if char.upper() in ['A', 'B', 'C', 'D']:
                    return char.upper()

        # 策略4：查找最后一个出现的独立字母（可能是总结时的答案）
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # 检查是否是单独的字母行
            if len(line) == 1 and line.upper() in ['A', 'B', 'C', 'D']:
                return line.upper()
            # 检查是否包含 "选X" 或 "choose X" 等模式
            choice_pattern = r'(?:选择?|choose|select|pick)\s*([A-D])'
            match = re.search(choice_pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # 策略5：查找 "X is correct" 或 "X 是正确的" 模式
        correct_pattern = r'([A-D])\s*(?:is|为|是)\s*(?:correct|right|正确)'
        matches = re.findall(correct_pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # 取最后一个匹配

        # 策略6：查找所有独立出现的字母，取最后一个
        all_letters = re.findall(r'(?:^|\s|[.!?\n])\s*([A-D])(?:\s|[.!?,\n]|$)', response,
                                 re.MULTILINE | re.IGNORECASE)
        if all_letters:
            return all_letters[-1].upper()

        # 策略7：在整个文本中查找第一个字母（最后的兜底）
        for char in response:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()

        # 如果所有策略都失败，返回INVALID
        return "INVALID"

    def _weighted_vote(self, votes: List[Tuple[str, float]]) -> Tuple[str, Dict]:
        """
        加权投票机制

        Args:
            votes: List of (answer, weight)

        Returns:
            (final_answer, details)
        """
        # 统计每个答案的加权得分
        scores = {}
        for answer, weight in votes:
            if answer not in scores:
                scores[answer] = 0.0
            scores[answer] += weight

        # 找出得分最高的答案
        if not scores:
            final_answer = "INVALID"
        else:
            final_answer = max(scores.items(), key=lambda x: x[1])[0]

        # 构建详细信息
        details = {
            'votes': votes,
            'scores': scores,
            'final_answer': final_answer
        }

        return final_answer, details


class VotingMetrics:
    """投票系统性能指标"""

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.total_time = 0.0

        # 每个智能体的准确率
        self.agent_correct = {}
        self.agent_total = {}

        # 投票统计
        self.unanimous_votes = 0  # 一致投票
        self.split_votes = 0  # 分歧投票

        # 详细记录
        self.results = []

    def update(self,
               predicted: str,
               target: str,
               voting_details: Dict,
               question: str,
               execution_time: float):
        """更新指标"""
        is_correct = (predicted == target)
        self.correct += int(is_correct)
        self.total += 1
        self.total_time += execution_time

        # 记录每个智能体的表现
        for answer, weight in voting_details['votes']:
            agent_id = f"agent_{voting_details['votes'].index((answer, weight))}"

            if agent_id not in self.agent_correct:
                self.agent_correct[agent_id] = 0
                self.agent_total[agent_id] = 0

            self.agent_total[agent_id] += 1
            if answer == target:
                self.agent_correct[agent_id] += 1

        # 投票一致性分析
        answers = [vote[0] for vote in voting_details['votes']]
        if len(set(answers)) == 1:
            self.unanimous_votes += 1
        else:
            self.split_votes += 1

        # 详细记录
        self.results.append({
            'question': question[:100] + '...',
            'predicted': predicted,
            'target': target,
            'correct': is_correct,
            'votes': voting_details['votes'],
            'scores': voting_details['scores'],
            'time': execution_time
        })

    def get_accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def get_agent_accuracy(self, agent_id: str) -> float:
        if agent_id not in self.agent_total or self.agent_total[agent_id] == 0:
            return 0.0
        return self.agent_correct[agent_id] / self.agent_total[agent_id]

    def print_summary(self):
        print("\n" + "=" * 80)
        print("VOTING SYSTEM PERFORMANCE SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overall Performance:")
        print(f"  Accuracy: {self.get_accuracy():.2%} ({self.correct}/{self.total})")
        print(f"  Avg Time: {self.total_time / self.total:.2f}s per question")

        print(f"\n🤖 Individual Agent Performance:")
        for agent_id in sorted(self.agent_correct.keys()):
            acc = self.get_agent_accuracy(agent_id)
            correct = self.agent_correct[agent_id]
            total = self.agent_total[agent_id]
            print(f"  {agent_id}: {acc:.2%} ({correct}/{total})")

        print(f"\n🗳️  Voting Statistics:")
        print(f"  Unanimous Votes: {self.unanimous_votes} ({self.unanimous_votes / self.total:.1%})")
        print(f"  Split Votes: {self.split_votes} ({self.split_votes / self.total:.1%})")

        print(f"\n💰 Cost Estimate:")
        print(f"  Total Cost: ${Cost.instance().value:.4f}")
        print(f"  Prompt Tokens: {PromptTokens.instance().value / 1000:.1f}k")
        print(f"  Completion Tokens: {CompletionTokens.instance().value / 1000:.1f}k")

        print("\n" + "=" * 80)

    def save_results(self, output_path: Path):
        """保存详细结果"""
        results_dict = {
            'summary': {
                'accuracy': self.get_accuracy(),
                'correct': self.correct,
                'total': self.total,
                'avg_time': self.total_time / self.total if self.total > 0 else 0,
                'unanimous_votes': self.unanimous_votes,
                'split_votes': self.split_votes,
                'total_cost': Cost.instance().value
            },
            'agent_performance': {
                agent_id: {
                    'accuracy': self.get_agent_accuracy(agent_id),
                    'correct': self.agent_correct[agent_id],
                    'total': self.agent_total[agent_id]
                }
                for agent_id in self.agent_correct.keys()
            },
            'results': self.results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")


async def run_voting_experiment(
        num_agents: int,
        weights: List[float] = None,
        limit_questions: int = None,
        batch_size: int = 4,
        save_results: bool = True,
        debug_first_n: int = 3,  # 前N个问题启用调试
        **kwargs
):
    """
    运行多LLM投票实验

    Args:
        num_agents: 智能体数量（3或6）
        weights: 自定义权重列表
        limit_questions: 限制问题数量
        batch_size: 批处理大小
        save_results: 是否保存结果
        debug_first_n: 前N个问题启用调试模式
    """

    # 配置LLM和权重
    llm_configs = get_llm_configs(num_agents, weights)

    # 初始化投票系统
    voting_system = MultiLLMVotingSystem(llm_configs, domain="mmlu")

    # 加载数据集
    print("Loading MMLU validation dataset...")
    download()
    dataset = MMLUDataset('val')

    total_questions = min(len(dataset), limit_questions) if limit_questions else len(dataset)
    print(f"Testing on {total_questions} questions")
    if debug_first_n > 0:
        print(f"Debug mode enabled for first {debug_first_n} questions\n")
    else:
        print()

    # 初始化指标
    metrics = VotingMetrics()

    # 重置计数器
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    # 批处理执行
    num_batches = math.ceil(total_questions / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_questions)

        batch_tasks = []
        batch_records = []

        for idx in range(start_idx, end_idx):
            record = dataset[idx]
            input_dict = dataset.record_to_input(record)
            question = input_dict['task']

            # 前N个问题启用调试
            enable_debug = (idx < debug_first_n)

            batch_tasks.append(voting_system.vote_on_question(question, debug=enable_debug))
            batch_records.append(record)

        # 并发执行批次
        batch_start = time.time()
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        batch_time = time.time() - batch_start

        # 处理结果
        for record, result in zip(batch_records, batch_results):
            if isinstance(result, Exception):
                print(f"\n❌ Error: {result}")
                continue

            final_answer, voting_details = result
            target = dataset.record_to_target_answer(record)
            question = dataset.record_to_input(record)['task']

            # 更新指标
            metrics.update(
                predicted=final_answer,
                target=target,
                voting_details=voting_details,
                question=question,
                execution_time=batch_time / len(batch_records)
            )

        # 每5个批次打印进度
        if (batch_idx + 1) % 5 == 0:
            print(f"\n--- Progress: {end_idx}/{total_questions} ---")
            print(f"  Current Accuracy: {metrics.get_accuracy():.2%}")
            print(f"  Avg Time: {metrics.total_time / metrics.total:.2f}s")

    # 打印最终结果
    metrics.print_summary()

    # 保存结果
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = GDesigner_ROOT / "result" / "multi_llm_voting"
        result_dir.mkdir(parents=True, exist_ok=True)

        output_file = result_dir / f"voting_{num_agents}agents_{timestamp}.json"
        metrics.save_results(output_file)

    # WandB记录
    if "wandb_run" in kwargs:
        kwargs["wandb_run"].log({
            "accuracy": metrics.get_accuracy(),
            "unanimous_votes_ratio": metrics.unanimous_votes / metrics.total,
            "avg_time": metrics.total_time / metrics.total,
            "total_cost": Cost.instance().value
        })

        # 记录每个agent的准确率
        for agent_id in metrics.agent_correct.keys():
            kwargs["wandb_run"].log({
                f"agent_accuracy/{agent_id}": metrics.get_agent_accuracy(agent_id)
            })

    return metrics


def get_llm_configs(num_agents: int, weights: List[float] = None) -> List[Tuple[str, float]]:
    """
    获取LLM配置和权重

    Args:
        num_agents: 智能体数量（3或6）
        weights: 自定义权重列表

    Returns:
        List of (llm_name, weight)
    """
    # 可用的LLM模型（按规模从小到大）
    available_models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B"
    ]

    # 根据智能体数量选择模型
    if num_agents == 3:
        selected_models = available_models
    elif num_agents == 6:
        # 每种模型各用两次
        selected_models = [model for model in available_models for _ in range(2)]
    else:
        raise ValueError(f"Unsupported num_agents: {num_agents}. Only 3 or 6 are supported.")

    # 设置权重
    if weights is None:
        # 默认权重：按模型规模递增
        if num_agents == 3:
            # weights = [0.22, 0.3, 0.48]  # 小模型权重低，大模型权重高
            weights = [0.5274390243902439, 0.6185567010309279, 0.7941176470588235]  # 小模型权重低，大模型权重高 1.940113372479995
        elif num_agents == 6:
            # weights = [0.11, 0.11, 0.15, 0.15, 0.24, 0.24]
            weights = [0.5274390243902439, 0.5274390243902439, 0.6185567010309279, 0.6185567010309279,
                       0.7941176470588235, 0.7941176470588235]
    else:
        if len(weights) != num_agents:
            raise ValueError(f"Length of weights ({len(weights)}) must equal num_agents ({num_agents})")

    # 标准化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return list(zip(selected_models, weights))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-LLM Weighted Voting System for MMLU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3个智能体，默认权重
  python experiments/run_multi_llm_voting.py --num_agents 3 --limit 100

  # 6个智能体，自定义权重
  python experiments/run_multi_llm_voting.py --num_agents 6 --weights 0.1 0.15 0.2 0.1 0.15 0.3

  # 完整验证集，无限制
  python experiments/run_multi_llm_voting.py --num_agents 3
        """
    )

    parser.add_argument(
        '--num_agents',
        type=int,
        default=3,
        choices=[3, 6],
        help='Number of agents (3 or 6)'
    )

    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Custom weights for each agent (must sum close to 1.0)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=153,
        help='Limit number of questions'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for parallel processing'
    )

    parser.add_argument(
        '--debug_first_n',
        type=int,
        default=3,
        help='Enable debug mode for first N questions (default: 3)'
    )

    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results'
    )

    parser.add_argument(
        '--weave_project',
        type=str,
        default='vito_chan/Multi-LLM-Voting',
        help='Weave project name'
    )

    return parser.parse_args()


async def main():
    import wandb

    args = parse_args()

    # 初始化追踪
    weave.init(project_name=args.weave_project)
    wandb_run = wandb.init(
        project="Multi-LLM-Voting",
        config=args,
        name=time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    print("\n" + "=" * 80)
    print("MULTI-LLM WEIGHTED VOTING SYSTEM - MMLU EXPERIMENT")
    print("=" * 80)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Agents: {args.num_agents}")
    print(f"Custom Weights: {args.weights if args.weights else 'Default'}")
    print("=" * 80)

    try:
        metrics = await run_voting_experiment(
            num_agents=args.num_agents,
            weights=args.weights,
            limit_questions=args.limit,
            batch_size=args.batch_size,
            save_results=not args.no_save,
            debug_first_n=args.debug_first_n,
            wandb_run=wandb_run
        )

        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

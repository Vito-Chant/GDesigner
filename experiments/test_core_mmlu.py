"""
CoRe Graph System Performance Testing on MMLU
测试 CoRe v4.3.2 在 MMLU 验证集上的性能

Usage:
    python test_core_mmlu.py --llm_name "your-model" --limit 50
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
from typing import Dict, List
from tqdm import tqdm
import math

import weave

# 导入 CoRe 组件
from GDesigner.CoRe.core_graph import CoReGraph
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT

# 导入 MMLU 数据集
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


class PerformanceMetrics:
    """性能指标收集器"""

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.total_time = 0.0
        self.total_tokens = 0
        self.total_steps = 0
        self.kv_cache_hits = 0
        self.loop_detections = 0
        self.termination_attempts = 0

        # 详细记录
        self.results = []
        self.routing_paths = []
        self.error_cases = []

    def update(self,
               predicted: str,
               target: str,
               result: 'CoReResult',
               question: str):
        """更新指标"""
        is_correct = (predicted == target)
        self.correct += int(is_correct)
        self.total += 1

        self.total_time += result.total_time
        self.total_tokens += result.total_cost_tokens
        self.total_steps += len(result.execution_trace)
        self.kv_cache_hits += result.kv_cache_hits
        self.loop_detections += result.loop_detections

        # 记录详细信息
        self.results.append({
            'question': question[:100] + '...' if len(question) > 100 else question,
            'predicted': predicted,
            'target': target,
            'correct': is_correct,
            'time': result.total_time,
            'tokens': result.total_cost_tokens,
            'steps': len(result.execution_trace),
            'routing_path': [d['selected'] for d in result.routing_decisions]
        })

        self.routing_paths.append(result.routing_decisions)

        if not is_correct:
            self.error_cases.append({
                'question': question,
                'predicted': predicted,
                'target': target,
                'routing_path': [d['selected'] for d in result.routing_decisions]
            })

    def get_accuracy(self) -> float:
        """计算准确率"""
        return self.correct / self.total if self.total > 0 else 0.0

    def get_avg_time(self) -> float:
        """计算平均时间"""
        return self.total_time / self.total if self.total > 0 else 0.0

    def get_avg_tokens(self) -> float:
        """计算平均 Token 消耗"""
        return self.total_tokens / self.total if self.total > 0 else 0.0

    def get_avg_steps(self) -> float:
        """计算平均步数"""
        return self.total_steps / self.total if self.total > 0 else 0.0

    def get_kv_cache_hit_rate(self) -> float:
        """计算 KV Cache 命中率"""
        total_routing_steps = sum(len(path) - 1 for path in self.routing_paths if len(path) > 1)
        return self.kv_cache_hits / total_routing_steps if total_routing_steps > 0 else 0.0

    def get_loop_rate(self) -> float:
        """计算循环检测率"""
        total_routing_steps = sum(len(path) for path in self.routing_paths)
        return self.loop_detections / total_routing_steps if total_routing_steps > 0 else 0.0

    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        print(f"\n📊 Accuracy Metrics:")
        print(f"  Correct:  {self.correct}/{self.total}")
        print(f"  Accuracy: {self.get_accuracy():.2%}")

        print(f"\n⏱️  Time Metrics:")
        print(f"  Total Time:    {self.total_time:.2f}s")
        print(f"  Avg Time/Task: {self.get_avg_time():.2f}s")

        print(f"\n🎯 Token Metrics:")
        print(f"  Total Tokens:     {self.total_tokens:,}")
        print(f"  Avg Tokens/Task:  {self.get_avg_tokens():.0f}")
        print(f"  Prompt Tokens:    {PromptTokens.instance().value / 1000:.1f}k")
        print(f"  Completion Tokens:{CompletionTokens.instance().value / 1000:.1f}k")

        print(f"\n🔄 Routing Metrics:")
        print(f"  Avg Steps/Task:   {self.get_avg_steps():.1f}")
        print(f"  KV Cache Hits:    {self.kv_cache_hits}")
        print(f"  KV Cache Hit Rate:{self.get_kv_cache_hit_rate():.1%}")
        print(f"  Loop Detections:  {self.loop_detections}")
        print(f"  Loop Rate:        {self.get_loop_rate():.1%}")

        print(f"\n💰 Cost Estimate:")
        print(f"  Total Cost: ${Cost.instance().value:.4f}")

        print("\n" + "=" * 80)

    def save_results(self, output_path: Path):
        """保存详细结果到文件"""
        results_dict = {
            'summary': {
                'accuracy': self.get_accuracy(),
                'correct': self.correct,
                'total': self.total,
                'avg_time': self.get_avg_time(),
                'avg_tokens': self.get_avg_tokens(),
                'avg_steps': self.get_avg_steps(),
                'kv_cache_hit_rate': self.get_kv_cache_hit_rate(),
                'loop_rate': self.get_loop_rate(),
                'total_cost': Cost.instance().value
            },
            'results': self.results,
            'error_cases': self.error_cases
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")


async def test_core_graph_on_mmlu(
        llm_name: str,
        available_roles: List[str],
        decision_method: str = "FinalRefer",
        num_rounds: int = 1,
        max_routing: int = 5,
        limit_questions: int = None,
        batch_size: int = 1,
        save_results: bool = True
):
    """
    在 MMLU 验证集上测试 CoRe Graph

    Args:
        llm_name: LLM 模型名称
        available_roles: 可用的 Agent 角色列表
        decision_method: 决策方法
        num_rounds: 每个 Agent 的轮数
        max_routing: 最大路由步数
        limit_questions: 限制测试的问题数量
        batch_size: 批处理大小
        save_results: 是否保存详细结果
    """

    # 初始化 CoRe Graph
    print("\n" + "=" * 80)
    print("INITIALIZING CoRe GRAPH SYSTEM")
    print("=" * 80)
    print(f"  LLM: {llm_name}")
    print(f"  Roles: {', '.join(available_roles)}")
    print(f"  Decision Method: {decision_method}")
    print(f"  Max Routing Steps: {max_routing}")
    print(f"  Num Rounds: {num_rounds}")
    print("=" * 80 + "\n")

    core_graph = CoReGraph(
        domain="mmlu",
        llm_name=llm_name,
        available_roles=available_roles,
        decision_method=decision_method,
        max_routing=max_routing,
        registry_save_path=None,  # 不保存 registry
        rag_top_k=3,
        max_loop_count=2
    )

    # 加载 MMLU 数据集
    print("Loading MMLU validation dataset...")
    download()
    dataset = MMLUDataset('val')

    # 限制问题数量
    if limit_questions:
        total_questions = min(len(dataset), limit_questions)
    else:
        total_questions = len(dataset)

    print(f"Testing on {total_questions} questions\n")

    # 初始化指标收集器
    metrics = PerformanceMetrics()

    # 重置全局计数器
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

            batch_tasks.append(
                core_graph.run_cognitive_relay(input_dict)
            )
            batch_records.append(record)

        # 并发执行批次
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 处理结果
        for record, result in zip(batch_records, batch_results):
            if isinstance(result, Exception):
                print(f"\n❌ Error processing question: {result}")
                # 记录为错误
                metrics.total += 1
                metrics.error_cases.append({
                    'question': dataset.record_to_input(record)['task'],
                    'error': str(result)
                })
                continue

            # 提取预测答案
            predicted = dataset.postprocess_answer(result.final_answer)
            target = dataset.record_to_target_answer(record)
            question = dataset.record_to_input(record)['task']

            # 更新指标
            metrics.update(predicted, target, result, question)

        # 每个批次后打印进度
        if (batch_idx + 1) % 5 == 0:
            print(f"\n--- Progress: {end_idx}/{total_questions} ---")
            print(f"  Current Accuracy: {metrics.get_accuracy():.2%}")
            print(f"  Avg Time: {metrics.get_avg_time():.2f}s")
            print(f"  Avg Tokens: {metrics.get_avg_tokens():.0f}")

    # 打印最终结果
    metrics.print_summary()

    # 保存详细结果
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = GDesigner_ROOT / "result" / "core_mmlu"
        result_dir.mkdir(parents=True, exist_ok=True)

        output_file = result_dir / f"core_test_{llm_name.replace('/', '_')}_{timestamp}.json"
        metrics.save_results(output_file)

    return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Test CoRe Graph System on MMLU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 测试 50 个问题，使用默认配置
  python test_core_mmlu.py --limit 50

  # 测试完整验证集，使用自定义 LLM
  python test_core_mmlu.py --llm_name "gpt-4o"

  # 自定义 Agent 配置
  python test_core_mmlu.py --roles "Mathematician" "Critic" --max_routing 3
        """
    )

    # 模型配置
    parser.add_argument(
        '--llm_name',
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help='LLM model name (default: Qwen/Qwen3-4B-Instruct-2507)'
    )

    # Agent 配置
    parser.add_argument(
        '--roles',
        nargs='+',
        default=['Knowlegable Expert', 'Critic', 'Mathematician', 'Psychologist', 'Historian', 'Lawyer'],
        # , 'Doctor', 'Economist', 'Programmer'
        help='List of agent roles'
    )

    parser.add_argument(
        '--decision_method',
        type=str,
        default='FinalRefer',
        help='Decision method (default: FinalRefer)'
    )

    # 执行配置
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=1,
        help='Number of rounds per agent (default: 1)'
    )

    parser.add_argument(
        '--max_routing',
        type=int,
        default=5,
        help='Maximum routing steps (default: 5)'
    )

    # 测试配置
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to test (default: all)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for parallel processing (default: 1)'
    )

    # 输出配置
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save detailed results to file'
    )

    parser.add_argument(
        '--weave_project',
        type=str,
        default='vito_chan/CoRe-MMLU-Test',
        help='Weave project name for logging'
    )

    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()

    # 初始化 Weave
    weave.init(project_name=args.weave_project)

    print("\n" + "=" * 80)
    print("CoRe GRAPH SYSTEM - MMLU PERFORMANCE TEST")
    print("=" * 80)
    print(f"Version: v4.3.2")
    print(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 运行测试
    try:
        metrics = await test_core_graph_on_mmlu(
            llm_name=args.llm_name,
            available_roles=args.roles,
            decision_method=args.decision_method,
            num_rounds=args.num_rounds,
            max_routing=args.max_routing,
            limit_questions=args.limit,
            batch_size=args.batch_size,
            save_results=not args.no_save
        )

        # 额外分析
        print("\n" + "=" * 80)
        print("ADDITIONAL ANALYSIS")
        print("=" * 80)

        # 分析路由路径
        if metrics.routing_paths:
            avg_path_length = sum(len(path) for path in metrics.routing_paths) / len(metrics.routing_paths)
            print(f"\n🛣️  Routing Path Analysis:")
            print(f"  Average Path Length: {avg_path_length:.1f} steps")

            # 统计最常用的 Agent
            from collections import Counter
            all_agents = []
            for path in metrics.routing_paths:
                all_agents.extend([d['selected'] for d in path])

            agent_counts = Counter(all_agents)
            print(f"\n  Most Used Agents:")
            for agent, count in agent_counts.most_common(5):
                usage_rate = count / len(all_agents)
                print(f"    {agent}: {count} times ({usage_rate:.1%})")

        # 分析错误模式
        if metrics.error_cases:
            print(f"\n❌ Error Analysis:")
            print(f"  Total Errors: {len(metrics.error_cases)}")
            print(f"\n  Sample Error Cases:")
            for i, error in enumerate(metrics.error_cases[:3], 1):
                print(f"\n  {i}. Question: {error['question'][:80]}...")
                print(f"     Predicted: {error['predicted']}")
                print(f"     Target: {error['target']}")
                print(f"     Path: {' -> '.join(error['routing_path'])}")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

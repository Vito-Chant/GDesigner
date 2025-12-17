"""
CoRe Graph System Performance Testing on HumanEval
测试 CoRe v4.3.2 在 HumanEval 数据集上的性能

Usage:
    python test_core_humaneval.py --llm_name "your-model" --limit 50
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

# 导入 HumanEval 数据集和执行器
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.tools.coding.python_executor import PyExecutor


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
               is_correct: bool,
               result: 'CoReResult',
               task: str,
               generated_code: str,
               test: str,
               feedback: str = ""):
        """更新指标"""
        self.correct += int(is_correct)
        self.total += 1

        self.total_time += result.total_time
        self.total_tokens += result.total_cost_tokens
        self.total_steps += len(result.execution_trace)
        self.kv_cache_hits += result.kv_cache_hits
        self.loop_detections += result.loop_detections

        # 记录详细信息
        self.results.append({
            'task': task[:100] + '...' if len(task) > 100 else task,
            'generated_code': generated_code[:200] + '...' if len(generated_code) > 200 else generated_code,
            'correct': is_correct,
            'time': result.total_time,
            'tokens': result.total_cost_tokens,
            'steps': len(result.execution_trace),
            'routing_path': [d['selected'] for d in result.routing_decisions],
            'feedback': feedback[:200] if feedback else ""
        })

        self.routing_paths.append(result.routing_decisions)

        if not is_correct:
            self.error_cases.append({
                'task': task,
                'generated_code': generated_code,
                'test': test,
                'feedback': feedback,
                'routing_path': [d['selected'] for d in result.routing_decisions]
            })

    def get_pass_at_1(self) -> float:
        """计算 Pass@1 准确率"""
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

        print(f"\n📊 Code Generation Metrics:")
        print(f"  Passed:   {self.correct}/{self.total}")
        print(f"  Pass@1:   {self.get_pass_at_1():.2%}")

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
                'pass_at_1': self.get_pass_at_1(),
                'passed': self.correct,
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


def extract_code_from_response(response: str) -> str:
    """从响应中提取 Python 代码"""
    # 尝试提取 ```python 代码块
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
        return code.strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
        return code.strip()
    else:
        # 如果没有代码块标记，返回整个响应
        return response.strip()


async def test_core_graph_on_humaneval(
        llm_name: str,
        available_roles: List[str],
        decision_method: str = "FinalWriteCode",
        num_rounds: int = 1,
        max_routing: int = 5,
        limit_questions: int = None,
        batch_size: int = 1,
        timeout: int = 100,
        save_results: bool = True,
        **kwargs
):
    """
    在 HumanEval 数据集上测试 CoRe Graph

    Args:
        llm_name: LLM 模型名称
        available_roles: 可用的 Agent 角色列表
        decision_method: 决策方法
        num_rounds: 每个 Agent 的轮数
        max_routing: 最大路由步数
        limit_questions: 限制测试的问题数量
        batch_size: 批处理大小
        timeout: 代码执行超时时间（秒）
        save_results: 是否保存详细结果
    """

    # 初始化 CoRe Graph
    print("\n" + "=" * 80)
    print("INITIALIZING CoRe GRAPH SYSTEM FOR HUMANEVAL")
    print("=" * 80)
    print(f"  LLM: {llm_name}")
    print(f"  Roles: {', '.join(available_roles)}")
    print(f"  Decision Method: {decision_method}")
    print(f"  Max Routing Steps: {max_routing}")
    print(f"  Num Rounds: {num_rounds}")
    print(f"  Code Execution Timeout: {timeout}s")
    print("=" * 80 + "\n")

    core_graph = CoReGraph(
        domain="humaneval",
        llm_name=llm_name,
        available_roles=available_roles,
        decision_method=decision_method,
        max_routing=max_routing,
        registry_save_path=None,
        rag_top_k=3,
        max_loop_count=4
    )

    # 加载 HumanEval 数据集
    print("Loading HumanEval dataset...")
    dataset_path = GDesigner_ROOT / "dataset" / "humaneval" / "humaneval-py.jsonl"
    dataset = JSONLReader.parse_file(dataset_path)

    if not dataset:
        raise FileNotFoundError(f"HumanEval dataset not found at {dataset_path}")

    # 限制问题数量
    if limit_questions:
        total_questions = min(len(dataset), limit_questions)
        # dataset = dataset[:total_questions]
        dataset = dataset[-total_questions:]
    else:
        total_questions = len(dataset)

    print(f"Testing on {total_questions} coding problems\n")

    # 初始化指标收集器
    metrics = PerformanceMetrics()

    # 重置全局计数器
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    # 初始化代码执行器
    executor = PyExecutor()

    # 批处理执行
    num_batches = math.ceil(total_questions / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_questions)

        batch_tasks = []
        batch_records = []

        for idx in range(start_idx, end_idx):
            record = dataset[idx]
            task = record["prompt"]
            input_dict = {"task": task}

            batch_tasks.append(
                core_graph.run_cognitive_relay(input_dict)
            )
            batch_records.append(record)

        # 并发执行批次
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 处理结果
        for record, result in zip(batch_records, batch_results):
            task = record["prompt"]
            test = record["test"]

            if isinstance(result, Exception):
                print(f"\n❌ Error processing task: {result}")
                # 记录为错误
                metrics.total += 1
                metrics.error_cases.append({
                    'task': task,
                    'error': str(result)
                })
                continue

            # 提取生成的代码
            generated_code = extract_code_from_response(result.final_answer)

            # 执行代码测试
            try:
                is_passing, feedback, _ = executor.execute(
                    generated_code,
                    [test],
                    timeout=timeout
                )
            except Exception as e:
                is_passing = False
                feedback = f"Execution error: {str(e)}"

            # 更新指标
            metrics.update(
                is_correct=is_passing,
                result=result,
                task=task,
                generated_code=generated_code,
                test=test,
                feedback=feedback
            )

        # 每个批次后打印进度
        if (batch_idx + 1) % 5 == 0:
            print(f"\n--- Progress: {end_idx}/{total_questions} ---")
            print(f"  Current Pass@1: {metrics.get_pass_at_1():.2%}")
            print(f"  Avg Time: {metrics.get_avg_time():.2f}s")
            print(f"  Avg Tokens: {metrics.get_avg_tokens():.0f}")

    # 打印最终结果
    metrics.print_summary()

    # 保存详细结果
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = GDesigner_ROOT / "result" / "core_humaneval"
        result_dir.mkdir(parents=True, exist_ok=True)

        output_file = result_dir / f"core_test_{llm_name.replace('/', '_')}_{timestamp}.json"
        metrics.save_results(output_file)

    if "wandb_run" in kwargs:
        kwargs["wandb_run"].log({
            "pass_at_1": metrics.get_pass_at_1(),
            "avg_time": metrics.get_avg_time(),
            "avg_tokens": metrics.get_avg_tokens(),
            "avg_steps": metrics.get_avg_steps(),
            "kv_cache_hit_rate": metrics.get_kv_cache_hit_rate(),
            "loop_rate": metrics.get_loop_rate(),
            "total_cost": Cost.instance().value
        })

    return metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Test CoRe Graph System on HumanEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 测试 50 个问题，使用默认配置
  python test_core_humaneval.py --limit 50

  # 测试完整数据集，使用自定义 LLM
  python test_core_humaneval.py --llm_name "gpt-4o"

  # 自定义 Agent 配置
  python test_core_humaneval.py --roles "Project Manager" "Algorithm Designer" "Programming Expert" --max_routing 8
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
        default=['Programming_Expert'],  # , 'Project_Manager', 'Algorithm_Designer', 'Test_Analyst'
        help='List of agent roles for code generation'
    )

    parser.add_argument(
        '--decision_method',
        type=str,
        default='FinalWriteCode',
        help='Decision method (default: FinalWriteCode)'
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
        default=8,
        help='Maximum routing steps (default: 8)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=100,
        help='Code execution timeout in seconds (default: 100)'
    )

    # 测试配置
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems to test (default: all 164 problems)'
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
        default='vito_chan/CoRe-HumanEval-Test',
        help='Weave project name for logging'
    )

    return parser.parse_args()


async def main():
    """主函数"""
    import wandb

    args = parse_args()

    # 处理角色名称（支持空格和下划线）
    if len(args.roles) == 1 and ' ' in args.roles[0]:
        args.roles = args.roles[0].split()
    args.roles = [r.replace('_', ' ') for r in args.roles]

    # 初始化 Weave 和 WandB
    weave.init(project_name=args.weave_project)
    wandb_run = wandb.init(
        project="CoRe-HumanEval-Test",
        config=args,
        name=time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    print("\n" + "=" * 80)
    print("CoRe GRAPH SYSTEM - HUMANEVAL PERFORMANCE TEST")
    print("=" * 80)
    print(f"Version: v4.3.2")
    print(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 运行测试
    try:
        metrics = await test_core_graph_on_humaneval(
            llm_name=args.llm_name,
            available_roles=args.roles,
            decision_method=args.decision_method,
            num_rounds=args.num_rounds,
            max_routing=args.max_routing,
            limit_questions=args.limit,
            batch_size=args.batch_size,
            timeout=args.timeout,
            save_results=not args.no_save,
            wandb_run=wandb_run
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
            print(f"  Total Failed: {len(metrics.error_cases)}")
            print(f"\n  Sample Failed Cases:")
            for i, error in enumerate(metrics.error_cases[:3], 1):
                print(f"\n  {i}. Task: {error['task'][:80]}...")
                print(f"     Feedback: {error.get('feedback', 'N/A')[:100]}...")
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

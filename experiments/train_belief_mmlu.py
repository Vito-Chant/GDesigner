"""
CoRe Graph Belief Training on MMLU Dev Set - v5.0 (Generalized Beliefs)
在 MMLU dev 集上训练泛化信念系统，然后在 val 集上测试

主要改动:
1. 使用 GeneralizedBelief 替代 BeliefUpdate
2. 更新统计指标以反映能力维度
3. 改进可视化输出
4. 优化信念更新逻辑

Usage:
    python experiments/train_belief_mmlu.py --llm_name "your-model" --train_samples 100
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
from collections import defaultdict

import weave

# 导入 CoRe 组件 (v5.0)
from GDesigner.CoRe.core_graph import CoReGraph, CoReResult
from GDesigner.CoRe.generalized_belief import GeneralizedBelief  # 新增
from GDesigner.CoRe.capability_taxonomy import ALL_CAPABILITIES  # 新增
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT

# 导入 MMLU 数据集
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


# ============================================================================
# 统计指标收集器 (v5.0 升级版)
# ============================================================================

class BeliefTrainingMetricsV5:
    """
    信念训练指标收集器 (v5.0 升级版)

    新增特性:
    - 按能力维度统计
    - 跟踪信念泛化质量
    - 记录能力覆盖率
    """

    def __init__(self):
        # 基础统计
        self.total_samples = 0
        self.correct_samples = 0
        self.total_belief_updates = 0

        # 按能力维度统计
        self.updates_by_capability = defaultdict(int)  # {capability: count}
        self.success_by_capability = defaultdict(lambda: {'success': 0, 'total': 0})

        # 信念质量统计
        self.generalized_beliefs_count = 0  # 泛化信念数量
        self.belief_quality_scores = []  # 信念质量分数

        # 详细记录
        self.belief_changes = []
        self.capability_coverage = set()  # 涉及的能力维度

    def update(
        self,
        is_correct: bool,
        belief_updates: List[GeneralizedBelief],
        result: CoReResult
    ):
        """
        更新训练指标 (v5.0)

        Args:
            is_correct: 任务是否成功
            belief_updates: 信念更新列表 (现在是 GeneralizedBelief)
            result: CoRe 执行结果
        """
        self.total_samples += 1
        if is_correct:
            self.correct_samples += 1

        self.total_belief_updates += len(belief_updates)

        for belief in belief_updates:
            # 统计能力维度
            capability = belief.capability_dimension
            self.updates_by_capability[capability] += 1
            self.capability_coverage.add(capability)

            # 统计成功率
            if is_correct:
                self.success_by_capability[capability]['success'] += 1
            self.success_by_capability[capability]['total'] += 1

            # 评估信念质量
            quality_score = self._evaluate_belief_quality(belief)
            self.belief_quality_scores.append(quality_score)

            if quality_score >= 0.8:
                self.generalized_beliefs_count += 1

            # 记录变化
            self.belief_changes.append({
                'capability': capability,
                'from': belief.from_agent,
                'to': belief.to_agent,
                'description': belief.general_description,
                'success_rate': belief.success_rate,
                'contexts': belief.applicable_contexts,
                'limitations': belief.known_limitations,
                'quality_score': quality_score
            })

    def _evaluate_belief_quality(self, belief: GeneralizedBelief) -> float:
        """
        评估信念质量 (0-1)

        标准:
        1. 描述简洁性 (< 80 字符)
        2. 不包含具体数字
        3. 包含适用场景
        4. 样本数量充足
        """
        import re

        score = 1.0

        # 1. 长度检查
        if len(belief.general_description) > 80:
            score -= 0.2

        # 2. 具体性检查 (不应包含大数字)
        if re.search(r'\b\d{2,}\b', belief.general_description):
            score -= 0.3

        # 3. 上下文丰富度
        if not belief.applicable_contexts:
            score -= 0.2

        # 4. 样本充足性
        if belief.total_count < 2:
            score -= 0.3

        return max(0.0, score)

    def get_accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0.0

    def get_avg_quality_score(self) -> float:
        if not self.belief_quality_scores:
            return 0.0
        return sum(self.belief_quality_scores) / len(self.belief_quality_scores)

    def print_summary(self):
        """打印训练摘要 (v5.0 增强版)"""
        print("\n" + "=" * 80)
        print("BELIEF TRAINING SUMMARY (v5.0 Generalized Beliefs)")
        print("=" * 80)

        print(f"\n📊 Training Metrics:")
        print(f"  Samples Processed: {self.total_samples}")
        print(f"  Accuracy: {self.get_accuracy():.2%} ({self.correct_samples}/{self.total_samples})")

        print(f"\n🧠 Belief Statistics:")
        print(f"  Total Updates: {self.total_belief_updates}")
        print(f"  High-Quality Beliefs: {self.generalized_beliefs_count} ({self.generalized_beliefs_count/max(1, self.total_belief_updates):.1%})")
        print(f"  Avg Quality Score: {self.get_avg_quality_score():.2f}")
        print(f"  Capability Coverage: {len(self.capability_coverage)} dimensions")

        print(f"\n📋 Capability Breakdown:")
        # 按更新次数排序
        sorted_caps = sorted(
            self.updates_by_capability.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for capability, count in sorted_caps[:10]:  # 显示前10个
            cap_name = capability.replace('_', ' ').title()

            # 计算该能力的成功率
            stats = self.success_by_capability[capability]
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                print(f"  {cap_name:30s}: {count:3d} updates | Success: {success_rate:.1%}")
            else:
                print(f"  {cap_name:30s}: {count:3d} updates")

        if len(sorted_caps) > 10:
            print(f"  ... and {len(sorted_caps) - 10} more capabilities")

        print(f"\n🔍 Recent High-Quality Beliefs (last 5):")
        high_quality = [
            change for change in self.belief_changes
            if change['quality_score'] >= 0.8
        ][-5:]

        for i, change in enumerate(high_quality, 1):
            cap_name = change['capability'].replace('_', ' ').title()
            print(f"\n  {i}. [{cap_name}] {change['from']} → {change['to']}")
            print(f"     \"{change['description']}\"")
            print(f"     Success Rate: {change['success_rate']:.1%}")
            if change['contexts']:
                print(f"     Contexts: {', '.join(change['contexts'][:2])}")
            if change['limitations']:
                print(f"     Limitations: {', '.join(change['limitations'][:2])}")
            print(f"     Quality: {change['quality_score']:.2f}")


# ============================================================================
# 主训练函数 (v5.0 适配)
# ============================================================================

async def train_beliefs_on_mmlu_dev(
    llm_name: str,
    available_roles: List[str],
    decision_method: str = "FinalRefer",
    num_rounds: int = 1,
    max_routing: int = 5,
    train_samples: int = 100,
    batch_size: int = 1,
    save_registry: bool = True,
    **kwargs
):
    """
    在 MMLU dev 集上训练泛化信念系统 (v5.0)

    主要改动:
    - 使用 GeneralizedBelief
    - 自动任务抽象
    - 按能力维度统计
    """

    print("\n" + "=" * 80)
    print("BELIEF TRAINING ON MMLU DEV SET (v5.0 Generalized)")
    print("=" * 80)
    print(f"  LLM: {llm_name}")
    print(f"  Roles: {', '.join(available_roles)}")
    print(f"  Training Samples: {train_samples}")
    print(f"  Batch Size: {batch_size}")
    print("=" * 80 + "\n")

    # 准备保存路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = GDesigner_ROOT / "result" / "belief_training_v5"
    result_dir.mkdir(parents=True, exist_ok=True)

    registry_save_path = None
    if save_registry:
        registry_save_path = result_dir / f"mind_registry_v5_{timestamp}.json"

    # === 关键修改: 初始化 CoReGraph (会自动使用 v5.0 组件) ===
    core_graph = CoReGraph(
        domain="mmlu",
        llm_name=llm_name,
        available_roles=available_roles,
        decision_method=decision_method,
        max_routing=max_routing,
        registry_save_path=registry_save_path,  # 会使用 MindRegistryV5
        rag_top_k=3,
        max_loop_count=4
    )

    # 验证是否使用了 v5.0 组件
    print(f"[Verification] Using MindRegistry type: {type(core_graph.mind_registry).__name__}")
    print(f"[Verification] Using BeliefEvolver type: {type(core_graph.belief_evolver).__name__}")

    # 加载训练数据集
    print("\nLoading MMLU dev dataset...")
    download()
    train_dataset = MMLUDataset('dev')

    total_samples = min(len(train_dataset), train_samples)
    print(f"Training on {total_samples} samples from dev set\n")

    # === 使用新的指标收集器 ===
    metrics = BeliefTrainingMetricsV5()

    # 重置计数器
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    # 批处理训练
    num_batches = math.ceil(total_samples / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Training batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        batch_tasks = []
        batch_records = []

        for idx in range(start_idx, end_idx):
            record = train_dataset[idx]
            input_dict = train_dataset.record_to_input(record)

            batch_tasks.append(
                core_graph.run_cognitive_relay(input_dict)
            )
            batch_records.append(record)

        # 并发执行批次
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 处理结果并更新信念
        for record, result in zip(batch_records, batch_results):
            if isinstance(result, Exception):
                print(f"\n❌ Error: {result}")
                continue

            # 获取答案
            predicted = train_dataset.postprocess_answer(result.final_answer)
            target = train_dataset.record_to_target_answer(record)
            is_correct = (predicted == target)

            # === 关键步骤: 信念更新 (v5.0 会自动生成泛化信念) ===
            belief_updates = await process_execution_trace_and_update_beliefs_v5(
                core_graph=core_graph,
                result=result,
                task_success=is_correct,
                question=train_dataset.record_to_input(record)['task']
            )

            # 更新指标
            metrics.update(is_correct, belief_updates, result)

        # 每 10 个批次打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"\n--- Progress: {end_idx}/{total_samples} ---")
            print(f"  Current Accuracy: {metrics.get_accuracy():.2%}")
            print(f"  Total Belief Updates: {metrics.total_belief_updates}")
            print(f"  High-Quality Beliefs: {metrics.generalized_beliefs_count}")
            print(f"  Avg Quality Score: {metrics.get_avg_quality_score():.2f}")
            print(f"  Capability Coverage: {len(metrics.capability_coverage)} dimensions")
            print(f"  Avg Cost: ${Cost.instance().value:.4f}")

    # 打印训练总结
    metrics.print_summary()

    print(f"\n💰 Training Cost:")
    print(f"  Total: ${Cost.instance().value:.4f}")
    print(f"  Prompt Tokens: {PromptTokens.instance().value / 1000:.1f}k")
    print(f"  Completion Tokens: {CompletionTokens.instance().value / 1000:.1f}k")

    # === 保存训练报告 (v5.0 增强版) ===
    training_report = {
        'version': 'v5.0',
        'config': {
            'llm_name': llm_name,
            'available_roles': available_roles,
            'train_samples': total_samples,
            'batch_size': batch_size
        },
        'metrics': {
            'accuracy': metrics.get_accuracy(),
            'total_samples': metrics.total_samples,
            'correct_samples': metrics.correct_samples,
            'total_belief_updates': metrics.total_belief_updates,
            'high_quality_beliefs': metrics.generalized_beliefs_count,
            'avg_quality_score': metrics.get_avg_quality_score(),
            'capability_coverage': len(metrics.capability_coverage)
        },
        'capability_breakdown': {
            cap: count for cap, count in metrics.updates_by_capability.items()
        },
        'belief_changes': metrics.belief_changes[-50:],  # 最近50条
        'cost': {
            'total': Cost.instance().value,
            'prompt_tokens': PromptTokens.instance().value,
            'completion_tokens': CompletionTokens.instance().value
        }
    }

    report_path = result_dir / f"training_report_v5_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Training report saved to: {report_path}")

    if save_registry:
        print(f"✓ Mind registry saved to: {registry_save_path}")

    # === WandB 日志 (v5.0 新增指标) ===
    if "wandb_run" in kwargs:
        kwargs["wandb_run"].log({
            "train/accuracy": metrics.get_accuracy(),
            "train/belief_updates": metrics.total_belief_updates,
            "train/high_quality_beliefs": metrics.generalized_beliefs_count,
            "train/avg_quality_score": metrics.get_avg_quality_score(),
            "train/capability_coverage": len(metrics.capability_coverage),
            "train/cost": Cost.instance().value
        })

    return core_graph, registry_save_path


# ============================================================================
# 信念更新处理器 (v5.0 适配)
# ============================================================================

async def process_execution_trace_and_update_beliefs_v5(
    core_graph: CoReGraph,
    result: CoReResult,
    task_success: bool,
    question: str
) -> List[GeneralizedBelief]:
    """
    处理执行轨迹并更新信念 (v5.0 适配版)

    主要改动:
    - 返回 GeneralizedBelief 而非 BeliefUpdate
    - 支持能力维度分组
    - 自动任务抽象

    Returns:
        List of GeneralizedBelief objects
    """
    from GDesigner.CoRe.belief_evolver import InteractionTrace  # 保持原有导入

    all_updates = []

    # 检查执行轨迹
    if not result.execution_trace:
        print("[Warning] Empty execution trace, skipping belief update")
        return all_updates

    # 遍历执行轨迹
    for i, trace_step in enumerate(result.execution_trace):
        try:
            # === 提取当前步骤信息 ===
            current_agent = trace_step.get('agent', 'unknown')
            current_output = trace_step.get('output', '')

            # 处理 tuple 格式
            if isinstance(current_output, tuple):
                current_output = current_output[0] if len(current_output) > 0 else ''

            # === 找到对应的路由决策 ===
            if i < len(result.routing_decisions):
                routing = result.routing_decisions[i]
                suggestion = routing.get('suggestion', 'Continue the work')
            else:
                suggestion = 'Complete the task'

            # === 确定下一个 Agent ===
            if i + 1 < len(result.execution_trace):
                next_step = result.execution_trace[i + 1]
                next_agent = next_step.get('agent', 'unknown')
            else:
                next_agent = core_graph.decision_maker_id

            # === 构建 InteractionTrace ===
            interaction = InteractionTrace(
                from_agent=current_agent,
                to_agent=next_agent,
                task=question,
                suggestion=suggestion,
                output=current_output,
                success=task_success,
                failure_reason=None if task_success else "Task failed"
            )

            # === 构建完整交互链 ===
            full_chain = []
            for j, step in enumerate(result.execution_trace):
                step_agent = step.get('agent', 'unknown')
                step_output = step.get('output', '')
                if isinstance(step_output, tuple):
                    step_output = step_output[0] if len(step_output) > 0 else ''

                if j + 1 < len(result.execution_trace):
                    step_next = result.execution_trace[j + 1].get('agent', 'unknown')
                else:
                    step_next = core_graph.decision_maker_id

                step_trace = InteractionTrace(
                    from_agent=step_agent,
                    to_agent=step_next,
                    task=question,
                    suggestion=result.routing_decisions[j].get('suggestion', '') if j < len(result.routing_decisions) else '',
                    output=step_output,
                    success=(j == i) if not task_success else True
                )
                full_chain.append(step_trace)

            # === 调用 BeliefEvolver 更新信念 (v5.0 会自动生成泛化信念) ===
            updates = await core_graph.belief_evolver.evolve_beliefs_from_interaction(
                interaction_trace=interaction,
                full_chain=full_chain,
                task_success=task_success,
                critic_feedback=None
            )

            all_updates.extend(updates)

        except Exception as e:
            import traceback
            print(f"[Warning] Belief update failed for {current_agent}:")
            print(f"  Error: {e}")
            traceback.print_exc()
            continue

    return all_updates


# ============================================================================
# 测试函数 (保持不变)
# ============================================================================

async def test_with_trained_beliefs(
    core_graph: CoReGraph,
    test_samples: int = None,
    batch_size: int = 4,
    **kwargs
):
    """使用训练好的信念在 val 集上测试"""

    print("\n" + "=" * 80)
    print("TESTING WITH TRAINED BELIEFS ON MMLU VAL SET")
    print("=" * 80)

    test_dataset = MMLUDataset('val')
    total_samples = min(len(test_dataset), test_samples) if test_samples else len(test_dataset)

    print(f"Testing on {total_samples} samples from val set\n")

    # 重置计数器
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    correct = 0
    total = 0

    num_batches = math.ceil(total_samples / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Testing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        batch_tasks = []
        batch_records = []

        for idx in range(start_idx, end_idx):
            record = test_dataset[idx]
            input_dict = test_dataset.record_to_input(record)

            batch_tasks.append(
                core_graph.run_cognitive_relay(input_dict)
            )
            batch_records.append(record)

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for record, result in zip(batch_records, batch_results):
            if isinstance(result, Exception):
                continue

            predicted = test_dataset.postprocess_answer(result.final_answer)
            target = test_dataset.record_to_target_answer(record)

            if predicted == target:
                correct += 1
            total += 1

        if (batch_idx + 1) % 10 == 0:
            accuracy = correct / total if total > 0 else 0
            print(f"\n--- Progress: {end_idx}/{total_samples} ---")
            print(f"  Current Accuracy: {accuracy:.2%}")

    final_accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"\n📊 Performance:")
    print(f"  Accuracy: {final_accuracy:.2%} ({correct}/{total})")
    print(f"\n💰 Test Cost:")
    print(f"  Total: ${Cost.instance().value:.4f}")
    print(f"  Prompt Tokens: {PromptTokens.instance().value / 1000:.1f}k")
    print(f"  Completion Tokens: {CompletionTokens.instance().value / 1000:.1f}k")

    if "wandb_run" in kwargs:
        kwargs["wandb_run"].log({
            "test/accuracy": final_accuracy,
            "test/correct": correct,
            "test/total": total,
            "test/cost": Cost.instance().value
        })

    return final_accuracy


# ============================================================================
# 命令行参数解析 (保持不变)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CoRe Belief System on MMLU (v5.0 Generalized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--llm_name', type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument('--roles', nargs='+', default=['Mathematician', 'Programmer', 'Knowlegable Expert', 'Critic'])
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--max_routing', type=int, default=5)
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_samples', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--no_save_registry', action='store_true')
    parser.add_argument('--weave_project', type=str, default='vito_chan/CoRe-Belief-Training-V5')

    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

async def main():
    import wandb

    args = parse_args()

    # 处理角色名称
    if len(args.roles) == 1 and ' ' in args.roles[0]:
        args.roles = args.roles[0].split()
    args.roles = [r.replace('_', ' ') for r in args.roles]

    # 初始化追踪
    weave.init(project_name=args.weave_project)
    wandb_run = wandb.init(
        project="CoRe-Belief-Training-V5",
        config=args,
        name=f"v5.0_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    print("\n" + "=" * 80)
    print("CoRe BELIEF TRAINING & TESTING PIPELINE (v5.0 Generalized)")
    print("=" * 80)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Version: Generalized Belief System v5.0")
    print("=" * 80)

    try:
        # === 阶段 1: 训练信念 ===
        core_graph, registry_path = await train_beliefs_on_mmlu_dev(
            llm_name=args.llm_name,
            available_roles=args.roles,
            decision_method=args.decision_method,
            num_rounds=args.num_rounds,
            max_routing=args.max_routing,
            train_samples=args.train_samples,
            batch_size=args.batch_size,
            save_registry=not args.no_save_registry,
            wandb_run=wandb_run
        )

        print("\n" + "=" * 80)
        print("✅ BELIEF TRAINING COMPLETED")
        print("=" * 80)

        # === 阶段 2: 测试 ===
        if args.test_samples is not None or input("\nRun testing? (y/n): ").lower() == 'y':
            accuracy = await test_with_trained_beliefs(
                core_graph=core_graph,
                test_samples=args.test_samples,
                batch_size=args.test_batch_size,
                wandb_run=wandb_run
            )

            print("\n" + "=" * 80)
            print("✅ TESTING COMPLETED")
            print(f"Final Accuracy: {accuracy:.2%}")
            print("=" * 80)

        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY (v5.0)")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
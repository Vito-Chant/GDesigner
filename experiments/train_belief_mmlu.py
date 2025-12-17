"""
CoRe Graph Belief Training on MMLU Dev Set
在 MMLU dev 集上训练 Belief Evolver，然后在 val 集上测试

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

import weave

# 导入 CoRe 组件
from GDesigner.CoRe.core_graph import CoReGraph, CoReResult
from GDesigner.CoRe.belief_evolver import InteractionTrace, BeliefUpdate
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.utils.const import GDesigner_ROOT

# 导入 MMLU 数据集
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


class BeliefTrainingMetrics:
    """信念训练指标收集器"""

    def __init__(self):
        self.total_samples = 0
        self.correct_samples = 0
        self.total_belief_updates = 0
        self.positive_updates = 0
        self.negative_updates = 0

        # 按更新类型分类
        self.update_types = {
            'positive_reinforcement': 0,
            'failure_attribution': 0,
            'nuanced_update': 0
        }

        # 信念变化记录
        self.belief_changes = []

    def update(self, is_correct: bool, belief_updates: List[BeliefUpdate], result: CoReResult):
        """更新训练指标"""
        self.total_samples += 1
        if is_correct:
            self.correct_samples += 1

        self.total_belief_updates += len(belief_updates)

        for update in belief_updates:
            # 统计正负更新
            if update.confidence_change > 0:
                self.positive_updates += 1
            elif update.confidence_change < 0:
                self.negative_updates += 1

            # 记录变化
            self.belief_changes.append({
                'from': update.from_agent,
                'to': update.to_agent,
                'old_belief': update.old_belief,
                'new_belief': update.new_belief,
                'confidence_change': update.confidence_change,
                'reason': update.update_reason
            })

    def get_accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0.0

    def print_summary(self):
        print("\n" + "=" * 80)
        print("BELIEF TRAINING SUMMARY")
        print("=" * 80)

        print(f"\n📊 Training Metrics:")
        print(f"  Samples Processed: {self.total_samples}")
        print(f"  Accuracy: {self.get_accuracy():.2%} ({self.correct_samples}/{self.total_samples})")

        print(f"\n🧠 Belief Updates:")
        print(f"  Total Updates: {self.total_belief_updates}")
        print(f"  Positive (↑): {self.positive_updates}")
        print(f"  Negative (↓): {self.negative_updates}")
        print(f"  Neutral (→): {self.total_belief_updates - self.positive_updates - self.negative_updates}")

        if self.belief_changes:
            print(f"\n📈 Recent Belief Changes (last 5):")
            for i, change in enumerate(self.belief_changes[-5:], 1):
                print(f"\n  {i}. {change['from']} → {change['to']}")
                print(f"     New: {change['new_belief'][:80]}...")
                print(f"     Δ Confidence: {change['confidence_change']:+.2f}")
                print(f"     Reason: {change['reason'][:60]}...")


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
    在 MMLU dev 集上训练信念系统

    Args:
        llm_name: LLM 模型名称
        available_roles: 可用的 Agent 角色列表
        decision_method: 决策方法
        num_rounds: 每个 Agent 的轮数
        max_routing: 最大路由步数
        train_samples: 训练样本数量
        batch_size: 批处理大小
        save_registry: 是否保存 MindRegistry
    """

    print("\n" + "=" * 80)
    print("BELIEF TRAINING ON MMLU DEV SET")
    print("=" * 80)
    print(f"  LLM: {llm_name}")
    print(f"  Roles: {', '.join(available_roles)}")
    print(f"  Training Samples: {train_samples}")
    print(f"  Batch Size: {batch_size}")
    print("=" * 80 + "\n")

    # 准备保存路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = GDesigner_ROOT / "result" / "belief_training"
    result_dir.mkdir(parents=True, exist_ok=True)

    registry_save_path = None
    if save_registry:
        registry_save_path = result_dir / f"mind_registry_{timestamp}.json"

    # 初始化 CoRe Graph
    core_graph = CoReGraph(
        domain="mmlu",
        llm_name=llm_name,
        available_roles=available_roles,
        decision_method=decision_method,
        max_routing=max_routing,
        registry_save_path=registry_save_path,
        rag_top_k=3,
        max_loop_count=4
    )

    # 加载训练数据集
    print("Loading MMLU dev dataset...")
    download()
    train_dataset = MMLUDataset('dev')

    # 限制训练样本数量
    total_samples = min(len(train_dataset), train_samples)
    print(f"Training on {total_samples} samples from dev set\n")

    # 初始化指标
    metrics = BeliefTrainingMetrics()

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

            # === 关键步骤：信念更新 ===
            belief_updates = await process_execution_trace_and_update_beliefs(
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
            print(f"  Avg Cost: ${Cost.instance().value:.4f}")

    # 打印训练总结
    metrics.print_summary()

    print(f"\n💰 Training Cost:")
    print(f"  Total: ${Cost.instance().value:.4f}")
    print(f"  Prompt Tokens: {PromptTokens.instance().value / 1000:.1f}k")
    print(f"  Completion Tokens: {CompletionTokens.instance().value / 1000:.1f}k")

    # 保存训练报告
    training_report = {
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
            'positive_updates': metrics.positive_updates,
            'negative_updates': metrics.negative_updates
        },
        'belief_changes': metrics.belief_changes,
        'cost': {
            'total': Cost.instance().value,
            'prompt_tokens': PromptTokens.instance().value,
            'completion_tokens': CompletionTokens.instance().value
        }
    }

    report_path = result_dir / f"training_report_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Training report saved to: {report_path}")

    if save_registry:
        print(f"✓ Mind registry saved to: {registry_save_path}")

    if "wandb_run" in kwargs:
        kwargs["wandb_run"].log({
            "train/accuracy": metrics.get_accuracy(),
            "train/belief_updates": metrics.total_belief_updates,
            "train/positive_updates": metrics.positive_updates,
            "train/negative_updates": metrics.negative_updates,
            "train/cost": Cost.instance().value
        })

    return core_graph, registry_save_path


async def process_execution_trace_and_update_beliefs(
        core_graph: CoReGraph,
        result: CoReResult,
        task_success: bool,
        question: str
) -> List[BeliefUpdate]:
    """
    处理执行轨迹并更新信念 (v2 - 修复版)

    关键修复:
    1. 正确导入 InteractionTrace
    2. 处理 execution_trace 中的 dict 格式
    3. 构建完整的交互链
    """
    from GDesigner.CoRe.belief_evolver import InteractionTrace, BeliefUpdate

    all_updates = []

    # 检查执行轨迹是否为空
    if not result.execution_trace:
        print("[Warning] Empty execution trace, skipping belief update")
        return all_updates

    # 遍历执行轨迹（每个 trace_step 是一个 dict）
    for i, trace_step in enumerate(result.execution_trace):
        try:
            # === 1. 提取当前步骤信息 ===
            current_agent = trace_step.get('agent', 'unknown')
            current_output = trace_step.get('output', '')

            # 如果 output 是 tuple（来自 CoRe Agent），提取字符串部分
            if isinstance(current_output, tuple):
                current_output = current_output[0] if len(current_output) > 0 else ''

            # === 2. 找到对应的路由决策 ===
            if i < len(result.routing_decisions):
                routing = result.routing_decisions[i]
                suggestion = routing.get('suggestion', 'Continue the work')
            else:
                suggestion = 'Complete the task'

            # === 3. 确定下一个 Agent ===
            if i + 1 < len(result.execution_trace):
                next_step = result.execution_trace[i + 1]
                next_agent = next_step.get('agent', 'unknown')
            else:
                # 最后一步，下一个是 Decision Maker
                next_agent = core_graph.decision_maker_id

            # === 4. 构建 InteractionTrace 对象 ===
            interaction = InteractionTrace(
                from_agent=current_agent,
                to_agent=next_agent,
                task=question,
                suggestion=suggestion,
                output=current_output,
                success=task_success,  # 整体任务是否成功
                failure_reason=None if task_success else "Task failed"
            )

            # === 5. 构建完整的交互链（用于上下文分析）===
            # BeliefEvolver 需要完整链来做失败归因
            full_chain = []
            for j, step in enumerate(result.execution_trace):
                step_agent = step.get('agent', 'unknown')
                step_output = step.get('output', '')
                if isinstance(step_output, tuple):
                    step_output = step_output[0] if len(step_output) > 0 else ''

                # 确定这一步的下一个 Agent
                if j + 1 < len(result.execution_trace):
                    step_next = result.execution_trace[j + 1].get('agent', 'unknown')
                else:
                    step_next = core_graph.decision_maker_id

                # 构建 InteractionTrace
                step_trace = InteractionTrace(
                    from_agent=step_agent,
                    to_agent=step_next,
                    task=question,
                    suggestion=result.routing_decisions[j].get('suggestion', '') if j < len(
                        result.routing_decisions) else '',
                    output=step_output,
                    success=(j == i) if not task_success else True  # 只有失败的那一步标记为失败
                )
                full_chain.append(step_trace)

            # === 6. 调用 BeliefEvolver 更新信念 ===
            updates = await core_graph.belief_evolver.evolve_beliefs_from_interaction(
                interaction_trace=interaction,
                full_chain=full_chain,
                task_success=task_success,
                critic_feedback=None
            )

            all_updates.extend(updates)

        except Exception as e:
            # 详细的错误信息
            import traceback
            print(f"[Warning] Belief update failed for {current_agent}:")
            print(f"  Error: {e}")
            print(f"  Trace step: {trace_step}")
            traceback.print_exc()
            continue

    return all_updates


async def test_with_trained_beliefs(
        core_graph: CoReGraph,
        test_samples: int = None,
        batch_size: int = 4,
        **kwargs
):
    """
    使用训练好的信念在 val 集上测试
    """

    print("\n" + "=" * 80)
    print("TESTING WITH TRAINED BELIEFS ON MMLU VAL SET")
    print("=" * 80)

    # 加载测试数据集
    test_dataset = MMLUDataset('val')
    total_samples = min(len(test_dataset), test_samples) if test_samples else len(test_dataset)

    print(f"Testing on {total_samples} samples from val set\n")

    # 重置计数器
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    # 测试指标
    correct = 0
    total = 0

    # 批处理测试
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

        # 并发执行
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # 计算准确率
        for record, result in zip(batch_records, batch_results):
            if isinstance(result, Exception):
                continue

            predicted = test_dataset.postprocess_answer(result.final_answer)
            target = test_dataset.record_to_target_answer(record)

            if predicted == target:
                correct += 1
            total += 1

        # 进度输出
        if (batch_idx + 1) % 10 == 0:
            accuracy = correct / total if total > 0 else 0
            print(f"\n--- Progress: {end_idx}/{total_samples} ---")
            print(f"  Current Accuracy: {accuracy:.2%}")

    # 最终结果
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CoRe Belief System on MMLU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 训练 100 个样本
  python experiments/train_belief_mmlu.py --train_samples 100

  # 训练后直接测试
  python experiments/train_belief_mmlu.py --train_samples 200 --test_samples 500

  # 自定义 Agent 角色
  python experiments/train_belief_mmlu.py --roles "Mathematician" "Critic" --train_samples 150
        """
    )

    # 模型配置
    parser.add_argument(
        '--llm_name',
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help='LLM model name'
    )

    # Agent 配置
    parser.add_argument(
        '--roles',
        nargs='+',
        default=['Mathematician', 'Programmer', 'Knowlegable Expert', 'Critic'],
        help='List of agent roles'
    )

    parser.add_argument(
        '--decision_method',
        type=str,
        default='FinalRefer',
        help='Decision method'
    )

    # 执行配置
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=1,
        help='Number of rounds per agent'
    )

    parser.add_argument(
        '--max_routing',
        type=int,
        default=5,
        help='Maximum routing steps'
    )

    # 训练配置
    parser.add_argument(
        '--train_samples',
        type=int,
        default=100,
        help='Number of training samples from dev set'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )

    # 测试配置
    parser.add_argument(
        '--test_samples',
        type=int,
        default=None,
        help='Number of test samples from val set (None = all)'
    )

    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=4,
        help='Batch size for testing'
    )

    # 输出配置
    parser.add_argument(
        '--no_save_registry',
        action='store_true',
        help='Do not save MindRegistry'
    )

    parser.add_argument(
        '--weave_project',
        type=str,
        default='vito_chan/CoRe-Belief-Training',
        help='Weave project name'
    )

    return parser.parse_args()


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
        project="CoRe-Belief-Training",
        config=args,
        name=time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    print("\n" + "=" * 80)
    print("CoRe BELIEF TRAINING & TESTING PIPELINE")
    print("=" * 80)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # ===== 阶段 1: 训练信念 =====
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

        # ===== 阶段 2: 测试 =====
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
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
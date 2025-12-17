"""
信念可视化工具
用于分析和可视化训练后的 MindRegistry

Usage:
    python tools/visualize_beliefs.py --registry result/belief_training/mind_registry_*.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


def load_registry(registry_path: Path) -> Dict:
    """加载 MindRegistry"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_beliefs(beliefs: List[Dict]):
    """分析信念统计"""
    print("\n" + "=" * 80)
    print("BELIEF STATISTICS")
    print("=" * 80)

    # 按 Agent 统计
    from_counts = defaultdict(int)
    to_counts = defaultdict(int)
    confidence_by_agent = defaultdict(list)

    for belief in beliefs:
        from_agent = belief['from_agent']
        to_agent = belief['to_agent']
        confidence = belief['confidence']

        from_counts[from_agent] += 1
        to_counts[to_agent] += 1
        confidence_by_agent[from_agent].append(confidence)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total Beliefs: {len(beliefs)}")
    print(f"  Unique Agents (as evaluator): {len(from_counts)}")
    print(f"  Unique Agents (being evaluated): {len(to_counts)}")

    print(f"\n🧠 Most Active Evaluators:")
    for agent, count in sorted(from_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        avg_conf = sum(confidence_by_agent[agent]) / len(confidence_by_agent[agent])
        print(f"  {agent}: {count} beliefs (avg confidence: {avg_conf:.2f})")

    print(f"\n🎯 Most Evaluated Agents:")
    for agent, count in sorted(to_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {agent}: evaluated {count} times")

    # 信心度分布
    confidences = [b['confidence'] for b in beliefs]
    avg_conf = sum(confidences) / len(confidences)
    high_conf = sum(1 for c in confidences if c > 0.7)
    low_conf = sum(1 for c in confidences if c < 0.4)

    print(f"\n📈 Confidence Distribution:")
    print(f"  Average: {avg_conf:.2f}")
    print(f"  High confidence (>0.7): {high_conf} ({high_conf / len(beliefs) * 100:.1f}%)")
    print(f"  Low confidence (<0.4): {low_conf} ({low_conf / len(beliefs) * 100:.1f}%)")


def visualize_belief_network(beliefs: List[Dict], output_path: Path = None):
    """可视化信念网络"""
    print("\n" + "=" * 80)
    print("GENERATING BELIEF NETWORK VISUALIZATION")
    print("=" * 80)

    # 创建有向图
    G = nx.DiGraph()

    # 添加边（带权重）
    for belief in beliefs:
        from_agent = belief['from_agent']
        to_agent = belief['to_agent']
        confidence = belief['confidence']

        G.add_edge(from_agent, to_agent, weight=confidence)

    # 设置图形大小
    plt.figure(figsize=(16, 12))

    # 使用 spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # 绘制节点
    node_sizes = [G.degree(node) * 500 for node in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        alpha=0.8
    )

    # 绘制边（根据信心度设置颜色和宽度）
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(
        G, pos,
        width=[w * 3 for w in weights],
        alpha=0.5,
        edge_color=weights,
        edge_cmap=plt.cm.RdYlGn,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )

    # 绘制标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )

    plt.title("Agent Belief Network\n(Edge width/color = confidence)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
    else:
        plt.show()


def print_detailed_beliefs(beliefs: List[Dict], limit: int = 10):
    """打印详细的信念内容"""
    print("\n" + "=" * 80)
    print(f"DETAILED BELIEFS (showing top {limit})")
    print("=" * 80)

    # 按信心度排序
    sorted_beliefs = sorted(beliefs, key=lambda x: x['confidence'], reverse=True)

    for i, belief in enumerate(sorted_beliefs[:limit], 1):
        print(f"\n{i}. {belief['from_agent']} → {belief['to_agent']}")
        print(f"   Confidence: {belief['confidence']:.2f} (based on {belief['evidence_count']} interactions)")
        print(f"   Belief: {belief['content']}")
        print(f"   Last Updated: {belief['last_updated']}")


def compare_registries(old_path: Path, new_path: Path):
    """比较训练前后的信念变化"""
    print("\n" + "=" * 80)
    print("COMPARING REGISTRIES")
    print("=" * 80)

    old_data = load_registry(old_path)
    new_data = load_registry(new_path)

    old_beliefs = {(b['from_agent'], b['to_agent']): b for b in old_data['beliefs']}
    new_beliefs = {(b['from_agent'], b['to_agent']): b for b in new_data['beliefs']}

    # 找出变化
    added = set(new_beliefs.keys()) - set(old_beliefs.keys())
    removed = set(old_beliefs.keys()) - set(new_beliefs.keys())
    updated = set(new_beliefs.keys()) & set(old_beliefs.keys())

    print(f"\n📊 Changes:")
    print(f"  Added beliefs: {len(added)}")
    print(f"  Removed beliefs: {len(removed)}")
    print(f"  Updated beliefs: {len(updated)}")

    # 显示最大变化
    print(f"\n📈 Top 5 Confidence Increases:")
    changes = []
    for key in updated:
        old_conf = old_beliefs[key]['confidence']
        new_conf = new_beliefs[key]['confidence']
        delta = new_conf - old_conf
        if delta > 0:
            changes.append((key, delta, old_conf, new_conf))

    changes.sort(key=lambda x: x[1], reverse=True)
    for (from_a, to_a), delta, old_c, new_c in changes[:5]:
        print(f"  {from_a} → {to_a}: {old_c:.2f} → {new_c:.2f} (+{delta:.2f})")

    print(f"\n📉 Top 5 Confidence Decreases:")
    changes = []
    for key in updated:
        old_conf = old_beliefs[key]['confidence']
        new_conf = new_beliefs[key]['confidence']
        delta = new_conf - old_conf
        if delta < 0:
            changes.append((key, delta, old_conf, new_conf))

    changes.sort(key=lambda x: x[1])
    for (from_a, to_a), delta, old_c, new_c in changes[:5]:
        print(f"  {from_a} → {to_a}: {old_c:.2f} → {new_c:.2f} ({delta:.2f})")


def export_to_csv(beliefs: List[Dict], output_path: Path):
    """导出信念到 CSV"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'from_agent', 'to_agent', 'belief_type', 'content',
            'confidence', 'evidence_count', 'last_updated'
        ])
        writer.writeheader()
        writer.writerows(beliefs)

    print(f"✓ Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CoRe Belief System")
    parser.add_argument(
        '--registry',
        type=str,
        required=True,
        help='Path to mind_registry.json'
    )
    parser.add_argument(
        '--compare',
        type=str,
        help='Path to old registry for comparison'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for visualization (default: show)'
    )
    parser.add_argument(
        '--export_csv',
        type=str,
        help='Export beliefs to CSV'
    )
    parser.add_argument(
        '--no_viz',
        action='store_true',
        help='Skip visualization'
    )

    args = parser.parse_args()

    # 加载 registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"❌ Registry not found: {registry_path}")
        return

    data = load_registry(registry_path)
    beliefs = data['beliefs']

    print(f"\n✓ Loaded registry: {registry_path}")
    print(f"  Profiles: {len(data['profiles'])}")
    print(f"  Beliefs: {len(beliefs)}")

    # 分析统计
    analyze_beliefs(beliefs)

    # 详细信念
    print_detailed_beliefs(beliefs, limit=10)

    # 对比
    if args.compare:
        compare_registries(Path(args.compare), registry_path)

    # 可视化
    if not args.no_viz:
        output_path = Path(args.output) if args.output else None
        visualize_belief_network(beliefs, output_path)

    # 导出 CSV
    if args.export_csv:
        export_to_csv(beliefs, Path(args.export_csv))

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
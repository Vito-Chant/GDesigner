"""
Multi-LLM Voting Experiment Visualizer
基于Streamlit的实验结果可视化分析工具

Usage:
    streamlit run voting_visualizer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import asdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from experiments.run_multi_llm_voting import *

# 页面配置
st.set_page_config(
    page_title="Multi-LLM Voting Analyzer",
    page_icon="🗳️",
    layout="wide"
)


# ============================================================================
# 数据加载
# ============================================================================

@st.cache_data
def load_experiment_metadata(file_path: str):
    """加载实验元数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_experiment_json(file_path: str):
    """加载实验JSON汇总"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_experiments(data_dir: str) -> List[Dict[str, str]]:
    """查找所有实验文件"""
    experiments = []

    # 查找pickle文件
    pkl_files = glob.glob(os.path.join(data_dir, "metadata_*.pkl"))

    for pkl_path in pkl_files:
        exp_id = os.path.basename(pkl_path).replace("metadata_", "").replace(".pkl", "")
        json_path = os.path.join(data_dir, f"experiment_{exp_id}.json")

        experiments.append({
            "experiment_id": exp_id,
            "pickle_path": pkl_path,
            "json_path": json_path if os.path.exists(json_path) else None
        })

    return sorted(experiments, key=lambda x: x["experiment_id"], reverse=True)


# ============================================================================
# 可视化组件
# ============================================================================

def plot_scan_results(scan_results: List[Any]) -> go.Figure:
    """绘制扫描结果：LLM数量 vs 准确率"""
    if not scan_results:
        return None

    df = pd.DataFrame([
        {
            "num_agents": sr.num_agents if hasattr(sr, 'num_agents') else sr['num_agents'],
            "accuracy": sr.accuracy if hasattr(sr, 'accuracy') else sr['accuracy'],
            "unanimous_ratio": sr.unanimous_ratio if hasattr(sr, 'unanimous_ratio') else sr['unanimous_ratio'],
            "avg_agreement_ratio": sr.avg_agreement_ratio if hasattr(sr, 'avg_agreement_ratio') else sr[
                'avg_agreement_ratio']
        }
        for sr in scan_results
    ])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy vs Number of Agents", "Voting Consistency Metrics"),
        horizontal_spacing=0.1
    )

    # 准确率曲线
    fig.add_trace(
        go.Scatter(
            x=df['num_agents'],
            y=df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            marker=dict(size=10),
            line=dict(width=3, color='#2ecc71')
        ),
        row=1, col=1
    )

    # 添加趋势线
    z = np.polyfit(df['num_agents'], df['accuracy'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(df['num_agents'].min(), df['num_agents'].max(), 50)
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=p(x_smooth),
            mode='lines',
            name='Trend',
            line=dict(width=2, dash='dash', color='#95a5a6')
        ),
        row=1, col=1
    )

    # 一致性指标
    fig.add_trace(
        go.Scatter(
            x=df['num_agents'],
            y=df['unanimous_ratio'],
            mode='lines+markers',
            name='Unanimous Ratio',
            marker=dict(size=8),
            line=dict(width=2, color='#3498db')
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df['num_agents'],
            y=df['avg_agreement_ratio'],
            mode='lines+markers',
            name='Avg Agreement',
            marker=dict(size=8),
            line=dict(width=2, color='#e74c3c')
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Number of Agents", row=1, col=1)
    fig.update_xaxes(title_text="Number of Agents", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", tickformat='.1%', row=1, col=1)
    fig.update_yaxes(title_text="Ratio", tickformat='.1%', row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_agent_performance(question_records: List[Any]) -> go.Figure:
    """绘制各Agent的性能对比"""
    agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'llm_name': ''})

    for record in question_records:
        votes = record.agent_votes if hasattr(record, 'agent_votes') else record['agent_votes']
        for vote in votes:
            agent_id = vote.agent_id if hasattr(vote, 'agent_id') else vote['agent_id']
            is_correct = vote.is_correct if hasattr(vote, 'is_correct') else vote['is_correct']
            llm_name = vote.llm_name if hasattr(vote, 'llm_name') else vote['llm_name']

            agent_stats[agent_id]['total'] += 1
            agent_stats[agent_id]['llm_name'] = llm_name
            if is_correct:
                agent_stats[agent_id]['correct'] += 1

    df = pd.DataFrame([
        {
            'agent_id': agent_id,
            'llm_name': stats['llm_name'].split('/')[-1],
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'correct': stats['correct'],
            'total': stats['total']
        }
        for agent_id, stats in agent_stats.items()
    ]).sort_values('agent_id')

    fig = px.bar(
        df,
        x='agent_id',
        y='accuracy',
        color='llm_name',
        text=df['accuracy'].apply(lambda x: f'{x:.1%}'),
        title="Individual Agent Accuracy",
        labels={'accuracy': 'Accuracy', 'agent_id': 'Agent ID', 'llm_name': 'Model'}
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, yaxis_tickformat='.0%')

    return fig


def plot_vote_distribution(question_records: List[Any]) -> go.Figure:
    """绘制投票分布统计"""
    # 统计每道题的投票分布情况
    answer_counts = defaultdict(int)
    correct_when_unanimous = 0
    correct_when_split = 0
    total_unanimous = 0
    total_split = 0

    for record in question_records:
        is_unanimous = record.is_unanimous if hasattr(record, 'is_unanimous') else record['is_unanimous']
        is_correct = record.is_correct if hasattr(record, 'is_correct') else record['is_correct']
        vote_counts = record.raw_vote_counts if hasattr(record, 'raw_vote_counts') else record['raw_vote_counts']

        for answer, count in vote_counts.items():
            answer_counts[answer] += count

        if is_unanimous:
            total_unanimous += 1
            if is_correct:
                correct_when_unanimous += 1
        else:
            total_split += 1
            if is_correct:
                correct_when_split += 1

    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Vote Distribution", "Accuracy by Vote Agreement"),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )

    # 投票分布饼图
    fig.add_trace(
        go.Pie(
            labels=list(answer_counts.keys()),
            values=list(answer_counts.values()),
            hole=0.4,
            textinfo='label+percent',
            marker_colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ),
        row=1, col=1
    )

    # 一致性 vs 准确率
    categories = ['Unanimous Votes', 'Split Votes']
    accuracies = [
        correct_when_unanimous / total_unanimous if total_unanimous > 0 else 0,
        correct_when_split / total_split if total_split > 0 else 0
    ]
    counts = [total_unanimous, total_split]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=accuracies,
            text=[f'{a:.1%}<br>({c} questions)' for a, c in zip(accuracies, counts)],
            textposition='inside',
            marker_color=['#2ecc71', '#e74c3c']
        ),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="Accuracy", tickformat='.0%', row=1, col=2)

    return fig


def plot_entropy_distribution(question_records: List[Any]) -> go.Figure:
    """绘制熵分布和准确率关系"""
    data = []
    for record in question_records:
        entropy = record.entropy if hasattr(record, 'entropy') else record['entropy']
        is_correct = record.is_correct if hasattr(record, 'is_correct') else record['is_correct']
        agreement = record.agreement_ratio if hasattr(record, 'agreement_ratio') else record['agreement_ratio']

        data.append({
            'entropy': entropy,
            'is_correct': is_correct,
            'agreement_ratio': agreement,
            'result': 'Correct' if is_correct else 'Incorrect'
        })

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x='entropy',
        y='agreement_ratio',
        color='result',
        color_discrete_map={'Correct': '#2ecc71', 'Incorrect': '#e74c3c'},
        title="Entropy vs Agreement Ratio (colored by correctness)",
        labels={'entropy': 'Voting Entropy', 'agreement_ratio': 'Agreement Ratio'}
    )

    fig.update_layout(height=400)

    return fig


def display_question_details(question_records: List[Any], question_idx: int):
    """显示单道题的详细信息"""
    record = question_records[question_idx]

    # 基本信息
    question_id = record.question_id if hasattr(record, 'question_id') else record['question_id']
    question_text = record.question_text if hasattr(record, 'question_text') else record['question_text']
    correct_answer = record.correct_answer if hasattr(record, 'correct_answer') else record['correct_answer']
    final_answer = record.final_answer if hasattr(record, 'final_answer') else record['final_answer']
    is_correct = record.is_correct if hasattr(record, 'is_correct') else record['is_correct']

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Question {question_id}:**")
        st.text(question_text[:300] + "..." if len(question_text) > 300 else question_text)

    with col2:
        st.metric("Correct Answer", correct_answer)
        st.metric("Final Answer", final_answer,
                  delta="✓" if is_correct else "✗",
                  delta_color="normal" if is_correct else "inverse")

    with col3:
        entropy = record.entropy if hasattr(record, 'entropy') else record['entropy']
        agreement = record.agreement_ratio if hasattr(record, 'agreement_ratio') else record['agreement_ratio']
        st.metric("Entropy", f"{entropy:.3f}")
        st.metric("Agreement", f"{agreement:.1%}")

    # 投票详情
    st.markdown("---")
    st.markdown("**Agent Votes:**")

    votes = record.agent_votes if hasattr(record, 'agent_votes') else record['agent_votes']
    vote_df = pd.DataFrame([
        {
            'Agent': v.agent_id if hasattr(v, 'agent_id') else v['agent_id'],
            'Model': (v.llm_name if hasattr(v, 'llm_name') else v['llm_name']).split('/')[-1],
            'Answer': v.extracted_answer if hasattr(v, 'extracted_answer') else v['extracted_answer'],
            'Weight': v.weight if hasattr(v, 'weight') else v['weight'],
            'Correct': '✓' if (v.is_correct if hasattr(v, 'is_correct') else v['is_correct']) else '✗',
            'Time (s)': f"{(v.response_time if hasattr(v, 'response_time') else v['response_time']):.2f}"
        }
        for v in votes
    ])

    st.dataframe(vote_df, use_container_width=True, hide_index=True)

    # 投票分布可视化
    vote_dist = record.vote_distribution if hasattr(record, 'vote_distribution') else record['vote_distribution']
    if vote_dist:
        fig = px.pie(
            values=list(vote_dist.values()),
            names=list(vote_dist.keys()),
            title="Vote Distribution (Weighted)",
            hole=0.4
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 主应用
# ============================================================================

def main():
    st.title("🗳️ Multi-LLM Voting Experiment Analyzer")

    # 侧边栏 - 数据加载
    st.sidebar.header("📂 Data Selection")

    # 数据目录选择
    default_dir = "./result/multi_llm_voting_v2"
    data_dir = st.sidebar.text_input("Data Directory", value=default_dir)

    # 或者直接上传文件
    uploaded_file = st.sidebar.file_uploader("Or upload metadata file (.pkl)", type=['pkl'])

    metadata = None

    if uploaded_file:
        metadata = pickle.load(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    elif os.path.exists(data_dir):
        experiments = find_experiments(data_dir)

        if experiments:
            exp_options = [exp['experiment_id'] for exp in experiments]
            selected_exp = st.sidebar.selectbox("Select Experiment", exp_options)

            selected = next(e for e in experiments if e['experiment_id'] == selected_exp)

            if st.sidebar.button("Load Experiment"):
                metadata = load_experiment_metadata(selected['pickle_path'])
                st.sidebar.success(f"Loaded: {selected_exp}")
        else:
            st.sidebar.warning("No experiments found in directory")
    else:
        st.sidebar.warning("Directory not found")

    if metadata is None:
        st.info("👈 Please select or upload an experiment to analyze")

        # 显示使用说明
        st.markdown("""
        ## How to Use

        1. **Run an experiment** with the voting script:
        ```bash
        python run_multi_llm_voting_v2.py --homogeneous --llm_name "Qwen/Qwen3-4B" --num_agents 5 --scan_mode
        ```

        2. **Load the results** by:
           - Specifying the data directory in the sidebar
           - Or uploading the `.pkl` metadata file directly

        3. **Explore** the visualizations and analysis

        ## Features

        - 📈 **Scan Results**: See how accuracy changes with number of agents
        - 🤖 **Agent Performance**: Compare individual agent accuracy
        - 🗳️ **Vote Distribution**: Analyze voting patterns
        - 🔍 **Question Details**: Drill down into individual questions
        """)
        return

    # 主内容区域
    st.markdown("---")

    # 实验概览
    st.header("📊 Experiment Overview")

    col1, col2, col3, col4 = st.columns(4)

    records = metadata.question_records
    correct = sum(1 for r in records if (r.is_correct if hasattr(r, 'is_correct') else r['is_correct']))
    unanimous = sum(1 for r in records if (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous']))

    with col1:
        st.metric("Total Questions", len(records))
    with col2:
        st.metric("Accuracy", f"{correct / len(records):.1%}")
    with col3:
        st.metric("Unanimous Votes", f"{unanimous / len(records):.1%}")
    with col4:
        llm_configs = metadata.llm_configs
        st.metric("Agents", len(llm_configs))

    # LLM配置信息
    with st.expander("🔧 LLM Configuration"):
        config_df = pd.DataFrame([
            {"LLM": llm.split('/')[-1], "Weight": f"{weight:.3f}"}
            for llm, weight in llm_configs
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Scan Results",
        "🤖 Agent Performance",
        "🗳️ Vote Analysis",
        "🔍 Question Details"
    ])

    # Tab 1: 扫描结果
    with tab1:
        if metadata.scan_results:
            st.subheader("Accuracy vs Number of Agents")
            fig = plot_scan_results(metadata.scan_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # 扫描结果表格
            st.subheader("Detailed Scan Results")
            scan_df = pd.DataFrame([
                {
                    "# Agents": sr.num_agents if hasattr(sr, 'num_agents') else sr['num_agents'],
                    "Accuracy": f"{(sr.accuracy if hasattr(sr, 'accuracy') else sr['accuracy']):.2%}",
                    "Correct": sr.correct_count if hasattr(sr, 'correct_count') else sr['correct_count'],
                    "Unanimous": f"{(sr.unanimous_ratio if hasattr(sr, 'unanimous_ratio') else sr['unanimous_ratio']):.1%}",
                    "Avg Agreement": f"{(sr.avg_agreement_ratio if hasattr(sr, 'avg_agreement_ratio') else sr['avg_agreement_ratio']):.1%}"
                }
                for sr in metadata.scan_results
            ])
            st.dataframe(scan_df, use_container_width=True, hide_index=True)
        else:
            st.info("No scan results available. Run experiment with --scan_mode to enable.")

    # Tab 2: Agent性能
    with tab2:
        st.subheader("Individual Agent Performance")
        fig = plot_agent_performance(records)
        st.plotly_chart(fig, use_container_width=True)

        # Agent统计表
        agent_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'llm_name': '', 'total_time': 0})
        for record in records:
            votes = record.agent_votes if hasattr(record, 'agent_votes') else record['agent_votes']
            for vote in votes:
                agent_id = vote.agent_id if hasattr(vote, 'agent_id') else vote['agent_id']
                agent_stats[agent_id]['total'] += 1
                agent_stats[agent_id]['llm_name'] = \
                (vote.llm_name if hasattr(vote, 'llm_name') else vote['llm_name']).split('/')[-1]
                agent_stats[agent_id]['total_time'] += vote.response_time if hasattr(vote, 'response_time') else vote[
                    'response_time']
                if vote.is_correct if hasattr(vote, 'is_correct') else vote['is_correct']:
                    agent_stats[agent_id]['correct'] += 1

        stats_df = pd.DataFrame([
            {
                'Agent': aid,
                'Model': stats['llm_name'],
                'Accuracy': f"{stats['correct'] / stats['total']:.1%}",
                'Correct': stats['correct'],
                'Total': stats['total'],
                'Avg Time': f"{stats['total_time'] / stats['total']:.2f}s"
            }
            for aid, stats in sorted(agent_stats.items())
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Tab 3: 投票分析
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vote Distribution")
            fig = plot_vote_distribution(records)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Entropy vs Agreement")
            fig = plot_entropy_distribution(records)
            st.plotly_chart(fig, use_container_width=True)

        # 一致性分析
        st.subheader("Voting Pattern Analysis")

        # 按一致性程度分组
        unanimous_correct = sum(1 for r in records if
                                (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous']) and (
                                    r.is_correct if hasattr(r, 'is_correct') else r['is_correct']))
        unanimous_incorrect = sum(1 for r in records if
                                  (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous']) and not (
                                      r.is_correct if hasattr(r, 'is_correct') else r['is_correct']))
        split_correct = sum(1 for r in records if
                            not (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous']) and (
                                r.is_correct if hasattr(r, 'is_correct') else r['is_correct']))
        split_incorrect = sum(1 for r in records if
                              not (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous']) and not (
                                  r.is_correct if hasattr(r, 'is_correct') else r['is_correct']))

        pattern_df = pd.DataFrame({
            'Category': ['Unanimous & Correct', 'Unanimous & Incorrect', 'Split & Correct', 'Split & Incorrect'],
            'Count': [unanimous_correct, unanimous_incorrect, split_correct, split_incorrect],
            'Percentage': [
                unanimous_correct / len(records),
                unanimous_incorrect / len(records),
                split_correct / len(records),
                split_incorrect / len(records)
            ]
        })

        fig = px.bar(
            pattern_df,
            x='Category',
            y='Count',
            color='Category',
            color_discrete_sequence=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'],
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: 题目详情
    with tab4:
        st.subheader("Question Browser")

        # 过滤选项
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_correctness = st.selectbox(
                "Filter by Result",
                ["All", "Correct Only", "Incorrect Only"]
            )

        with col2:
            filter_agreement = st.selectbox(
                "Filter by Agreement",
                ["All", "Unanimous Only", "Split Only"]
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Question ID", "Entropy (High to Low)", "Entropy (Low to High)", "Agreement (High to Low)"]
            )

        # 过滤记录
        filtered_records = records.copy()

        if filter_correctness == "Correct Only":
            filtered_records = [r for r in filtered_records if
                                (r.is_correct if hasattr(r, 'is_correct') else r['is_correct'])]
        elif filter_correctness == "Incorrect Only":
            filtered_records = [r for r in filtered_records if
                                not (r.is_correct if hasattr(r, 'is_correct') else r['is_correct'])]

        if filter_agreement == "Unanimous Only":
            filtered_records = [r for r in filtered_records if
                                (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous'])]
        elif filter_agreement == "Split Only":
            filtered_records = [r for r in filtered_records if
                                not (r.is_unanimous if hasattr(r, 'is_unanimous') else r['is_unanimous'])]

        # 排序
        if sort_by == "Entropy (High to Low)":
            filtered_records = sorted(filtered_records,
                                      key=lambda r: r.entropy if hasattr(r, 'entropy') else r['entropy'], reverse=True)
        elif sort_by == "Entropy (Low to High)":
            filtered_records = sorted(filtered_records,
                                      key=lambda r: r.entropy if hasattr(r, 'entropy') else r['entropy'])
        elif sort_by == "Agreement (High to Low)":
            filtered_records = sorted(filtered_records,
                                      key=lambda r: r.agreement_ratio if hasattr(r, 'agreement_ratio') else r[
                                          'agreement_ratio'], reverse=True)

        st.info(f"Showing {len(filtered_records)} questions")

        # 问题选择器
        if filtered_records:
            question_options = [
                f"Q{r.question_id if hasattr(r, 'question_id') else r['question_id']}: " +
                f"{'✓' if (r.is_correct if hasattr(r, 'is_correct') else r['is_correct']) else '✗'} " +
                f"(Entropy: {(r.entropy if hasattr(r, 'entropy') else r['entropy']):.2f})"
                for r in filtered_records
            ]

            selected_q_idx = st.selectbox("Select Question", range(len(question_options)),
                                          format_func=lambda x: question_options[x])

            st.markdown("---")
            display_question_details(filtered_records, selected_q_idx)
        else:
            st.warning("No questions match the filters")

    # 页脚
    st.markdown("---")
    st.markdown(
        f"**Experiment ID:** {metadata.experiment_id} | "
        f"**Timestamp:** {metadata.timestamp} | "
        f"**Total Cost:** ${metadata.total_cost:.4f}"
    )


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# 尝试导入 UMAP，如果失败则回退到 PCA
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# === 配置 ===
DATA_DIR = "validation_data_v2"
st.set_page_config(page_title="MMLU Logprobs Visualization", layout="wide")

# === 1. 数据加载函数 (带缓存) ===
@st.cache_data
def load_data(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "batch_*.pkl"))
    data = []
    
    if not all_files:
        return pd.DataFrame(), []

    progress_bar = st.progress(0)
    for i, fpath in enumerate(all_files):
        with open(fpath, "rb") as f:
            batch = pickle.load(f)
            data.extend(batch)
        progress_bar.progress((i + 1) / len(all_files))
    progress_bar.empty()
    
    # 转换为 DataFrame 以便于处理 Metadata
    df = pd.DataFrame([
        {
            "sample_id": d["sample_id"],
            "role": d["role"],
            "is_correct": d["is_correct"],
            "seq_len": d["seq_len"],
            # 计算一些标量特征用于宏观分析
            "mean_gt_logprob": np.mean(d["gt_logprobs"]),
            "min_gt_logprob": np.min(d["gt_logprobs"]),
            "std_gt_logprob": np.std(d["gt_logprobs"]),
            "raw_data_idx": i # 记录原始列表索引方便检索矩阵
        }
        for i, d in enumerate(data)
    ])
    
    return df, data

# === 主程序开始 ===
st.title("🧠 LLM Logprobs Feature Visualization")

if not os.path.exists(DATA_DIR):
    st.error(f"Directory '{DATA_DIR}' not found. Please ensure the data directory exists.")
    st.stop()

df, raw_data_list = load_data(DATA_DIR)

if df.empty:
    st.warning("No data found in the directory.")
    st.stop()

# === 侧边栏过滤器 ===
st.sidebar.header("Filter Controls")

# 角色筛选
all_roles = df['role'].unique()
selected_roles = st.sidebar.multiselect(
    "Select Roles", 
    options=all_roles, 
    default=all_roles[:3] if len(all_roles) > 0 else None
)

# 过滤数据
filtered_df = df[df['role'].isin(selected_roles)]

if filtered_df.empty:
    st.warning("No data selected. Please select at least one role.")
    st.stop()

# === 关键修复：定义 Tabs ===
# 这里定义了 tab3，之前的报错就是因为这行代码缺失或被覆盖了
tab1, tab2, tab3 = st.tabs(["📊 Macro Statistics", "🌌 Embedding Space", "🔬 Micro Comparison"])

# ==========================================
# Tab 1: 宏观统计 (分布与相关性)
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confidence (Logprob) Distribution by Role")
        st.caption("Do specific roles make the model more confident about the input text?")
        fig_violin = px.box(
            filtered_df, 
            x="role", 
            y="mean_gt_logprob", 
            color="is_correct",
            points="outliers",
            title="Mean GT Logprob Distribution",
            hover_data=["sample_id"]
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col2:
        st.subheader("Accuracy by Role")
        # 计算准确率
        acc_df = filtered_df.groupby("role")["is_correct"].mean().reset_index()
        fig_bar = px.bar(
            acc_df, x="role", y="is_correct", 
            title="Win Rate (Accuracy)", 
            color="is_correct", 
            color_continuous_scale="Viridis",
            labels={"is_correct": "Accuracy"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Correlation: Seq Length vs. Confidence")
    fig_scatter = px.scatter(
        filtered_df,
        x="seq_len",
        y="mean_gt_logprob",
        color="role",
        symbol="is_correct",
        opacity=0.6,
        title="Sequence Length vs. Mean Logprob"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# Tab 2: 嵌入空间 (UMAP/PCA)
# ==========================================
with tab2:
    st.subheader("Feature Space Projection")
    st.caption("Clustering samples based on their logprob statistics (Mean, Min, Std, SeqLen).")
    
    feature_cols = ["mean_gt_logprob", "min_gt_logprob", "std_gt_logprob", "seq_len"]
    X = filtered_df[feature_cols].values
    
    col_opt, col_plot = st.columns([1, 4])
    
    with col_opt:
        method = st.radio("Projection Method", ["PCA", "UMAP"] if HAS_UMAP else ["PCA"])
    
    if method == "UMAP" and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(X)
        
    # 为了绘图不影响原 filtered_df，创建一个临时的 copy
    plot_df = filtered_df.copy()
    plot_df['x_emb'] = embedding[:, 0]
    plot_df['y_emb'] = embedding[:, 1]
    
    with col_plot:
        fig_emb = px.scatter(
            plot_df,
            x='x_emb', y='y_emb',
            color='role',
            symbol='is_correct',
            hover_data=['sample_id', 'mean_gt_logprob'],
            title=f"{method} Projection of Logprob Stats"
        )
        st.plotly_chart(fig_emb, use_container_width=True)

# ==========================================
# Tab 3: 微观矩阵视察 (增强版：Logprob + Entropy)
# ==========================================
with tab3:
    st.subheader("Micro-Level Inspection: Confidence vs. Uncertainty")
    
    # 1. 选择 Sample ID
    all_sample_ids = sorted(filtered_df['sample_id'].unique())
    if not all_sample_ids:
        st.warning("No samples found.")
    else:
        # 使用 selectbox 选择样本
        c_sel1, c_sel2 = st.columns([1, 3])
        with c_sel1:
            s_id = st.selectbox("Select Sample ID", all_sample_ids)
        
        # 获取该 Sample ID 下所有 Role 的数据
        sample_records = filtered_df[filtered_df['sample_id'] == s_id]
        
        # 显示题干预览 (如果有题干文本更好，这里只能显示长度)
        st.caption(f"Comparing {len(sample_records)} roles on Sample {s_id}. Sequence Length: {sample_records.iloc[0]['seq_len']}")
        
        st.markdown("---")
        
        # ========================================================
        # Chart A: Ground Truth Logprobs (Confidence)
        # ========================================================
        st.markdown("### 1. Confidence: GT Logprob Curves")
        st.caption("Higher is better (closer to 0). Low dips indicate where the model was surprised by the text.")
        
        fig_compare = go.Figure()
        
        # 遍历该 Sample 下的所有 Role 数据
        for idx, row in sample_records.iterrows():
            role_name = row['role']
            is_correct = row['is_correct']
            raw_entry = raw_data_list[row['raw_data_idx']]
            gt_vector = raw_entry['gt_logprobs']
            
            # 样式：做对是实线，做错是点线
            line_style = dict(dash='solid') if is_correct else dict(dash='dot')
            status_icon = "✓" if is_correct else "✗"
            
            fig_compare.add_trace(go.Scatter(
                y=gt_vector,
                mode='lines', 
                name=f"{role_name} ({status_icon})",
                line=line_style,
                opacity=0.8,
                hovertemplate=f"<b>{role_name}</b><br>Logprob: %{{y:.2f}}<extra></extra>"
            ))

        fig_compare.update_layout(
            yaxis_title="Log Probability (Confidence)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ========================================================
        # Chart B: Entropy (Uncertainty)
        # ========================================================
        st.markdown("### 2. Uncertainty: Token-wise Entropy")
        st.caption("Lower is better (Stable). High spikes mean the model is 'confused' and considering many alternatives.")
        
        fig_entropy = go.Figure()
        
        for idx, row in sample_records.iterrows():
            role_name = row['role']
            is_correct = row['is_correct']
            raw_entry = raw_data_list[row['raw_data_idx']]
            
            # 计算熵: H = - sum(p * logp)
            mat = raw_entry['features'] # shape: [Seq, 20]
            
            # 1. 转换回概率空间 (Approximation based on Top-20)
            probs = np.exp(mat) 
            
            # 2. 计算熵
            # 加上 1e-10 防止 log(0) 虽然 mat 本身就是 log
            # 直接用 p * logp 计算: probs * mat
            entropy_vec = -np.sum(probs * mat, axis=1)
            
            # 样式保持一致
            line_style = dict(dash='solid') if is_correct else dict(dash='dot')
            status_icon = "✓" if is_correct else "✗"
            
            fig_entropy.add_trace(go.Scatter(
                y=entropy_vec,
                mode='lines', 
                name=f"{role_name} ({status_icon})",
                line=line_style,
                opacity=0.8,
                hovertemplate=f"<b>{role_name}</b><br>Entropy: %{{y:.2f}}<extra></extra>"
            ))

        fig_entropy.update_layout(
            xaxis_title="Token Position",
            yaxis_title="Entropy (nats)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_entropy, use_container_width=True)

        # ========================================================
        # Chart C: Matrix Heatmap (Drill down)
        # ========================================================
        st.markdown("---")
        st.markdown("### 3. Deep Dive: Top-20 Distribution Heatmap")
        
        c1, c2 = st.columns([1, 3])
        with c1:
            roles_available = sample_records['role'].unique()
            s_role = st.selectbox("Select Role to inspect Matrix", roles_available)
        
        if s_role:
            record_row = sample_records[sample_records['role'] == s_role].iloc[0]
            raw_record = raw_data_list[record_row['raw_data_idx']]
            mat = raw_record['features'] # [Seq, 20]
            
            heatmap_data = np.maximum(mat, -20) # 截断极小值
            
            with c2:
                fig_heat = px.imshow(
                    heatmap_data,
                    labels=dict(x="Rank (0=Top1)", y="Token Position", color="Logprob"),
                    aspect="auto",
                    color_continuous_scale="Magma",
                    title=f"Logprobs Heatmap: {s_role}"
                )
                fig_heat.update_layout(height=500)
                st.plotly_chart(fig_heat, use_container_width=True)
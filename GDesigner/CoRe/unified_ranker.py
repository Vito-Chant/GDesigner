"""
CoRe Framework v4.1: Unified Ranker Module
System 1.5 (Reranker) + System 2 (LLM) 混合架构

关键设计：
1. Reranker用于冷启动和RAG检索（直接对文本打分，无需向量库）
2. LLM用于事后路由和见解生成（复杂推理和策略规划）

v4.2 更新：
- 移除 Confidence 字段
- 强制 CoT (先 REASONING 再 SELECTED)
- Insight 改为 SUGGESTION (建议性而非指令性)
- 路由决策时注入 Agent Input Context 和 Routing History
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import weave
from sentence_transformers import CrossEncoder
import numpy as np


@dataclass
class RoutingDecision:
    """路由决策结果 (v4.2: 移除 confidence)"""
    selected_agent: str
    reasoning: str  # **v4.2新增**: CoT推理过程
    path_used: str  # 'reranker_cold_start', 'llm_post_hoc'
    insight_instruction: Optional[str]  # **v4.2改名**: 原为insight，现为suggestion
    alternative_agents: List[Tuple[str, float]]
    cost_tokens: int


class UnifiedRanker:
    """
    统一的Ranker模块

    职责划分：
    - System 1.5 (Reranker):
      * 冷启动：从所有Agent中选择最适合的第一棒
      * RAG检索：从历史列表中检索相关上下文

    - System 2 (LLM):
      * 事后路由：基于当前Agent的私有视角决策下一棒
      * 见解生成：为下一个Agent生成战略性Suggestion
    """

    def __init__(
            self,
            llm,  # LLM实例（用于慢路径）
            reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        """
        初始化Unified Ranker

        Args:
            llm: LLM实例，用于复杂的路由决策和Insight生成
            reranker_model_name: Cross-Encoder模型名称
        """
        self.llm = llm

        # **关键修改**: 使用Cross-Encoder而非Bi-Encoder
        # Cross-Encoder直接对(Query, Document)对打分，精度更高
        print(f"Loading Cross-Encoder: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)

        # 统计信息
        self.stats = {
            'cold_start_count': 0,
            'rag_retrieval_count': 0,
            'post_hoc_route_count': 0,
            'total_tokens_used': 0
        }

    async def cold_start(
            self,
            task: str,
            profiles: Dict[str, str]  # agent_id -> profile_text
    ) -> str:
        """
        System 1.5: 冷启动 - 使用Reranker选择第一个Agent

        工作原理：
        1. 构造所有(Task, Profile)对
        2. Cross-Encoder计算每个Agent与Task的匹配度
        3. 返回得分最高的Agent

        Args:
            task: 任务描述
            profiles: 所有Agent的公开Profile字典

        Returns:
            最佳匹配的Agent ID
        """
        self.stats['cold_start_count'] += 1

        if not profiles:
            raise ValueError("No agent profiles provided for cold start")

        # 构造(Query, Document)对
        agent_ids = list(profiles.keys())
        pairs = [(task, profiles[agent_id]) for agent_id in agent_ids]

        # Reranker打分
        scores = self.reranker.predict(pairs)

        # 选择最高分
        best_idx = np.argmax(scores)
        selected_agent = agent_ids[best_idx]

        print(f"[Cold Start] Selected {selected_agent} with score {scores[best_idx]:.3f}")

        return selected_agent

    def retrieve(
            self,
            task: str,
            history_list: List[str],  # 纯文本历史列表
            top_k: int = 3
    ) -> str:
        """
        System 1.5: RAG检索 - 从历史中检索相关上下文

        工作原理：
        1. 构造所有(Task, HistoryItem)对
        2. Cross-Encoder计算每个历史条目与Task的相关性
        3. 返回Top-k个最相关的历史文本

        **关键优势**: 无需维护向量库，直接在纯文本列表上操作

        Args:
            task: 当前任务
            history_list: 历史输出列表（纯文本）
            top_k: 返回Top-k个最相关的历史

        Returns:
            拼接的相关历史文本
        """
        if not history_list:
            return ""

        self.stats['rag_retrieval_count'] += 1

        # 构造(Query, Document)对
        pairs = [(task, history_item) for history_item in history_list]

        # Reranker打分
        scores = self.reranker.predict(pairs)

        # 选择Top-k
        top_k = min(top_k, len(history_list))
        top_indices = np.argsort(scores)[-top_k:][::-1]  # 降序排列

        # 提取对应的文本
        retrieved_texts = [history_list[i] for i in top_indices]

        print(f"[RAG] Retrieved {top_k} items from {len(history_list)} history entries")

        # 用分隔符拼接
        return "\n\n---\n\n".join(retrieved_texts)

    @weave.op()
    async def route_llm(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context_from_registry: str,  # 来自MindRegistry的上下文（包含私有信念）
            agent_input_context: Dict,  # **v4.2新增**: Agent完整输入上下文
            routing_history: List[Dict]  # **v4.2新增**: 历史路由决策
    ) -> RoutingDecision:
        """
        System 2: 事后路由 + 见解生成 (v4.2增强)

        这是LLM的核心决策环节：
        1. 基于当前Agent的主观视角（私有信念）
        2. 分析Task和当前Output
        3. **考虑Agent的输入上下文和路径历史**
        4. 决定下一个最合适的Agent
        5. **生成Suggestion**：建议性指导（非强制指令）

        Args:
            task: 原始任务
            current_output: 当前Agent的输出
            current_agent_id: 当前Agent的ID（视角）
            candidate_agents: 候选Agent列表
            context_from_registry: MindRegistry提供的上下文（含私有信念）
            agent_input_context: 当前Agent的完整输入（含RAG/Insight）
            routing_history: 历史路由路径

        Returns:
            RoutingDecision包含：
            - selected_agent: 下一个Agent
            - reasoning: CoT推理过程
            - insight_instruction: 战略性建议
        """
        self.stats['post_hoc_route_count'] += 1

        # 构建Prompt (v4.2增强版)
        prompt = self._build_llm_route_prompt(
            task, current_output, current_agent_id,
            candidate_agents, context_from_registry,
            agent_input_context, routing_history
        )

        # 调用LLM
        messages = [
            {
                'role': 'system',
                'content': 'You are an expert at task delegation and strategic planning in multi-agent systems.'
            },
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)

        # 解析响应 (v4.2适配CoT格式)
        selected_agent, reasoning, suggestion = self._parse_llm_route_response(
            response, candidate_agents
        )

        # 估算token成本
        token_cost = len(prompt.split()) + len(response.split())
        self.stats['total_tokens_used'] += token_cost

        print(f"[LLM Route] {current_agent_id} -> {selected_agent}")
        print(f"[Reasoning] {reasoning[:100]}..." if len(reasoning) > 100 else f"[Reasoning] {reasoning}")

        return RoutingDecision(
            selected_agent=selected_agent,
            reasoning=reasoning,
            path_used='llm_post_hoc',
            insight_instruction=suggestion,
            alternative_agents=[(cand, 0.0) for cand in candidate_agents if cand != selected_agent],
            cost_tokens=token_cost
        )

    def _build_llm_route_prompt(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context: str,
            agent_input_context: Dict,
            routing_history: List[Dict]
    ) -> str:
        """
        构建LLM路由的Prompt (v4.2重构版)

        **关键改进**:
        1. 强制CoT：先REASONING后SELECTED
        2. Insight改为SUGGESTION（建议性）
        3. 注入Agent Input Context和Routing History
        4. 移除Confidence要求
        """

        # === 构建Agent Input Context描述 ===
        input_context_str = ""
        if 'retrieved_history' in agent_input_context and agent_input_context['retrieved_history']:
            input_context_str += f"\n**RAG Context Provided**: Yes\n"
        if 'insight' in agent_input_context and agent_input_context['insight']:
            input_context_str += f"**Previous Suggestion**: {agent_input_context['insight']}\n"

        # === 构建Routing History描述 ===
        history_str = ""
        if routing_history:
            history_str = "\n=== ROUTING PATH SO FAR ===\n"
            for i, decision in enumerate(routing_history[-3:], 1):  # 最近3步
                history_str += f"Step {i}: {decision.get('selected', 'Unknown')} "
                history_str += f"(Reasoning: {decision.get('reasoning', 'N/A')[:50]}...)\n"

        prompt = f"""You are {current_agent_id}, coordinating a multi-agent system to solve a complex task.

=== TASK ===
{task}

=== YOUR RECENT OUTPUT ===
{current_output}

{input_context_str}

{history_str}

=== YOUR PERSPECTIVE (Private Beliefs) ===
{context}

=== CANDIDATE AGENTS ===
{', '.join(candidate_agents)}

=== YOUR DECISION ===
Based on:
1. The task requirements
2. Your output and the context you received
3. The routing path taken so far
4. Your beliefs about each candidate's capabilities

Please decide:
1. **First, explain your reasoning** (analyze the situation)
2. **Then, select the MOST SUITABLE next agent**
3. **Finally, provide a SUGGESTION** (not a command, but advice)
   - What aspect should they consider?
   - What pitfalls might exist?
   - How can they build on your work?

**CRITICAL**: Respond in this EXACT format (REASONING must come first):

REASONING: <your step-by-step analysis of why this agent is best>
SELECTED: <agent_id>
SUGGESTION: <brief, actionable advice for the next agent - keep it under 50 words>
"""
        return prompt

    def _parse_llm_route_response(
            self,
            response: str,
            candidate_agents: List[str]
    ) -> Tuple[str, str, str]:
        """
        解析LLM路由响应 (v4.2适配CoT格式)

        期望格式：
        REASONING: ...
        SELECTED: agent_id
        SUGGESTION: ...

        Returns:
            (selected_agent, reasoning, suggestion)
        """

        lines = response.strip().split('\n')

        selected_agent = None
        reasoning = ""
        suggestion = ""

        for line in lines:
            line = line.strip()
            if line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('SELECTED:'):
                selected_agent = line.replace('SELECTED:', '').strip()
            elif line.startswith('SUGGESTION:'):
                suggestion = line.replace('SUGGESTION:', '').strip()

        # 容错处理
        if not selected_agent or selected_agent not in candidate_agents:
            print(f"[Warning] Invalid selection '{selected_agent}', using first candidate")
            selected_agent = candidate_agents[0]
            reasoning = reasoning or "Failed to parse LLM response, using first candidate as fallback"
            suggestion = "Please continue the task using your expertise"

        if not suggestion:
            suggestion = "Build on the previous work and focus on quality"

        if not reasoning:
            reasoning = "Selection based on agent capabilities"

        return selected_agent, reasoning, suggestion

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_ops = (self.stats['cold_start_count'] +
                     self.stats['rag_retrieval_count'] +
                     self.stats['post_hoc_route_count'])

        return {
            **self.stats,
            'total_operations': total_ops,
            'avg_tokens_per_llm_route': (
                self.stats['total_tokens_used'] / self.stats['post_hoc_route_count']
                if self.stats['post_hoc_route_count'] > 0 else 0
            )
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import asyncio

    # Mock LLM用于测试
    class MockLLM:
        async def agen(self, messages):
            return """REASONING: The current mathematical analysis is complete and requires translation into executable code. The code expert has proven track record in numerical implementations and can handle edge cases like division by zero with proper validation.
SELECTED: code_expert_1
SUGGESTION: Focus on input validation before applying the formula. Consider edge cases where a=0. Return None or raise a clear exception for invalid inputs."""

    async def test_unified_ranker():
        print("="*70)
        print("CoRe v4.2 - Unified Ranker Test Suite")
        print("="*70)

        llm = MockLLM()
        ranker = UnifiedRanker(
            llm=llm,
            reranker_model_name="BAAI/bge-reranker-v2-m3"
        )

        # ===== 测试: LLM路由 (v4.2) =====
        print("\n[TEST] LLM Post-hoc Routing (v4.2)")
        print("-"*70)

        context_from_registry = """
Your beliefs about candidates:

About code_expert_1:
- Excellent at translating mathematical concepts into clean, executable code (confidence: 0.85)
- Has successfully handled similar numerical computation tasks before (evidence: 5 interactions)

About math_solver_2:
- Strong in theoretical proofs but less experienced in practical implementation (confidence: 0.60)
"""

        agent_input_context = {
            'task': "Implement the quadratic formula",
            'retrieved_history': "Previous analysis showed discriminant = 49",
            'insight': "Focus on numerical stability"
        }

        routing_history = [
            {'selected': 'math_solver_1', 'reasoning': 'Initial analysis needed'}
        ]

        decision = await ranker.route_llm(
            task="Implement the quadratic formula to solve ax^2 + bx + c = 0",
            current_output="I've completed the mathematical analysis: discriminant is 49, two real solutions expected",
            current_agent_id="math_solver_1",
            candidate_agents=["code_expert_1", "math_solver_2", "decision_maker"],
            context_from_registry=context_from_registry,
            agent_input_context=agent_input_context,
            routing_history=routing_history
        )

        print(f"✓ Selected Agent: {decision.selected_agent}")
        print(f"✓ Reasoning: {decision.reasoning[:100]}...")
        print(f"✓ Suggestion: {decision.insight_instruction}")
        print(f"✓ Token Cost: {decision.cost_tokens}")

        # ===== 统计信息 =====
        print("\n[STATISTICS]")
        print("-"*70)
        stats = ranker.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\n" + "="*70)
        print("All tests completed successfully! ✓")
        print("="*70)

    # 运行测试
    asyncio.run(test_unified_ranker())
"""
CoRe Framework v4.1: Unified Ranker Module
System 1.5 (Reranker) + System 2 (LLM) 混合架构

关键设计：
1. Reranker用于冷启动和RAG检索（直接对文本打分，无需向量库）
2. LLM用于事后路由和见解生成（复杂推理和策略规划）
"""

import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import weave
from sentence_transformers import CrossEncoder
import numpy as np


@dataclass
class RoutingDecision:
    """路由决策结果"""
    selected_agent: str
    reasoning: str
    confidence: float
    path_used: str  # 'reranker_cold_start', 'llm_post_hoc'
    insight_instruction: Optional[str]  # **v4.1新增**：策略指令
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
      * 见解生成：为下一个Agent生成战略性Insight指令
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
            context_from_registry: str  # 来自MindRegistry的上下文（包含私有信念）
    ) -> RoutingDecision:
        """
        System 2: 事后路由 + 见解生成

        这是LLM的核心决策环节：
        1. 基于当前Agent的主观视角（私有信念）
        2. 分析Task和当前Output
        3. 决定下一个最合适的Agent
        4. **生成Insight指令**：告诉下一个Agent应该重点关注什么

        Args:
            task: 原始任务
            current_output: 当前Agent的输出
            current_agent_id: 当前Agent的ID（视角）
            candidate_agents: 候选Agent列表
            context_from_registry: MindRegistry提供的上下文（含私有信念）

        Returns:
            RoutingDecision包含：
            - selected_agent: 下一个Agent
            - insight_instruction: 战略性指导
            - reasoning: 决策理由
        """
        self.stats['post_hoc_route_count'] += 1

        # 构建Prompt
        prompt = self._build_llm_route_prompt(
            task, current_output, current_agent_id,
            candidate_agents, context_from_registry
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

        # 解析响应
        selected_agent, reasoning, confidence, insight = self._parse_llm_route_response(
            response, candidate_agents
        )

        # 估算token成本
        token_cost = len(prompt.split()) + len(response.split())
        self.stats['total_tokens_used'] += token_cost

        print(f"[LLM Route] {current_agent_id} -> {selected_agent}")
        # print(f"[Insight] {insight[:100]}..." if len(insight) > 100 else f"[Insight] {insight}")

        return RoutingDecision(
            selected_agent=selected_agent,
            reasoning=reasoning,
            confidence=confidence,
            path_used='llm_post_hoc',
            insight_instruction=insight,
            alternative_agents=[(cand, 0.0) for cand in candidate_agents if cand != selected_agent],
            cost_tokens=token_cost
        )

    def _build_llm_route_prompt(
            self,
            task: str,
            current_output: str,
            current_agent_id: str,
            candidate_agents: List[str],
            context: str
    ) -> str:
        """
        构建LLM路由的Prompt

        Prompt结构：
        1. 角色设定：你是current_agent
        2. 任务上下文：Task + 你的Output
        3. 私有信念：你对候选者的看法（来自MindRegistry）
        4. 指令：选择下一个Agent + 生成Insight
        """

        prompt = f"""You are {current_agent_id}, coordinating a multi-agent system to solve a complex task.

=== TASK ===
{task}

=== YOUR RECENT OUTPUT ===
{current_output}

=== YOUR PERSPECTIVE (Private Beliefs) ===
{context}

=== CANDIDATE AGENTS ===
{', '.join(candidate_agents)}

=== YOUR DECISION ===
Based on:
1. The task requirements
2. Your output so far
3. Your beliefs about each candidate's capabilities

Please:
1. Select the MOST SUITABLE next agent
2. Provide a STRATEGIC INSIGHT for them
   - What specific aspect should they focus on?
   - What pitfalls should they avoid?
   - How can they best leverage your work?

Respond in this exact format:
SELECTED: <agent_id>
REASONING: <your detailed reasoning about why this agent is the best choice>
INSIGHT: <specific strategic instruction for the next agent, be concrete and actionable>
CONFIDENCE: <0.0-1.0>
"""
        return prompt

    def _parse_llm_route_response(
            self,
            response: str,
            candidate_agents: List[str]
    ) -> Tuple[str, str, float, str]:
        """
        解析LLM路由响应

        期望格式：
        SELECTED: agent_id
        REASONING: ...
        INSIGHT: ...
        CONFIDENCE: 0.0-1.0

        Returns:
            (selected_agent, reasoning, confidence, insight)
        """

        lines = response.strip().split('\n')

        selected_agent = None
        reasoning = ""
        insight = ""
        confidence = 0.5

        for line in lines:
            line = line.strip()
            if line.startswith('SELECTED:'):
                selected_agent = line.replace('SELECTED:', '').strip()
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
            elif line.startswith('INSIGHT:'):
                insight = line.replace('INSIGHT:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    confidence = max(0.0, min(1.0, confidence))  # 限制范围
                except:
                    confidence = 0.5

        # 容错处理
        if not selected_agent or selected_agent not in candidate_agents:
            print(f"[Warning] Invalid selection '{selected_agent}', using first candidate")
            selected_agent = candidate_agents[0]
            reasoning = "Failed to parse LLM response, using first candidate as fallback"
            insight = "Please continue the task based on previous context"

        if not insight:
            insight = "Continue with the task using your expertise"

        return selected_agent, reasoning, confidence, insight

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
            return """SELECTED: code_expert_1
REASONING: The task requires implementing a mathematical formula in code. While the math analysis is complete, we need someone who can translate this into robust, executable code. Code expert has proven track record in numerical implementations.
INSIGHT: Focus on edge case handling, especially division by zero when a=0. Implement input validation before applying the formula. Consider returning None or raising a custom exception for invalid inputs.
CONFIDENCE: 0.85"""

    async def test_unified_ranker():
        print("="*70)
        print("CoRe v4.1 - Unified Ranker Test Suite")
        print("="*70)

        llm = MockLLM()
        ranker = UnifiedRanker(
            llm=llm,
            reranker_model_name="BAAI/bge-reranker-v2-m3"
        )

        # ===== 测试1: 冷启动 =====
        print("\n[TEST 1] Cold Start")
        print("-"*70)

        profiles = {
            "math_solver_1": "Math expert specializing in algebraic problem solving and equation analysis. Strong in theoretical mathematics.",
            "code_expert_1": "Programming expert focused on numerical algorithms and scientific computing. Proficient in Python and algorithm implementation.",
            "analyst_1": "Strategic analyst good at breaking down complex problems and planning solution approaches."
        }

        best_agent = await ranker.cold_start(
            task="Implement the quadratic formula to solve ax^2 + bx + c = 0",
            profiles=profiles
        )
        print(f"✓ Cold Start Selected: {best_agent}")

        # ===== 测试2: RAG检索 =====
        print("\n[TEST 2] RAG Retrieval")
        print("-"*70)

        history = [
            "Step 1: Analyzed the problem structure. This is a standard quadratic equation.",
            "Step 2: Identified coefficients: a=2, b=5, c=-3",
            "Step 3: Applied discriminant formula: b²-4ac = 25 - 4(2)(-3) = 49",
            "Step 4: Since discriminant > 0, there are two real solutions",
            "Step 5: Prepared implementation plan: use quadratic formula x = (-b ± √Δ) / (2a)"
        ]

        retrieved = ranker.retrieve(
            task="Implement the quadratic formula in Python",
            history_list=history,
            top_k=3
        )
        print(f"✓ Retrieved Context:\n{retrieved[:200]}...")

        # ===== 测试3: LLM路由 =====
        print("\n[TEST 3] LLM Post-hoc Routing")
        print("-"*70)

        context_from_registry = """
Your beliefs about candidates:

About code_expert_1:
- Excellent at translating mathematical concepts into clean, executable code (confidence: 0.85)
- Has successfully handled similar numerical computation tasks before (evidence: 5 interactions)
- Sometimes overlooks edge cases without explicit reminders (confidence: 0.70)

About math_solver_2:
- Strong in theoretical proofs but less experienced in practical implementation (confidence: 0.60)
- Good collaborator but may not be the best choice for coding tasks (evidence: 3 interactions)
"""

        decision = await ranker.route_llm(
            task="Implement the quadratic formula to solve ax^2 + bx + c = 0",
            current_output="I've completed the mathematical analysis: discriminant is 49, two real solutions expected at x = (-5 ± 7) / 4",
            current_agent_id="math_solver_1",
            candidate_agents=["code_expert_1", "math_solver_2", "decision_maker"],
            context_from_registry=context_from_registry
        )

        print(f"✓ Selected Agent: {decision.selected_agent}")
        print(f"✓ Reasoning: {decision.reasoning[:100]}...")
        print(f"✓ Insight Instruction: {decision.insight_instruction[:100]}...")
        print(f"✓ Confidence: {decision.confidence:.2f}")
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
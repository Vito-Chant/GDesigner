"""
CoRe Framework v4.2: Belief Evolution Module
重点：分析路由决策质量，仅更新私有信念

v4.2 更新:
- insight_instruction 改为 suggestion (命名统一)
- 适配移除 confidence 的路由决策格式
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime


@dataclass
class InteractionTrace:
    """完整的Agent交互记录 (v4.2更新)"""
    from_agent: str
    to_agent: str
    task: str
    suggestion: str  # **v4.2修改**: 从insight_instruction改为suggestion
    output: str
    success: bool
    failure_reason: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BeliefUpdate:
    """
    信念更新操作 (v4.4 - Success Rate 版)
    """
    from_agent: str
    to_agent: str
    old_belief: Optional[str]
    new_belief: str
    update_reason: str

    # ✅ 新增：基于统计的更新
    success_delta: int  # 成功次数变化 (+1 成功, 0 失败)
    total_delta: int  # 总次数变化 (通常是 +1)

    # 兼容性字段
    @property
    def confidence_change(self) -> float:
        """兼容性属性：估算置信度变化"""
        if self.total_delta == 0:
            return 0.0

        # 如果是成功，正向；失败，负向
        if self.success_delta > 0:
            return 0.1  # 成功时提升
        else:
            return -0.1  # 失败时降低


class BeliefEvolver:
    """
    在线信念进化

    **v4.2关键修改**:
    - 分析对象：Routing Decision (不再依赖 Confidence)
    - 评估重点：Sender的Suggestion是否有效？
    - 更新范围：严格限制为私有信念（Sender -> Receiver）
    """

    def __init__(self, llm, mind_registry):
        self.llm = llm
        self.mind_registry = mind_registry
        self.update_history = []

    async def evolve_beliefs_from_interaction(
            self,
            interaction_trace: InteractionTrace,
            full_chain: List[InteractionTrace],
            task_success: bool,
            critic_feedback: Optional[str] = None
    ) -> List[BeliefUpdate]:
        """
        基于交互结果更新信念

        **v4.2核心逻辑**:
        分析Sender的Suggestion质量：
        - 是否有效指导了Receiver？
        - 还是Receiver能力确实不行？
        """

        if task_success and interaction_trace.success:
            # 正向强化
            updates = await self._positive_reinforcement(
                interaction_trace, critic_feedback
            )
        elif not interaction_trace.success:
            # 失败归因
            updates = await self._failure_attribution(
                interaction_trace, full_chain, critic_feedback
            )
        else:
            # 部分成功 - 细微更新
            updates = await self._nuanced_update(
                interaction_trace, task_success, critic_feedback
            )

        # 应用更新
        for update in updates:
            self._apply_belief_update(update)

        self.update_history.extend(updates)
        return updates

    async def _positive_reinforcement(
            self,
            trace: InteractionTrace,
            feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """当交互成功时强化正向信念 (v4.4 版)"""

        existing_beliefs = self.mind_registry.get_beliefs_about(
            to_agent=trace.to_agent,
            from_agent=trace.from_agent
        )

        prompt = f"""An agent interaction was SUCCESSFUL:

    From: {trace.from_agent}
    To: {trace.to_agent}
    Task: {trace.task}
    Suggestion Given: {trace.suggestion}
    Outcome: SUCCESS

    Existing beliefs: {[b.content for b in existing_beliefs] if existing_beliefs else 'None'}

    Generate a SHORT belief statement that {trace.from_agent} should hold about {trace.to_agent}.
    Focus on their capability and how well they responded to the strategic guidance.
    Keep it under 100 characters.

    Format: BELIEF: <statement>"""

        if feedback:
            prompt += f"\n\nAdditional feedback: {feedback}"

        messages = [
            {'role': 'system', 'content': 'You generate concise belief statements.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)
        new_belief = self._extract_belief_statement(response)

        # ✅ 新逻辑：记录成功
        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=existing_beliefs[0].content if existing_beliefs else None,
            new_belief=new_belief,
            update_reason="Positive reinforcement from successful interaction",
            success_delta=1,  # 成功 +1
            total_delta=1  # 总数 +1
        )

        return [update]

    async def _failure_attribution(
            self,
            trace: InteractionTrace,
            full_chain: List[InteractionTrace],
            feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """失败归因 (v4.4 版)"""

        # ... 现有的 prompt 构建逻辑 ...

        response = await self.llm.agen(messages)
        root_cause, belief_update, strategy_change = self._parse_failure_analysis(response)

        # ✅ 新逻辑：记录失败
        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=None,
            new_belief=belief_update,
            update_reason=f"Failure attribution: {root_cause}. Strategy: {strategy_change}",
            success_delta=0,  # 失败 +0
            total_delta=1  # 总数 +1
        )

        return [update]

    async def _nuanced_update(
            self,
            trace: InteractionTrace,
            task_success: bool,
            feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """处理局部成功但全局失败（或反之）的情况 (v4.4 版)"""

        # ... 现有的 prompt 构建逻辑 ...

        response = await self.llm.agen(messages)
        belief = self._extract_belief_statement(response)

        # ✅ 新逻辑：根据局部和全局结果决定
        # 局部成功 + 全局失败 = 部分成功 (0.5)
        # 局部失败 + 全局成功 = 不计入统计
        if trace.success and not task_success:
            success_delta = 0  # 虽然局部成功，但全局失败
        elif not trace.success and task_success:
            success_delta = 0  # 局部失败不计入成功
        else:
            success_delta = 1 if trace.success else 0

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=None,
            new_belief=belief,
            update_reason="Nuanced update from mixed outcomes",
            success_delta=success_delta,
            total_delta=1
        )

        return [update]

    def _apply_belief_update(self, update: BeliefUpdate):
        """将信念更新应用到Mind Registry (v4.4 版)"""

        from GDesigner.CoRe.mind_registry import RelationalBelief

        existing = self.mind_registry.get_beliefs_about(
            to_agent=update.to_agent,
            from_agent=update.from_agent
        )

        if existing:
            # ✅ 更新现有信念：累加统计
            old_belief = existing[0]
            new_success_count = old_belief.success_count + update.success_delta
            new_total_count = old_belief.total_count + update.total_delta
            evidence_count = old_belief.evidence_count + 1
        else:
            # ✅ 创建新信念：初始统计
            new_success_count = update.success_delta
            new_total_count = update.total_delta
            evidence_count = 1

        new_belief = RelationalBelief(
            from_agent=update.from_agent,
            to_agent=update.to_agent,
            belief_type='capability_assessment',
            content=update.new_belief,
            success_count=new_success_count,
            total_count=new_total_count,
            evidence_count=evidence_count,
            last_updated=datetime.now().isoformat()
        )

        self.mind_registry.add_belief(new_belief)

    def _extract_belief_statement(self, response: str) -> str:
        """从LLM响应中提取信念陈述"""
        for line in response.split('\n'):
            if line.strip().startswith('BELIEF:'):
                return line.replace('BELIEF:', '').strip()

        for line in response.split('\n'):
            if line.strip():
                return line.strip()

        return "Capability assessment updated based on interaction"

    def _parse_failure_analysis(self, response: str) -> Tuple[str, str, str]:
        """解析失败分析响应 (v4.2: 移除confidence返回)"""

        root_cause = "unknown"
        belief_update = "Performance below expectations"
        strategy_change = "Consider alternative agents or clearer suggestions"

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('ROOT_CAUSE:'):
                root_cause = line.replace('ROOT_CAUSE:', '').strip()
            elif line.startswith('BELIEF_UPDATE:'):
                belief_update = line.replace('BELIEF_UPDATE:', '').strip()
            elif line.startswith('STRATEGY_CHANGE:'):
                strategy_change = line.replace('STRATEGY_CHANGE:', '').strip()

        return root_cause, belief_update, strategy_change

    def get_evolution_summary(self) -> Dict:
        """获取信念进化摘要"""

        return {
            'total_updates': len(self.update_history),
            'positive_updates': sum(1 for u in self.update_history if u.confidence_change > 0),
            'negative_updates': sum(1 for u in self.update_history if u.confidence_change < 0),
            'avg_confidence_change': sum(u.confidence_change for u in self.update_history) / len(
                self.update_history) if self.update_history else 0
        }


# 使用示例
if __name__ == "__main__":
    from mind_registry import MindRegistry, AgentProfile


    class MockLLM:
        async def agen(self, messages):
            content = messages[-1]['content']
            if 'SUCCESSFUL' in content:
                return "BELIEF: Excellent at following strategic guidance for implementation tasks"
            else:
                return """ROOT_CAUSE: agent_capability
BELIEF_UPDATE: Struggles with edge cases despite clear suggestions
STRATEGY_CHANGE: Provide more detailed examples in future suggestions"""


    async def test_evolution():
        llm = MockLLM()
        registry = MindRegistry()

        registry.register_agent(AgentProfile(
            agent_id="code_expert_1",
            role="Code Expert",
            capabilities=["python", "algorithms"],
            specializations=["numerical"],
            limitations=[],
            description="Programming expert"
        ))

        evolver = BeliefEvolver(llm, registry)

        # 测试成功案例
        trace = InteractionTrace(
            from_agent="math_solver_1",
            to_agent="code_expert_1",
            task="Implement quadratic formula",
            suggestion="Focus on handling edge cases like division by zero",
            output="def solve(a,b,c): return (-b)/(2*a) if a != 0 else None",
            success=True
        )

        updates = await evolver.evolve_beliefs_from_interaction(
            trace, [trace], task_success=True
        )

        print("=== Positive Reinforcement ===")
        for update in updates:
            print(f"New Belief: {update.new_belief}")
            print(f"Confidence Change: {update.confidence_change:+.2f}")

        # 测试失败案例
        trace2 = InteractionTrace(
            from_agent="math_solver_1",
            to_agent="code_expert_1",
            task="Handle division by zero",
            suggestion="Add comprehensive error handling",
            output="def solve(a,b,c): return (-b)/(2*a)  # Still no validation!",
            success=False,
            failure_reason="Division by zero when a=0"
        )

        updates2 = await evolver.evolve_beliefs_from_interaction(
            trace2, [trace, trace2], task_success=False
        )

        print("\n=== Failure Attribution ===")
        for update in updates2:
            print(f"New Belief: {update.new_belief}")
            print(f"Reason: {update.update_reason}")

        print("\n=== Evolution Summary ===")
        print(evolver.get_evolution_summary())


    asyncio.run(test_evolution())

"""
CoRe Framework v4.1: Belief Evolution Module
重点：分析路由决策质量，仅更新私有信念
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime


@dataclass
class InteractionTrace:
    """完整的Agent交互记录"""
    from_agent: str
    to_agent: str
    task: str
    insight_instruction: str  # **修改**: 从handoff_note改为insight
    output: str
    success: bool
    failure_reason: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BeliefUpdate:
    """信念更新操作"""
    from_agent: str
    to_agent: str
    old_belief: Optional[str]
    new_belief: str
    update_reason: str
    confidence_change: float


class BeliefEvolver:
    """
    在线信念进化

    **v4.1关键修改**:
    - 分析对象：从Handoff Note改为Routing Decision
    - 评估重点：Sender给出的Insight指令是否误导了Receiver？
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

        **v4.1核心逻辑**:
        分析Sender的Insight指令质量：
        - 是否误导了Receiver？
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
        """当交互成功时强化正向信念"""

        existing_beliefs = self.mind_registry.get_beliefs_about(
            to_agent=trace.to_agent,
            from_agent=trace.from_agent
        )

        prompt = f"""An agent interaction was SUCCESSFUL:

From: {trace.from_agent}
To: {trace.to_agent}
Task: {trace.task}
Insight Instruction Given: {trace.insight_instruction}
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

        old_confidence = existing_beliefs[0].confidence if existing_beliefs else 0.5
        new_confidence = min(old_confidence + 0.1, 0.95)

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=existing_beliefs[0].content if existing_beliefs else None,
            new_belief=new_belief,
            update_reason="Positive reinforcement from successful interaction",
            confidence_change=new_confidence - old_confidence
        )

        return [update]

    async def _failure_attribution(
            self,
            trace: InteractionTrace,
            full_chain: List[InteractionTrace],
            feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """
        失败归因

        **v4.1关键问题**:
        是Sender的Insight指令有问题？还是Receiver能力不足？
        """

        prompt = f"""An agent handoff FAILED. Analyze the ROOT CAUSE:

FAILED INTERACTION:
From: {trace.from_agent} → To: {trace.to_agent}
Task: {trace.task}
Insight Instruction: {trace.insight_instruction}
Output: {trace.output}
Failure Reason: {trace.failure_reason}

FULL CHAIN (for context):
"""
        for i, step in enumerate(full_chain, 1):
            prompt += f"{i}. {step.from_agent}→{step.to_agent}: {'✓' if step.success else '✗'}\n"

        if feedback:
            prompt += f"\nCritic Feedback: {feedback}"

        prompt += """

CRITICAL ANALYSIS TASK:
1. Was the Insight Instruction from {from_agent} MISLEADING or INAPPROPRIATE?
   - Did it guide {to_agent} in the wrong direction?
   - Was it too vague or too specific?

2. OR was {to_agent} simply INCAPABLE of executing the task?
   - Did they lack the necessary skills?
   - Did they make obvious mistakes unrelated to the instruction?

3. What belief should {from_agent} update about {to_agent}?

Respond in this format:
ROOT_CAUSE: <misleading_instruction|agent_capability|both>
BELIEF_UPDATE: <new belief statement about {to_agent}'s capabilities>
STRATEGY_CHANGE: <what should change in future routing and instruction-giving>
CONFIDENCE: <0.0-1.0>
"""

        messages = [
            {'role': 'system', 'content': 'You are a failure analyst for multi-agent systems.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)

        root_cause, belief_update, strategy_change, confidence = self._parse_failure_analysis(response)

        updates = []

        # **严格限制**: 仅更新 Sender -> Receiver 的私有信念
        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=None,
            new_belief=belief_update,
            update_reason=f"Failure attribution: {root_cause}. Strategy: {strategy_change}",
            confidence_change=-0.2
        )
        updates.append(update)

        return updates

    async def _nuanced_update(
            self,
            trace: InteractionTrace,
            task_success: bool,
            feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """处理局部成功但全局失败（或反之）的情况"""

        prompt = f"""Mixed outcome analysis:

This step: {trace.from_agent} → {trace.to_agent}
Local success: {trace.success}
Global task success: {task_success}
Insight Given: {trace.insight_instruction}

Generate a NUANCED belief update that captures the complexity.
For example: "Good at X when given specific guidance, but struggles with Y"

Format: BELIEF: <nuanced statement>"""

        messages = [
            {'role': 'system', 'content': 'You create nuanced belief statements.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)
        belief = self._extract_belief_statement(response)

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=None,
            new_belief=belief,
            update_reason="Nuanced update from mixed outcomes",
            confidence_change=0.0
        )

        return [update]

    def _apply_belief_update(self, update: BeliefUpdate):
        """将信念更新应用到Mind Registry"""

        from mind_registry import RelationalBelief

        existing = self.mind_registry.get_beliefs_about(
            to_agent=update.to_agent,
            from_agent=update.from_agent
        )

        if existing:
            old_conf = existing[0].confidence
            evidence_count = existing[0].evidence_count + 1
        else:
            old_conf = 0.5
            evidence_count = 1

        new_conf = max(0.1, min(0.95, old_conf + update.confidence_change))

        new_belief = RelationalBelief(
            from_agent=update.from_agent,
            to_agent=update.to_agent,
            belief_type='capability_assessment',
            content=update.new_belief,
            confidence=new_conf,
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

    def _parse_failure_analysis(self, response: str) -> Tuple[str, str, str, float]:
        """解析失败分析响应"""

        root_cause = "unknown"
        belief_update = "Performance below expectations"
        strategy_change = "Consider alternative agents or clearer instructions"
        confidence = 0.3

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('ROOT_CAUSE:'):
                root_cause = line.replace('ROOT_CAUSE:', '').strip()
            elif line.startswith('BELIEF_UPDATE:'):
                belief_update = line.replace('BELIEF_UPDATE:', '').strip()
            elif line.startswith('STRATEGY_CHANGE:'):
                strategy_change = line.replace('STRATEGY_CHANGE:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    pass

        return root_cause, belief_update, strategy_change, confidence

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
BELIEF_UPDATE: Struggles with edge cases despite clear instructions
STRATEGY_CHANGE: Provide more detailed examples in future insights
CONFIDENCE: 0.4"""

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
            insight_instruction="Focus on handling edge cases like division by zero",
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
            insight_instruction="Add comprehensive error handling",
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
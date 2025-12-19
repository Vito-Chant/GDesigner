"""
泛化的信念进化器 (Generalized Belief Evolver)
"""

import asyncio
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import math

from GDesigner.CoRe.capability_taxonomy import ALL_CAPABILITIES, get_capability
from GDesigner.CoRe.task_abstractor import TaskAbstractor
from GDesigner.CoRe.generalized_belief import GeneralizedBelief


class GeneralizedBeliefEvolver:
    """
    泛化的信念进化器 (v5.0)

    核心理念:
    1. 信念应该是关于 Agent **能力** 的,而非 **任务** 的
    2. 通过多次交互,逐步抽象出通用规律
    3. 支持跨任务泛化和知识迁移
    """

    def __init__(
            self,
            llm,
            mind_registry,
            domain: str = "general",
            config: Optional[Dict] = None
    ):
        """
        Args:
            llm: LLM 实例
            mind_registry: MindRegistry 实例
            domain: 任务领域
            config: 配置参数 (可选)
        """
        self.llm = llm
        self.mind_registry = mind_registry
        self.domain = domain
        self.task_abstractor = TaskAbstractor(domain=domain)

        # 配置参数
        self.config = config or {}
        self.abstraction_threshold = self.config.get('abstraction_threshold', 3)
        self.enable_decay = self.config.get('enable_decay', True)
        self.decay_half_life_hours = self.config.get('decay_half_life_hours', 168.0)  # 7天

        # 统计
        self.update_history = []

    async def evolve_beliefs_from_interaction(
            self,
            interaction_trace,  # InteractionTrace (from original code)
            full_chain: List,
            task_success: bool,
            critic_feedback: Optional[str] = None
    ) -> List[GeneralizedBelief]:
        """
        信念进化主入口 (v5.0 泛化版)

        流程:
        1. 任务抽象 → 识别涉及的能力维度
        2. 为每个能力维度生成/更新信念
        3. 应用平滑更新机制
        """

        # === Step 1: 任务抽象 ===
        task_types = self.task_abstractor.extract_task_types(interaction_trace.task)
        task_complexity = self.task_abstractor.extract_complexity(
            interaction_trace.task,
            interaction_trace.output
        )
        key_concepts = self.task_abstractor.extract_key_concepts(interaction_trace.task)

        print(f"[BeliefEvolver v5.0] Task abstraction:")
        print(f"  Capabilities: {task_types}")
        print(f"  Complexity: {task_complexity}")
        print(f"  Key Concepts: {key_concepts}")

        all_updates = []

        # === Step 2: 为每个能力维度生成/更新信念 ===
        for capability in task_types:
            # 检索现有信念
            existing_beliefs = self.mind_registry.get_beliefs_by_capability(
                from_agent=interaction_trace.from_agent,
                to_agent=interaction_trace.to_agent,
                capability=capability
            )

            # 决定更新策略
            if task_success and interaction_trace.success:
                # 正向强化
                update = await self._positive_reinforcement_generalized(
                    interaction_trace,
                    capability,
                    task_complexity,
                    key_concepts,
                    existing_beliefs
                )
            elif not interaction_trace.success:
                # 失败归因
                update = await self._failure_attribution_generalized(
                    interaction_trace,
                    capability,
                    task_complexity,
                    key_concepts,
                    existing_beliefs,
                    full_chain
                )
            else:
                # 混合结果
                update = await self._nuanced_update_generalized(
                    interaction_trace,
                    capability,
                    task_complexity,
                    key_concepts,
                    task_success,
                    existing_beliefs
                )

            if update:
                all_updates.append(update)
                # 应用更新
                self.mind_registry.add_generalized_belief(update)

        self.update_history.extend(all_updates)
        return all_updates

    # ------------------------------------------------------------------------
    # 正向强化 (泛化版)
    # ------------------------------------------------------------------------

    async def _positive_reinforcement_generalized(
            self,
            trace,
            capability: str,
            complexity: str,
            key_concepts: List[str],
            existing_beliefs: List[GeneralizedBelief]
    ) -> Optional[GeneralizedBelief]:
        """生成泛化的正向信念"""

        capability_info = get_capability(capability)
        if not capability_info:
            print(f"[Warning] Unknown capability: {capability}")
            return None

        # 构建 Prompt
        history_context = ""
        if existing_beliefs:
            old_belief = existing_beliefs[0]
            history_context = f"""
**EXISTING BELIEF** (refine if needed):
Current Description: "{old_belief.general_description}"
Success Rate: {old_belief.success_rate:.1%} ({old_belief.success_count}/{old_belief.total_count})
Applicable: {', '.join(old_belief.applicable_contexts) if old_belief.applicable_contexts else 'Not specified'}

Your task is to REFINE this belief based on new evidence, NOT replace it entirely.
"""

        prompt = f"""You are evaluating an agent's **general capability**, NOT analyzing a specific task.

**Capability Being Assessed**: {capability_info.name}
Definition: {capability_info.description}

**Interaction Outcome**: ✓ SUCCESS
- Agent: {trace.to_agent}
- Task Complexity: {complexity}
- Key Concepts Involved: {', '.join(key_concepts) if key_concepts else 'General'}

{history_context}

**YOUR TASK**:
Generate a SHORT, GENERAL statement about this agent's capability in "{capability_info.name}".

**CRITICAL RULES**:
1. ❌ Do NOT mention specific task details (numbers, names, exact problems)
2. ✓ Focus on the GENERAL SKILL demonstrated
3. ✓ Keep it under 70 characters
4. ✓ Use present tense and active voice
5. ✓ Be specific about the capability level (e.g., "strong", "reliable", "proficient")

**Examples of GOOD beliefs**:
- "Strong at mathematical reasoning"
- "Reliable for code generation tasks"
- "Proficient in problem decomposition"
- "Effective with moderate complexity tasks"

**Examples of BAD beliefs** (too specific - DON'T do this):
- "Good at solving quadratic equations with real coefficients" ❌
- "Can calculate the sum of 15 and 27" ❌
- "Successfully implemented bubble sort for list of 10 items" ❌

**Output Format**:
BELIEF: <your general statement here>
"""

        messages = [
            {'role': 'system',
             'content': 'You are an expert at capability assessment. Focus on generalizable skills, not task specifics.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)
        belief_text = self._extract_belief_statement(response)

        # 验证信念质量 (检查是否过于具体)
        if not self._validate_belief_generality(belief_text):
            print(f"[Warning] Generated belief too specific, using fallback: {belief_text}")
            belief_text = f"Capable of {capability_info.name.lower()}"

        # 统计更新
        if existing_beliefs:
            old_belief = existing_beliefs[0]
            new_success = old_belief.success_count + 1
            new_total = old_belief.total_count + 1
            applicable_contexts = old_belief.applicable_contexts.copy()
            known_limitations = old_belief.known_limitations.copy()
            all_concepts = list(set(old_belief.key_concepts + key_concepts))
        else:
            new_success = 1
            new_total = 1
            applicable_contexts = []
            known_limitations = []
            all_concepts = key_concepts

        # 更新适用场景
        context_key = f"{complexity}_tasks"
        if context_key not in applicable_contexts:
            applicable_contexts.append(context_key)

        # 创建新信念
        return GeneralizedBelief(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            capability_dimension=capability,
            general_description=belief_text,
            success_count=new_success,
            total_count=new_total,
            applicable_contexts=applicable_contexts,
            known_limitations=known_limitations,
            key_concepts=all_concepts[:5],  # 限制数量
            confidence=new_success / new_total if new_total > 0 else 0.5,
            evidence_count=new_total
        )

    # ------------------------------------------------------------------------
    # 失败归因 (泛化版)
    # ------------------------------------------------------------------------

    async def _failure_attribution_generalized(
            self,
            trace,
            capability: str,
            complexity: str,
            key_concepts: List[str],
            existing_beliefs: List[GeneralizedBelief],
            full_chain: List
    ) -> Optional[GeneralizedBelief]:
        """生成泛化的失败诊断信念"""

        capability_info = get_capability(capability)
        if not capability_info:
            return None

        history_context = ""
        if existing_beliefs:
            old_belief = existing_beliefs[0]
            history_context = f"""
**EXISTING BELIEF**:
"{old_belief.general_description}"
Success Rate: {old_belief.success_rate:.1%}
Known Limitations: {', '.join(old_belief.known_limitations) if old_belief.known_limitations else 'None recorded'}
"""

        prompt = f"""You are diagnosing an agent's **capability limitation**, NOT explaining a specific failure.

**Capability Being Assessed**: {capability_info.name}
Definition: {capability_info.description}

**Interaction Outcome**: ✗ FAILURE
- Agent: {trace.to_agent}
- Task Complexity: {complexity}
- Failure Type: {trace.failure_reason or 'Unspecified'}
- Key Concepts: {', '.join(key_concepts) if key_concepts else 'General'}

{history_context}

**YOUR TASK**:
Identify a GENERAL limitation or weakness related to "{capability_info.name}".

**CRITICAL RULES**:
1. ❌ Do NOT describe the specific task that failed
2. ✓ Focus on the CAPABILITY GAP demonstrated
3. ✓ Be constructive (what is missing, not just "bad")
4. ✓ Keep it under 80 characters
5. ✓ Consider if this is a consistent pattern or isolated incident

**Examples of GOOD beliefs**:
- "Struggles with multi-step reasoning"
- "Weak at handling edge cases"
- "Needs clearer problem specifications"
- "Limited experience with complex scenarios"

**Examples of BAD beliefs** (too specific - DON'T do this):
- "Failed to solve x² + 5x + 6 = 0" ❌
- "Couldn't implement bubble sort for list [3,1,4,1,5]" ❌

**Output Format**:
BELIEF: <general limitation statement>
"""

        messages = [
            {'role': 'system',
             'content': 'You are an expert at identifying capability gaps. Focus on patterns, not isolated failures.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)
        belief_text = self._extract_belief_statement(response)

        # 验证
        if not self._validate_belief_generality(belief_text):
            belief_text = f"Needs improvement in {capability_info.name.lower()}"

        # 统计更新 (失败不增加成功计数)
        if existing_beliefs:
            old_belief = existing_beliefs[0]
            new_success = old_belief.success_count
            new_total = old_belief.total_count + 1
            applicable_contexts = old_belief.applicable_contexts.copy()
            known_limitations = old_belief.known_limitations.copy()
            all_concepts = list(set(old_belief.key_concepts + key_concepts))
        else:
            new_success = 0
            new_total = 1
            applicable_contexts = []
            known_limitations = []
            all_concepts = key_concepts

        # 更新限制
        limitation_key = f"struggles_with_{complexity}_tasks"
        if limitation_key not in known_limitations:
            known_limitations.append(limitation_key)

        return GeneralizedBelief(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            capability_dimension=capability,
            general_description=belief_text,
            success_count=new_success,
            total_count=new_total,
            applicable_contexts=applicable_contexts,
            known_limitations=known_limitations,
            key_concepts=all_concepts[:5],
            confidence=new_success / new_total if new_total > 0 else 0.3,
            evidence_count=new_total
        )

    # ------------------------------------------------------------------------
    # 细微更新 (泛化版)
    # ------------------------------------------------------------------------

    async def _nuanced_update_generalized(
            self,
            trace,
            capability: str,
            complexity: str,
            key_concepts: List[str],
            task_success: bool,
            existing_beliefs: List[GeneralizedBelief]
    ) -> Optional[GeneralizedBelief]:
        """处理混合结果的更新"""

        if not existing_beliefs:
            return None

        old_belief = existing_beliefs[0]

        # 统计更新 (部分成功计为 0.5)
        success_increment = 0.5 if trace.success else 0

        return GeneralizedBelief(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            capability_dimension=capability,
            general_description=old_belief.general_description,
            success_count=old_belief.success_count + success_increment,
            total_count=old_belief.total_count + 1,
            applicable_contexts=old_belief.applicable_contexts,
            known_limitations=old_belief.known_limitations,
            key_concepts=list(set(old_belief.key_concepts + key_concepts))[:5],
            confidence=old_belief.confidence * 0.95,  # 略微降低信心
            evidence_count=old_belief.evidence_count + 1
        )

    # ------------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------------

    def _extract_belief_statement(self, response: str) -> str:
        """从 LLM 响应提取信念陈述"""
        for line in response.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('BELIEF:'):
                return line_stripped.replace('BELIEF:', '').strip()

        # 降级: 返回第一个非空行
        for line in response.split('\n'):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('**'):
                return line_stripped

        return "Capability assessment updated"

    def _validate_belief_generality(self, belief_text: str) -> bool:
        """
        验证信念是否足够泛化

        检查规则:
        1. 不包含具体数字
        2. 不包含引号内容
        3. 长度合理
        """
        import re

        # 检查具体数字 (允许百分比)
        if re.search(r'\b\d{2,}\b', belief_text):  # 两位以上的数字
            return False

        # 检查引号
        if '"' in belief_text or "'" in belief_text:
            return False

        # 检查长度
        if len(belief_text) > 100:
            return False

        return True
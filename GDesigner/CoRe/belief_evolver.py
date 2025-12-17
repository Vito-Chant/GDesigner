"""
CoRe Framework v4.5: 改进的信念进化模块
核心创新：引入信念动量与平滑更新机制

关键改进：
1. 贝叶斯平滑更新：不直接丢弃历史经验
2. 信念动量：历史证据权重递减而非清零
3. 内容渐进融合：LLM 生成考虑历史信念
4. 异常检测：防止单次事件过度影响
5. 信念衰减：旧证据权重随时间衰减
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import asyncio
from datetime import datetime, timedelta
import math


@dataclass
class InteractionTrace:
    """完整的Agent交互记录"""
    from_agent: str
    to_agent: str
    task: str
    suggestion: str
    output: str
    success: bool
    failure_reason: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BeliefUpdate:
    """信念更新操作 (v4.5 增强版)"""
    from_agent: str
    to_agent: str
    old_belief: Optional[str]
    new_belief: str
    update_reason: str

    # 统计信息
    success_delta: int
    total_delta: int

    # v4.5 新增：更新元信息
    confidence_before: float = 0.5
    confidence_after: float = 0.5
    evidence_strength: float = 1.0  # 本次证据的强度（0-1）
    update_mode: str = "normal"  # normal, reinforcement, correction

    @property
    def confidence_change(self) -> float:
        return self.confidence_after - self.confidence_before


@dataclass
class BeliefEvolutionConfig:
    """信念进化配置参数"""
    # 动量参数
    momentum_factor: float = 0.7  # 历史信念保留权重 [0-1]
    learning_rate: float = 0.3    # 新证据学习率 [0-1]

    # 贝叶斯先验
    prior_strength: int = 2        # 等价于 N 次先验观察
    prior_success_rate: float = 0.5  # 先验成功率

    # 异常检测
    outlier_threshold: float = 2.5  # 标准差阈值
    min_samples_for_outlier: int = 5  # 最少样本数才启用异常检测

    # 信念衰减
    enable_decay: bool = True
    decay_half_life_hours: float = 168.0  # 7天半衰期

    # 内容融合
    enable_content_fusion: bool = True  # 是否融合历史信念内容
    fusion_temperature: float = 0.7     # 融合程度（越高越保守）


class BeliefEvolver:
    """
    改进的信念进化器 (v4.5)

    核心创新：
    1. 平滑更新：不激进丢弃历史
    2. 动量机制：积累长期经验
    3. 智能融合：结合 LLM 生成与历史信念
    """

    def __init__(self, llm, mind_registry, config: Optional[BeliefEvolutionConfig] = None):
        self.llm = llm
        self.mind_registry = mind_registry
        self.config = config or BeliefEvolutionConfig()
        self.update_history = []

    async def evolve_beliefs_from_interaction(
        self,
        interaction_trace: InteractionTrace,
        full_chain: List[InteractionTrace],
        task_success: bool,
        critic_feedback: Optional[str] = None
    ) -> List[BeliefUpdate]:
        """信念进化主入口"""

        # 获取现有信念
        existing_beliefs = self.mind_registry.get_beliefs_about(
            to_agent=interaction_trace.to_agent,
            from_agent=interaction_trace.from_agent
        )

        # 计算当前置信度（作为更新前基线）
        confidence_before = existing_beliefs[0].confidence if existing_beliefs else 0.5

        # 根据交互结果选择更新策略
        if task_success and interaction_trace.success:
            updates = await self._positive_reinforcement(
                interaction_trace, existing_beliefs, critic_feedback
            )
        elif not interaction_trace.success:
            updates = await self._failure_attribution(
                interaction_trace, full_chain, existing_beliefs, critic_feedback
            )
        else:
            updates = await self._nuanced_update(
                interaction_trace, task_success, existing_beliefs, critic_feedback
            )

        # 应用更新（使用平滑机制）
        for update in updates:
            self._apply_smooth_belief_update(update, existing_beliefs)

        self.update_history.extend(updates)
        return updates

    async def _positive_reinforcement(
        self,
        trace: InteractionTrace,
        existing_beliefs: List,
        feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """正向强化（v4.5 改进版）"""

        # 1. 评估当前状态
        has_history = len(existing_beliefs) > 0
        old_content = existing_beliefs[0].content if has_history else None
        old_stats = {
            'success': existing_beliefs[0].success_count if has_history else 0,
            'total': existing_beliefs[0].total_count if has_history else 0
        }

        # 2. 计算证据强度
        evidence_strength = self._calculate_evidence_strength(
            success=True,
            existing_stats=old_stats,
            is_outlier=False
        )

        # 3. 生成新信念内容（考虑历史）
        new_content = await self._generate_belief_with_history(
            trace=trace,
            old_belief=old_content,
            interaction_type="positive",
            feedback=feedback
        )

        # 4. 计算更新后的置信度
        new_success = old_stats['success'] + 1
        new_total = old_stats['total'] + 1
        confidence_after = self._bayesian_confidence(new_success, new_total)

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=old_content,
            new_belief=new_content,
            update_reason=f"Positive reinforcement (strength: {evidence_strength:.2f})",
            success_delta=1,
            total_delta=1,
            confidence_before=existing_beliefs[0].confidence if has_history else 0.5,
            confidence_after=confidence_after,
            evidence_strength=evidence_strength,
            update_mode="reinforcement"
        )

        return [update]

    async def _failure_attribution(
        self,
        trace: InteractionTrace,
        full_chain: List[InteractionTrace],
        existing_beliefs: List,
        feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """失败归因（v4.5 改进版）"""

        has_history = len(existing_beliefs) > 0
        old_content = existing_beliefs[0].content if has_history else None
        old_stats = {
            'success': existing_beliefs[0].success_count if has_history else 0,
            'total': existing_beliefs[0].total_count if has_history else 0
        }

        # 异常检测：防止单次失败过度影响
        is_outlier = self._detect_outlier(
            current_success=False,
            historical_stats=old_stats
        )

        # 降低异常事件的证据强度
        evidence_strength = self._calculate_evidence_strength(
            success=False,
            existing_stats=old_stats,
            is_outlier=is_outlier
        )

        # 生成诊断性信念（考虑历史）
        new_content = await self._generate_belief_with_history(
            trace=trace,
            old_belief=old_content,
            interaction_type="failure",
            feedback=feedback,
            full_chain=full_chain
        )

        # 计算更新后置信度
        new_success = old_stats['success']  # 失败不增加成功
        new_total = old_stats['total'] + 1
        confidence_after = self._bayesian_confidence(new_success, new_total)

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=old_content,
            new_belief=new_content,
            update_reason=f"Failure attribution (outlier: {is_outlier}, strength: {evidence_strength:.2f})",
            success_delta=0,
            total_delta=1,
            confidence_before=existing_beliefs[0].confidence if has_history else 0.5,
            confidence_after=confidence_after,
            evidence_strength=evidence_strength,
            update_mode="correction"
        )

        return [update]

    async def _nuanced_update(
        self,
        trace: InteractionTrace,
        task_success: bool,
        existing_beliefs: List,
        feedback: Optional[str]
    ) -> List[BeliefUpdate]:
        """细微更新（局部成功但全局失败，或反之）"""

        has_history = len(existing_beliefs) > 0
        old_content = existing_beliefs[0].content if has_history else None
        old_stats = {
            'success': existing_beliefs[0].success_count if has_history else 0,
            'total': existing_beliefs[0].total_count if has_history else 0
        }

        # 部分成功的证据强度较低
        evidence_strength = 0.5

        # 决定成功增量
        if trace.success and not task_success:
            success_delta = 0  # 局部成功但全局失败 -> 不计入
        elif not trace.success and task_success:
            success_delta = 0  # 局部失败但全局成功 -> 不计入
        else:
            success_delta = 1 if trace.success else 0

        new_content = await self._generate_belief_with_history(
            trace=trace,
            old_belief=old_content,
            interaction_type="nuanced",
            feedback=feedback
        )

        new_success = old_stats['success'] + success_delta
        new_total = old_stats['total'] + 1
        confidence_after = self._bayesian_confidence(new_success, new_total)

        update = BeliefUpdate(
            from_agent=trace.from_agent,
            to_agent=trace.to_agent,
            old_belief=old_content,
            new_belief=new_content,
            update_reason="Nuanced update from mixed outcomes",
            success_delta=success_delta,
            total_delta=1,
            confidence_before=existing_beliefs[0].confidence if has_history else 0.5,
            confidence_after=confidence_after,
            evidence_strength=evidence_strength,
            update_mode="normal"
        )

        return [update]

    def _apply_smooth_belief_update(
        self,
        update: BeliefUpdate,
        existing_beliefs: List
    ):
        """
        应用平滑信念更新 (v4.5 核心创新)

        关键改进：
        1. 使用动量保留历史经验
        2. 信念内容渐进融合
        3. 考虑时间衰减
        """
        from GDesigner.CoRe.mind_registry import RelationalBelief

        if existing_beliefs:
            old_belief = existing_beliefs[0]

            # === 1. 应用时间衰减 ===
            if self.config.enable_decay:
                decay_factor = self._calculate_time_decay(old_belief.last_updated)
                effective_old_count = old_belief.total_count * decay_factor
            else:
                effective_old_count = old_belief.total_count

            # === 2. 动量更新统计 ===
            # 不是简单累加，而是加权平均
            momentum = self.config.momentum_factor
            new_success_count = int(
                momentum * old_belief.success_count +
                (1 - momentum) * update.success_delta * self.config.learning_rate * 10
            )
            new_total_count = int(
                momentum * effective_old_count +
                (1 - momentum) * update.total_delta * 10
            )

            # 保证总数不小于成功数
            new_total_count = max(new_total_count, new_success_count)

            # === 3. 内容渐进融合 ===
            if self.config.enable_content_fusion and old_belief.content:
                fused_content = self._fuse_belief_content(
                    old_content=old_belief.content,
                    new_content=update.new_belief,
                    fusion_weight=self.config.fusion_temperature,
                    evidence_strength=update.evidence_strength
                )
            else:
                fused_content = update.new_belief

            evidence_count = old_belief.evidence_count + 1

        else:
            # 新建信念
            new_success_count = update.success_delta
            new_total_count = update.total_delta
            fused_content = update.new_belief
            evidence_count = 1

        # 创建新信念对象
        new_belief = RelationalBelief(
            from_agent=update.from_agent,
            to_agent=update.to_agent,
            belief_type='capability_assessment',
            content=fused_content,
            success_count=new_success_count,
            total_count=new_total_count,
            evidence_count=evidence_count,
            last_updated=datetime.now().isoformat()
        )

        self.mind_registry.add_belief(new_belief)

    async def _generate_belief_with_history(
        self,
        trace: InteractionTrace,
        old_belief: Optional[str],
        interaction_type: str,
        feedback: Optional[str],
        full_chain: Optional[List[InteractionTrace]] = None
    ) -> str:
        """
        生成考虑历史的新信念 (v4.5 核心创新)

        关键改进：
        - Prompt 中注入历史信念
        - 要求 LLM 渐进式调整而非推翻
        """

        # 构建历史上下文
        history_context = ""
        if old_belief:
            history_context = f"""
**EXISTING BELIEF (Historical Context)**:
"{old_belief}"

**IMPORTANT**: This belief is based on accumulated evidence. Your new assessment should:
1. Build upon this foundation (don't discard it)
2. Make incremental adjustments based on new evidence
3. Preserve valuable insights from past interactions
"""

        if interaction_type == "positive":
            prompt = f"""An agent interaction was SUCCESSFUL.

{history_context}

**NEW INTERACTION**:
- From: {trace.from_agent}
- To: {trace.to_agent}
- Task: {trace.task}
- Suggestion: {trace.suggestion}
- Outcome: SUCCESS

**TASK**: Generate an UPDATED belief about {trace.to_agent}'s capabilities.
- Keep it concise (under 100 chars)
- If historical belief exists, refine it rather than replace it
- Focus on confirming/strengthening positive patterns

Format: BELIEF: <statement>"""

        elif interaction_type == "failure":
            prompt = f"""An agent interaction FAILED.

{history_context}

**NEW INTERACTION**:
- From: {trace.from_agent}
- To: {trace.to_agent}
- Task: {trace.task}
- Suggestion: {trace.suggestion}
- Outcome: FAILURE
- Reason: {trace.failure_reason}

**TASK**: Generate an UPDATED belief that explains this failure.
- Keep it concise (under 100 chars)
- If historical belief exists, add nuance rather than contradict
- Identify specific weaknesses or gaps

Format: BELIEF: <statement>"""

        else:  # nuanced
            prompt = f"""An agent interaction had MIXED results.

{history_context}

**NEW INTERACTION**:
- From: {trace.from_agent}
- To: {trace.to_agent}
- Task: {trace.task}
- Outcome: PARTIAL SUCCESS

**TASK**: Generate an UPDATED belief reflecting partial success.
- Keep it concise (under 100 chars)
- Balance strengths and limitations

Format: BELIEF: <statement>"""

        if feedback:
            prompt += f"\n\nAdditional Feedback: {feedback}"

        messages = [
            {'role': 'system', 'content': 'You are an expert at incremental belief refinement.'},
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)
        return self._extract_belief_statement(response)

    def _fuse_belief_content(
        self,
        old_content: str,
        new_content: str,
        fusion_weight: float,
        evidence_strength: float
    ) -> str:
        """
        融合旧信念与新信念内容

        策略：
        - 如果新证据强，偏向新内容
        - 如果旧信念稳定，偏向保留
        - 使用加权拼接或提取关键词
        """
        # 简化版：基于权重选择
        effective_weight = fusion_weight * (1 - evidence_strength)

        if effective_weight > 0.7:
            # 高度保守：保留旧信念，只追加新信息
            if len(new_content) < 50:
                return f"{old_content}; {new_content}"
            else:
                return old_content
        elif effective_weight > 0.3:
            # 中等融合：优先新内容，但引用旧见解
            return new_content
        else:
            # 激进更新：新证据足够强，采用新内容
            return new_content

    def _calculate_evidence_strength(
        self,
        success: bool,
        existing_stats: Dict,
        is_outlier: bool
    ) -> float:
        """
        计算证据强度 [0-1]

        考虑因素：
        1. 样本数量（样本越多，单次影响越小）
        2. 是否异常（异常事件降低权重）
        3. 当前趋势（与历史一致性）
        """
        total = existing_stats['total']

        # 样本数调整
        if total == 0:
            sample_factor = 1.0
        else:
            # 样本越多，单次影响越小
            sample_factor = 1.0 / math.log2(total + 2)

        # 异常惩罚
        outlier_factor = 0.3 if is_outlier else 1.0

        # 趋势一致性
        if total > 0:
            historical_rate = existing_stats['success'] / total
            consistency = 1.0 if success == (historical_rate > 0.5) else 0.7
        else:
            consistency = 1.0

        strength = sample_factor * outlier_factor * consistency
        return max(0.1, min(1.0, strength))

    def _detect_outlier(
        self,
        current_success: bool,
        historical_stats: Dict
    ) -> bool:
        """
        检测是否为异常事件

        使用贝叶斯异常检测：
        - 如果历史成功率很高，突然失败 -> 异常
        - 如果历史失败率很高，突然成功 -> 异常
        """
        total = historical_stats['total']

        if total < self.config.min_samples_for_outlier:
            return False

        success_count = historical_stats['success']
        historical_rate = success_count / total

        # 计算标准差
        variance = historical_rate * (1 - historical_rate)
        std = math.sqrt(variance / total) if total > 0 else 0

        # 检测偏离
        expected = 1.0 if current_success else 0.0
        deviation = abs(expected - historical_rate) / (std + 1e-6)

        return deviation > self.config.outlier_threshold

    def _bayesian_confidence(self, success_count: int, total_count: int) -> float:
        """
        贝叶斯置信度估计（带先验）
        """
        prior_s = self.config.prior_strength * self.config.prior_success_rate
        prior_t = self.config.prior_strength

        posterior_success = success_count + prior_s
        posterior_total = total_count + prior_t

        return posterior_success / posterior_total if posterior_total > 0 else 0.5

    def _calculate_time_decay(self, last_updated: str) -> float:
        """
        计算时间衰减因子

        使用指数衰减：
        decay_factor = 0.5^(t / half_life)
        """
        if not self.config.enable_decay:
            return 1.0

        try:
            last_time = datetime.fromisoformat(last_updated)
            now = datetime.now()
            hours_elapsed = (now - last_time).total_seconds() / 3600

            decay_factor = 0.5 ** (hours_elapsed / self.config.decay_half_life_hours)
            return max(0.1, decay_factor)  # 最低保留 10%
        except:
            return 1.0

    def _extract_belief_statement(self, response: str) -> str:
        """从 LLM 响应提取信念陈述"""
        for line in response.split('\n'):
            if line.strip().startswith('BELIEF:'):
                return line.replace('BELIEF:', '').strip()

        for line in response.split('\n'):
            if line.strip():
                return line.strip()

        return "Capability assessment updated"

    def get_evolution_summary(self) -> Dict:
        """获取进化摘要"""
        if not self.update_history:
            return {
                'total_updates': 0,
                'avg_confidence_change': 0.0
            }

        return {
            'total_updates': len(self.update_history),
            'positive_updates': sum(1 for u in self.update_history if u.confidence_change > 0),
            'negative_updates': sum(1 for u in self.update_history if u.confidence_change < 0),
            'avg_confidence_change': sum(u.confidence_change for u in self.update_history) / len(self.update_history),
            'avg_evidence_strength': sum(u.evidence_strength for u in self.update_history) / len(self.update_history),
            'outlier_corrections': sum(1 for u in self.update_history if 'outlier: True' in u.update_reason)
        }
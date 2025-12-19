"""
泛化的信念数据结构 (Generalized Belief)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class GeneralizedBelief:
    """
    泛化的能力信念 (v5.0)

    核心设计理念:
    - 信念是关于 **能力** 的,不是关于 **任务** 的
    - 可以跨多个具体任务复用
    - 包含适用场景和已知限制
    """

    # === 基本信息 ===
    from_agent: str  # 评估者
    to_agent: str  # 被评估者

    # === 能力维度 (核心) ===
    capability_dimension: str  # 对应 capability_taxonomy 中的 key

    # === 泛化的描述 ===
    general_description: str  # 通用描述,不包含具体任务细节

    # === 统计信息 ===
    success_count: int  # 成功次数
    total_count: int  # 总尝试次数

    # === 上下文信息 (可选) ===
    applicable_contexts: List[str] = field(default_factory=list)
    # 适用场景列表,例如: ["simple_tasks", "moderate_tasks", "mathematical_domain"]

    known_limitations: List[str] = field(default_factory=list)
    # 已知限制,例如: ["struggles_with_edge_cases", "needs_clear_specifications"]

    key_concepts: List[str] = field(default_factory=list)
    # 相关概念,例如: ["algebra", "geometry"]

    # === 元信息 ===
    confidence: float = 0.5  # 置信度 [0, 1]
    evidence_count: int = 1  # 证据数量 (总交互次数)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # === 衰减参数 (可选) ===
    decay_factor: float = 1.0  # 时间衰减因子

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count

    @property
    def reliability_score(self) -> float:
        """
        综合可靠性评分

        考虑因素:
        - 成功率
        - 样本数量 (少样本降低信心)
        - 时间衰减
        """
        # 基础成功率
        base_rate = self.success_rate

        # 样本量修正 (贝叶斯平滑)
        # 少样本时向 0.5 收缩
        prior_strength = 2
        adjusted_rate = (self.success_count + 1) / (self.total_count + prior_strength)

        # 时间衰减
        reliability = adjusted_rate * self.decay_factor

        return reliability

    def to_text(self, include_stats: bool = True) -> str:
        """
        转换为自然语言描述

        Args:
            include_stats: 是否包含统计信息
        """
        text = f"{self.from_agent} → {self.to_agent}: {self.general_description}"

        if include_stats:
            rate = self.success_rate

            # 可靠性描述
            if self.total_count == 0:
                reliability = "no evidence"
            elif rate >= 0.8:
                reliability = f"highly reliable ({self.success_count}/{self.total_count})"
            elif rate >= 0.6:
                reliability = f"generally reliable ({self.success_count}/{self.total_count})"
            elif rate >= 0.4:
                reliability = f"moderately reliable ({self.success_count}/{self.total_count})"
            else:
                reliability = f"less reliable ({self.success_count}/{self.total_count})"

            text += f" [{reliability}]"

            # 适用场景
            if self.applicable_contexts:
                contexts = ", ".join(self.applicable_contexts[:2])
                text += f" | Contexts: {contexts}"

            # 限制
            if self.known_limitations:
                limitations = ", ".join(self.known_limitations[:2])
                text += f" | Limitations: {limitations}"

        return text

    def to_dict(self) -> Dict:
        """转换为字典 (用于序列化)"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneralizedBelief':
        """从字典创建 (用于反序列化)"""
        return cls(**data)

    def matches_context(self, context: str) -> bool:
        """判断信念是否适用于给定上下文"""
        return context in self.applicable_contexts

    def has_limitation(self, limitation_key: str) -> bool:
        """判断是否存在特定限制"""
        return any(limitation_key in lim for lim in self.known_limitations)
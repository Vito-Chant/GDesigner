"""
任务抽象器 (Task Abstractor)
从具体任务中提取抽象特征
"""

import re
from typing import Set, Dict, List
from GDesigner.CoRe.capability_taxonomy import ALL_CAPABILITIES


class TaskAbstractor:
    """从具体任务中提取抽象特征"""

    def __init__(self, domain: str = "general"):
        """
        Args:
            domain: 任务所属领域 (用于选择相关的能力维度)
        """
        self.domain = domain
        self.capabilities = ALL_CAPABILITIES

    def extract_task_types(self, task: str) -> Set[str]:
        """
        识别任务涉及的能力维度

        Returns:
            Set of capability IDs (e.g., {"mathematical_reasoning", "code_generation"})
        """
        task_lower = task.lower()
        matched_capabilities = set()

        # 评分机制: 每个能力维度计算匹配分数
        capability_scores = {}

        for cap_id, cap_dim in self.capabilities.items():
            score = 0

            # 1. 关键词匹配
            for keyword in cap_dim.keywords:
                if keyword in task_lower:
                    # 完整单词匹配得分更高
                    if re.search(r'\b' + re.escape(keyword) + r'\b', task_lower):
                        score += 2
                    else:
                        score += 1

            # 2. 名称匹配 (如果任务中包含能力名称本身)
            if cap_dim.name.lower() in task_lower:
                score += 3

            if score > 0:
                capability_scores[cap_id] = score

        # 选择得分最高的能力 (至少得分 >= 2)
        if capability_scores:
            # 获取最高分
            max_score = max(capability_scores.values())

            # 选择得分 >= max_score * 0.5 的能力 (允许多个高得分能力)
            threshold = max(2, max_score * 0.5)
            matched_capabilities = {
                cap_id for cap_id, score in capability_scores.items()
                if score >= threshold
            }

        # 如果没有匹配到任何能力,返回默认
        if not matched_capabilities:
            matched_capabilities.add("analytical_thinking")  # 通用兜底

        return matched_capabilities

    def extract_complexity(self, task: str, output: str = "") -> str:
        """
        评估任务复杂度

        Args:
            task: 任务描述
            output: Agent 输出 (可选,用于辅助判断)

        Returns:
            "simple", "moderate", "complex"
        """
        # 综合多个因素评估复杂度
        complexity_score = 0

        # 1. 任务长度
        word_count = len(task.split())
        if word_count < 20:
            complexity_score += 0
        elif word_count < 50:
            complexity_score += 1
        elif word_count < 100:
            complexity_score += 2
        else:
            complexity_score += 3

        # 2. 复杂度关键词
        complexity_keywords = {
            "simple": ["simple", "basic", "easy", "straightforward"],
            "moderate": ["moderate", "intermediate", "standard"],
            "complex": ["complex", "advanced", "difficult", "challenging",
                        "multi-step", "sophisticated", "intricate"]
        }

        task_lower = task.lower()
        for level, keywords in complexity_keywords.items():
            if any(kw in task_lower for kw in keywords):
                if level == "simple":
                    complexity_score -= 1
                elif level == "complex":
                    complexity_score += 2

        # 3. 多步骤指示
        multi_step_indicators = ["first", "then", "next", "finally", "step", "stage"]
        if sum(1 for ind in multi_step_indicators if ind in task_lower) >= 2:
            complexity_score += 1

        # 4. 输出长度 (如果提供)
        if output:
            output_lines = len(output.split('\n'))
            if output_lines > 50:
                complexity_score += 1

        # 映射分数到等级
        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 3:
            return "moderate"
        else:
            return "complex"

    def abstract_task_description(
            self,
            task: str,
            max_length: int = 100
    ) -> str:
        """
        生成任务的抽象描述 (移除具体细节)

        Args:
            task: 原始任务描述
            max_length: 最大长度

        Returns:
            抽象后的任务描述
        """
        abstract = task

        # 1. 移除具体数字
        abstract = re.sub(r'\b\d+(\.\d+)?\b', 'N', abstract)

        # 2. 移除专有名词 (首字母大写的词,但保留句首)
        words = abstract.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                words[i] = 'Entity'
        abstract = ' '.join(words)

        # 3. 移除引号内的具体内容
        abstract = re.sub(r'"[^"]*"', '"..."', abstract)
        abstract = re.sub(r"'[^']*'", "'...'", abstract)

        # 4. 截断
        if len(abstract) > max_length:
            abstract = abstract[:max_length] + "..."

        return abstract

    def extract_key_concepts(self, task: str) -> List[str]:
        """
        提取任务中的关键概念 (用于信念的上下文标注)

        Returns:
            List of key concepts (e.g., ["algebra", "equation solving"])
        """
        concepts = []

        # 简单的关键词提取 (可以用更复杂的 NLP 方法)
        task_lower = task.lower()

        # 从能力关键词中匹配
        for cap_id, cap_dim in self.capabilities.items():
            for keyword in cap_dim.keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', task_lower):
                    concepts.append(keyword)

        # 去重并限制数量
        concepts = list(set(concepts))[:5]

        return concepts
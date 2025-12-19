"""
能力分类体系 (Capability Taxonomy)
定义所有可能的 Agent 能力维度
"""

from typing import List
from dataclasses import dataclass


@dataclass
class CapabilityDimension:
    """能力维度定义"""
    name: str
    description: str
    keywords: List[str]  # 用于任务分类的关键词
    domain: str = "general"  # 所属领域 (可选)


# 核心能力维度 (跨领域通用)
CORE_CAPABILITIES = {
    # 推理类
    "mathematical_reasoning": CapabilityDimension(
        name="Mathematical Reasoning",
        description="Ability to perform mathematical analysis, calculations, and proofs",
        keywords=["math", "calculate", "equation", "algebra", "geometry", "arithmetic",
                  "solve", "compute", "number", "formula"],
        domain="reasoning"
    ),

    "logical_reasoning": CapabilityDimension(
        name="Logical Reasoning",
        description="Ability to perform logical deduction, inference, and argumentation",
        keywords=["logic", "deduce", "infer", "conclude", "reason", "proof", "argument",
                  "syllogism", "valid", "sound"],
        domain="reasoning"
    ),

    "analytical_thinking": CapabilityDimension(
        name="Analytical Thinking",
        description="Ability to break down and analyze complex information",
        keywords=["analyze", "examine", "investigate", "study", "evaluate", "assess",
                  "compare", "contrast", "interpret"],
        domain="reasoning"
    ),

    # 代码类
    "code_generation": CapabilityDimension(
        name="Code Generation",
        description="Ability to write correct, efficient, and maintainable code",
        keywords=["code", "program", "function", "implement", "algorithm", "write",
                  "develop", "script", "programming"],
        domain="coding"
    ),

    "code_debugging": CapabilityDimension(
        name="Code Debugging",
        description="Ability to identify, diagnose, and fix code errors",
        keywords=["debug", "fix", "error", "bug", "troubleshoot", "diagnose",
                  "exception", "trace", "resolve"],
        domain="coding"
    ),

    "algorithm_design": CapabilityDimension(
        name="Algorithm Design",
        description="Ability to design efficient algorithms and data structures",
        keywords=["algorithm", "complexity", "optimize", "efficiency", "data structure",
                  "time complexity", "space complexity", "design"],
        domain="coding"
    ),

    # 问题解决类
    "problem_decomposition": CapabilityDimension(
        name="Problem Decomposition",
        description="Ability to break down complex problems into manageable sub-problems",
        keywords=["decompose", "breakdown", "structure", "organize", "plan", "divide",
                  "modularize", "separate", "partition"],
        domain="problem_solving"
    ),

    "pattern_recognition": CapabilityDimension(
        name="Pattern Recognition",
        description="Ability to identify patterns, trends, and relationships",
        keywords=["pattern", "trend", "recognize", "identify", "detect", "discover",
                  "similarity", "correlation", "relationship"],
        domain="problem_solving"
    ),

    "creative_synthesis": CapabilityDimension(
        name="Creative Synthesis",
        description="Ability to generate novel solutions or combine ideas innovatively",
        keywords=["creative", "innovative", "novel", "synthesis", "idea", "generate",
                  "brainstorm", "invent", "original"],
        domain="problem_solving"
    ),

    # 验证类
    "verification": CapabilityDimension(
        name="Verification",
        description="Ability to validate correctness of solutions and identify errors",
        keywords=["verify", "validate", "check", "review", "test", "confirm",
                  "ensure", "assess accuracy", "quality control"],
        domain="validation"
    ),

    "error_diagnosis": CapabilityDimension(
        name="Error Diagnosis",
        description="Ability to identify root causes of failures and provide explanations",
        keywords=["diagnose", "root cause", "failure", "why", "explain error",
                  "investigate", "trace", "pinpoint"],
        domain="validation"
    ),

    # 知识类
    "knowledge_retrieval": CapabilityDimension(
        name="Knowledge Retrieval",
        description="Ability to recall and apply domain-specific knowledge",
        keywords=["knowledge", "fact", "domain", "expertise", "recall", "remember",
                  "background", "context", "information"],
        domain="knowledge"
    ),

    "domain_expertise": CapabilityDimension(
        name="Domain Expertise",
        description="Deep understanding of specific domain concepts and practices",
        keywords=["expert", "specialist", "professional", "advanced", "specialized",
                  "in-depth", "comprehensive", "mastery"],
        domain="knowledge"
    ),

    # 沟通协作类
    "explanation": CapabilityDimension(
        name="Explanation",
        description="Ability to clearly explain concepts, solutions, and reasoning",
        keywords=["explain", "clarify", "describe", "illustrate", "communicate",
                  "present", "articulate", "convey", "teach"],
        domain="communication"
    ),

    "coordination": CapabilityDimension(
        name="Coordination",
        description="Ability to coordinate with other agents and integrate outputs",
        keywords=["coordinate", "collaborate", "integrate", "combine", "merge",
                  "synthesize outputs", "work together", "teamwork"],
        domain="communication"
    ),
}

# 领域特定能力 (可扩展)
DOMAIN_SPECIFIC_CAPABILITIES = {
    # MMLU 相关
    "historical_knowledge": CapabilityDimension(
        name="Historical Knowledge",
        description="Understanding of historical events, figures, and contexts",
        keywords=["history", "historical", "past", "era", "period", "century",
                  "ancient", "civilization"],
        domain="mmlu"
    ),

    "scientific_knowledge": CapabilityDimension(
        name="Scientific Knowledge",
        description="Understanding of scientific principles and phenomena",
        keywords=["science", "scientific", "physics", "chemistry", "biology",
                  "experiment", "hypothesis", "theory"],
        domain="mmlu"
    ),

    # HumanEval 相关
    "python_programming": CapabilityDimension(
        name="Python Programming",
        description="Proficiency in Python language and its idioms",
        keywords=["python", "pythonic", "list comprehension", "decorator",
                  "generator", "iterator", "class"],
        domain="humaneval"
    ),

    # GSM8K 相关
    "word_problem_solving": CapabilityDimension(
        name="Word Problem Solving",
        description="Ability to interpret and solve mathematical word problems",
        keywords=["word problem", "story problem", "real-world", "application",
                  "scenario", "context"],
        domain="gsm8k"
    ),
}

# 合并所有能力维度
ALL_CAPABILITIES = {**CORE_CAPABILITIES, **DOMAIN_SPECIFIC_CAPABILITIES}


def get_capability(capability_id: str) -> CapabilityDimension:
    """获取能力维度定义"""
    return ALL_CAPABILITIES.get(capability_id)


def get_capabilities_by_domain(domain: str) -> dict:
    """获取特定领域的所有能力"""
    return {
        cap_id: cap for cap_id, cap in ALL_CAPABILITIES.items()
        if cap.domain == domain or cap.domain == "general"
    }
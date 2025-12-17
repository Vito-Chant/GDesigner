"""
CoRe v4.3 CodeWriting Agent - KV Cache 优化版
适配 CoRe Graph 的代码生成 Agent

**关键特性**:
- _async_execute 返回 (response, messages) 元组
- messages 包含完整对话历史（System + User + Assistant）
- 为 KV Cache 复用提供物理基础
"""

from typing import List, Any, Dict, Tuple
import weave

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.coding.python_executor import PyExecutor


@AgentRegistry.register('CoReCodeWriting')
class CodeWriting(Node):
    """
    CoRe v4.3 CodeWriting Agent

    **v4.3 KV Cache 优化版**:
    - 返回 (response, messages) 以支持 KV Cache
    - 处理 RAG Context 和 Suggestion
    - 支持内部测试验证
    """

    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "CoReCodeWriting", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)
        self.internal_tests = []

    def extract_example(self, prompt: str) -> list:
        """从 prompt 中提取示例测试用例"""
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    async def _process_inputs(
            self,
            raw_inputs: Dict[str, str],
            spatial_info: Dict[str, Dict],
            temporal_info: Dict[str, Dict],
            **kwargs
    ) -> Tuple[str, str]:
        """
        处理输入并构建完整的 Prompt

        职责：
        1. 构建基础 system_prompt 和 user_prompt
        2. 处理其他 Agent 的输出
        3. 执行内部测试（如果有）
        4. 拼接 RAG Context
        5. 拼接 Suggestion

        Returns:
            (system_prompt, user_prompt)
        """
        # === 1. 提取内部测试用例 ===
        self.internal_tests = self.extract_example(raw_inputs)

        # === 2. 基础 Prompt ===
        system_prompt = self.constraint

        # 添加系统级指令
        if 'system_instruction' in raw_inputs:
            system_prompt += f"\n\n**SYSTEM INSTRUCTION**:\n{raw_inputs['system_instruction']}"

        # === 3. 处理空间和时间信息 ===
        spatial_str = ""
        temporal_str = ""

        for id, info in spatial_info.items():
            output_content = info['output']

            # **关键逻辑**: 如果其他 Agent 提供了代码且通过了测试，可以直接返回
            if (output_content.startswith("```python") and
                    output_content.endswith("```") and
                    self.role != 'Normal Programmer' and
                    self.role != 'Stupid Programmer'):

                code = output_content.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(code, self.internal_tests, timeout=10)

                if is_solved and len(self.internal_tests):
                    # 代码已通过测试，直接返回
                    return "is_solved", output_content

                spatial_str += (
                    f"Agent {id} as a {info['role']}:\n\n"
                    f"The code written by the agent is:\n\n{output_content}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n {feedback}.\n\n"
                )
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {output_content}\n\n"

        for id, info in temporal_info.items():
            output_content = info['output']

            if (output_content.startswith("```python") and
                    output_content.endswith("```") and
                    self.role != 'Normal Programmer' and
                    self.role != 'Stupid Programmer'):

                code = output_content.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(code, self.internal_tests, timeout=10)

                if is_solved and len(self.internal_tests):
                    return "is_solved", output_content

                temporal_str += (
                    f"Agent {id} as a {info['role']}:\n\n"
                    f"The code written by the agent is:\n\n{output_content}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n {feedback}.\n\n"
                )
            else:
                temporal_str += f"Agent {id} as a {info['role']} provides the following info: {output_content}\n\n"

        # === 4. 构建基础任务描述 ===
        user_prompt = f"The task is:\n\n{raw_inputs['task']}\n"
        user_prompt += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str} \n\n" if len(
            spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs and feedbacks of some agents were: \n\n{temporal_str}" if len(
            temporal_str) else ""

        # === 5. 拼接 RAG Context ===
        if 'retrieved_history' in raw_inputs and raw_inputs['retrieved_history']:
            rag_context = raw_inputs['retrieved_history']
            user_prompt += f"""

=== RELEVANT CONTEXT FROM HISTORY ===
{rag_context}
=== END OF CONTEXT ===

"""

        # === 6. 拼接 Suggestion ===
        if 'insight' in raw_inputs and raw_inputs['insight']:
            suggestion = raw_inputs['insight']
            user_prompt += f"""

=== STRATEGIC SUGGESTION (HIGH PRIORITY) ===
{suggestion}

Please consider this guidance when writing your code.
=== END OF SUGGESTION ===

"""

        return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """同步执行（保留兼容性）"""
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)

        # 处理已解决的情况
        if system_prompt == "is_solved":
            return user_prompt

        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    @weave.op()
    async def _async_execute(
            self,
            input: Dict[str, str],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        异步执行 (v4.3 KV Cache 优化版)

        **关键变更**:
        返回值从 `str` 改为 `Tuple[str, List[Dict]]`

        Returns:
            (response, messages) 其中：
            - response: 生成的代码
            - messages: 完整对话历史 [System, User, Assistant]
        """
        # 1. 获取完整 Prompt
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)

        # 2. 处理已解决的情况（代码已通过测试）
        if system_prompt == "is_solved":
            # 返回空消息历史，因为不需要 LLM 调用
            return user_prompt, []

        # 3. 构建消息历史
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        # 4. 调用 LLM 生成代码
        response = await self.llm.agen(messages)

        # 5. Debug 输出（可选）
        print(f"[{self.agent_name}] Generated code (preview):")
        print(response[:200] + "..." if len(response) > 200 else response)

        # **6. 关键步骤：将 Assistant 的响应加入历史**
        messages.append({'role': 'assistant', 'content': response})

        # **7. 返回元组：(代码, 完整历史)**
        return response, messages
from typing import List, Any, Dict

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.coding.python_executor import PyExecutor


def extract_content(output: Any) -> str:
    """Helper function to handle (content, log_prob) tuple from trained agents."""
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0]
    return output


@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    """
    Final Decision Agent for Code Generation

    v4.3 CoRe 适配版:
    - 支持 CoRe Graph 的 spatial_info 格式
    - 处理来自多个 Agent 的代码和反馈
    - 智能选择最佳代码或生成新代码
    """

    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "FinalWriteCode", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        """从 prompt 中提取测试用例"""
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

    def _process_inputs(
            self,
            raw_inputs: Dict[str, str],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ) -> List[Any]:
        """
        处理输入 (v4.3 CoRe 适配版)

        **关键改进**:
        - 按步骤顺序排序 spatial_info
        - 智能提取代码和测试反馈
        - 优先选择已通过测试的代码
        """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{self.role}.\n {self.constraint}"

        spatial_str = ""
        self.internal_tests = self.extract_example(raw_inputs)

        best_passing_code = None  # 记录通过测试的最佳代码

        # ✅ 修改：按步骤顺序排序（支持 CoRe Graph 的命名格式）
        sorted_items = sorted(
            spatial_info.items(),
            key=lambda x: (
                int(x[0].split('_')[-1])
                if 'step' in x[0] and x[0].split('_')[-1].isdigit()
                else 999
            )
        )

        for agent_id, info in sorted_items:
            # ✅ 修改：提取内容（支持 tuple 格式）
            output_content = extract_content(info['output'])
            role_desc = info.get('role', agent_id)

            # 检查是否是 Python 代码
            if output_content.startswith("```python") and output_content.endswith("```"):
                code_snippet = output_content.lstrip("```python\n").rstrip("\n```")

                # 执行内部测试
                is_solved, feedback, state = PyExecutor().execute(
                    code_snippet,
                    self.internal_tests,
                    timeout=10
                )

                # 记录通过测试的代码
                if is_solved and len(self.internal_tests) > 0:
                    best_passing_code = output_content

                spatial_str += (
                    f"\n--- {role_desc} ---\n"
                    f"Code:\n{output_content}\n\n"
                    f"Test Result: {'✓ PASSED' if is_solved else '✗ FAILED'}\n"
                    f"Feedback: {feedback}\n"
                )
            else:
                spatial_str += (
                    f"\n--- {role_desc} ---\n"
                    f"Info: {output_content}\n"
                )

        # ✅ 修改：如果有通过测试的代码，直接返回
        if best_passing_code:
            return "use_passing_code", best_passing_code

        decision_few_shot = self.prompt_set.get_decision_few_shot()

        user_prompt = (
            f"{decision_few_shot}\n\n"
            f"The task is:\n\n{raw_inputs['task']}\n\n"
            f"The outputs and feedbacks from previous agents are:\n{spatial_str}\n\n"
            f"Based on the above analysis and code attempts, write your final implementation.\n"
            f"If one of the provided codes is close to correct, you can refine it.\n"
            f"Otherwise, write a new implementation from scratch."
        )

        return system_prompt, user_prompt

    def _execute(
            self,
            input: Dict[str, str],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ):
        """同步执行"""
        result = self._process_inputs(input, spatial_info, temporal_info)

        # 处理已通过测试的代码
        if result[0] == "use_passing_code":
            return result[1]

        system_prompt, user_prompt = result
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(
            self,
            input: Dict[str, str],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ):
        """异步执行"""
        result = self._process_inputs(input, spatial_info, temporal_info)

        # 处理已通过测试的代码
        if result[0] == "use_passing_code":
            print(f"[{self.agent_name}] Using pre-validated passing code")
            return result[1]

        system_prompt, user_prompt = result
        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        response = await self.llm.agen(message)

        print(f"[{self.agent_name}] Generated final code")

        return response


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "FinalRefer", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(
            self,
            raw_inputs: Dict[str, str],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ) -> List[Any]:
        """处理输入 (v4.3.2: 增强历史提取)"""

        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{self.role}.\n {self.constraint}"

        # ✅ 修改后: 更清晰的历史组织
        spatial_str = ""

        # 按步骤顺序排序（如果键名是 previous_step_N 格式）
        sorted_items = sorted(
            spatial_info.items(),
            key=lambda x: (
                int(x[0].split('_')[-1])
                if 'step' in x[0] and x[0].split('_')[-1].isdigit()
                else 999
            )
        )

        for agent_id, info in sorted_items:
            output_content = extract_content(info['output'])
            role_desc = info.get('role', agent_id)

            # ✅ 更结构化的输出格式
            spatial_str += f"\n--- {role_desc} ---\n"
            spatial_str += f"{output_content}\n"

        decision_few_shot = self.prompt_set.get_decision_few_shot()

        user_prompt = (
            f"{decision_few_shot}\n\n"
            f"The task is:\n\n{raw_inputs['task']}\n\n"
            f"The outputs from previous agents are:\n{spatial_str}\n\n"
            f"Based on the above analysis, provide your final decision."
        )

        return system_prompt, user_prompt

    # def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
    #                     **kwargs) -> List[Any]:
    #     """ To be overriden by the descendant class """
    #     """ Process the raw_inputs(most of the time is a List[Dict]) """
    #     self.role = self.prompt_set.get_decision_role()
    #     self.constraint = self.prompt_set.get_decision_constraint()
    #     system_prompt = f"{self.role}.\n {self.constraint}"
    #
    #     spatial_str = ""
    #     for id, info in spatial_info.items():
    #         # [Modification]: Extract content if output is a tuple
    #         output_content = extract_content(info['output'])
    #         spatial_str += id + ": " + output_content + "\n\n"
    #
    #     decision_few_shot = self.prompt_set.get_decision_few_shot()
    #     user_prompt = f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"
    #     return system_prompt, user_prompt

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self.llm.agen(message)
        # print(self.agent_name)
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response


@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = "", ):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                        **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            # [Modification]: Extract content if output is a tuple
            output_content = extract_content(info['output'])
            info_list.append(output_content)
        if len(info_list):
            output = info_list[-1]
        return output

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            # [Modification]: Extract content if output is a tuple
            output_content = extract_content(info['output'])
            info_list.append(output_content)
        if len(info_list):
            output = info_list[-1]
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = "", ):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                        **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            # [Modification]: Extract content if output is a tuple
            output_content = extract_content(info['output'])
            processed_output = self.prompt_set.postprocess_answer(output_content)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            # [Modification]: Extract content if output is a tuple
            output_content = extract_content(info['output'])
            processed_output = self.prompt_set.postprocess_answer(output_content)
            # print(processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
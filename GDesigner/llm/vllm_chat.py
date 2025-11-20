import aiohttp
from typing import List, Union, Optional, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import os
from dotenv import load_dotenv

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

load_dotenv()
# vLLM 默认地址
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', 'http://localhost:8000/v1')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat_vllm(
        model: str,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2048
):
    # vLLM 兼容 OpenAI 的 Chat Completions 接口
    request_url = f"{VLLM_BASE_URL}/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer EMPTY'  # vLLM 本地运行通常不需要 Key，或设为 EMPTY
    }

    # 构造标准的 OpenAI 请求体
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API Error {response.status}: {error_text}")

            response_data = await response.json()
            # 提取回复内容
            content = response_data['choices'][0]['message']['content']

            # 简单的 token 统计（可选）
            prompt_text = "".join([str(m.get('content', '')) for m in messages])
            cost_count(prompt_text, content, model)

            return content


@LLMRegistry.register('VLLMChat')
class VLLMChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: Union[List[Message], List[Dict], str],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        # 核心修复：兼容处理 Dict 和 Message 对象
        msgs_dict = []
        if isinstance(messages, str):
            msgs_dict = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    # 如果已经是字典，直接使用
                    msgs_dict.append(m)
                elif hasattr(m, 'role') and hasattr(m, 'content'):
                    # 如果是 Message 对象，转换为字典
                    msgs_dict.append({"role": m.role, "content": m.content})
                else:
                    # 兜底处理，防止其他类型导致崩溃
                    print(f"Warning: Unknown message format: {type(m)}")
                    continue

        return await achat_vllm(
            model=self.model_name,
            messages=msgs_dict,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        # 同步接口暂不实现，或者用 asyncio.run 包装
        pass
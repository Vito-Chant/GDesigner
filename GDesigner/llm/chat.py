import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional, Any
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import async_timeout
from openai import AsyncOpenAI
import weave

# 保持你原有的包引用路径
from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

# 加载环境变量
load_dotenv()

# 配置变量
MINE_BASE_URL = os.getenv("MINE_BASE_URL", "")
MINE_API_KEYS = os.getenv("MINE_API_KEYS", "")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


# --- 2. 核心通用请求逻辑 (私有函数) ---

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def _generic_achat(
        model: str,
        messages: List[Dict],
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        **kwargs
) -> str:
    """
    通用的异步聊天请求函数，底层使用 AsyncOpenAI。
    vLLM 和 GPT 类模型均可复用此逻辑。
    """
    # 如果 api_key 为空（如本地 vLLM），OpenAI SDK 需要一个非空占位符
    if not api_key:
        api_key = "EMPTY"

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        # 300秒超时，防止请求无限挂起
        async with async_timeout.timeout(300):
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs  # 传递其他可能的参数，如 top_p
            )

        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            # 简单的 token 统计与计费 (保留原有逻辑)
            prompt_text = "".join([str(m.get('content', '')) for m in messages])
            cost_count(prompt_text, response_message, model)
            return response_message
        else:
            raise RuntimeError("Response content is not a string.")

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request for model {model}: {e}")


# --- 4. 具体实现类 ---

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """
    用于调用通用/商业 API (如 OpenAI, DeepSeek 等)
    """

    def __init__(self, model_name: str, max_tokens: int = 2048, temperature: float = 1.0):
        # 将 model_name 传给基类的 llm_name
        super().__init__(llm_name=model_name, max_tokens=max_tokens, temperature=temperature)

    @weave.op()
    async def agen(self, messages: Union[List[Message], List[Dict], str], **kwargs) -> str:
        # 1. 预处理消息
        formatted_msgs = self._preprocess_messages(messages)

        # 2. 参数合并：优先使用 kwargs 中的参数，否则使用实例默认值
        req_temp = kwargs.get('temperature', self.temperature)
        req_max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # 3. 移除 kwargs 中已经处理过的参数，避免重复传给 API
        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}

        # 4. 发起请求
        return await _generic_achat(
            model=self.llm_name,
            messages=formatted_msgs,
            api_key=MINE_API_KEYS,
            base_url=MINE_BASE_URL,
            temperature=req_temp,
            max_tokens=req_max_tokens,
            **api_kwargs
        )


@LLMRegistry.register('VLLMChat')
class VLLMChat(LLM):
    """
    用于调用本地部署的 vLLM 模型
    """

    def __init__(self, model_name: str, max_tokens: int = 2048, temperature: float = 1.0):
        super().__init__(llm_name=model_name, max_tokens=max_tokens, temperature=temperature)

    @weave.op()
    async def agen(self, messages: Union[List[Message], List[Dict], str], **kwargs) -> str:
        # 1. 预处理消息
        formatted_msgs = self._preprocess_messages(messages)

        # 2. 参数合并
        req_temp = kwargs.get('temperature', self.temperature)
        req_max_tokens = kwargs.get('max_tokens', self.max_tokens)

        api_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}

        # 3. 发起请求 (使用 VLLM 的地址和空 Key)
        return await _generic_achat(
            model=self.llm_name,
            messages=formatted_msgs,
            api_key="EMPTY",
            base_url=VLLM_BASE_URL,
            temperature=req_temp,
            max_tokens=req_max_tokens,
            **api_kwargs
        )

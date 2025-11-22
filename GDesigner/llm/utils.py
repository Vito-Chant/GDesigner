from typing import List, Dict, Union
from GDesigner.llm.format import Message

from typing import List, Union, Dict


def convert_messages_to_dict(messages: Union[List[Message], List[Dict], str]) -> List[Dict]:
    """
    统一将输入消息转换为标准的字典列表格式 ({role, content})。

    合并特性：
    1. 支持单个字符串输入（自动转为 user 消息）。
    2. 支持 Message 对象列表（自动提取 role/content）。
    3. 支持字典列表（严格校验是否包含 role/content 键）。

    Args:
        messages: 输入消息，支持 str, List[Message], List[Dict]

    Returns:
        List[Dict]: 标准格式的消息列表 [{"role": "...", "content": "..."}]

    Raises:
        ValueError: 如果字典输入缺少 'role' 或 'content' 字段。
        TypeError: 如果列表中的元素既不是字典也不是 Message 对象。
    """
    # 1. 处理空值
    if not messages:
        return []

    # 2. 处理单个字符串输入 (来自原 _parse_messages 的灵活性)
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    # 3. 确保是列表
    if not isinstance(messages, list):
        raise TypeError(f"Messages must be a list or string, got {type(messages)}")

    msgs_dict = []

    # 逐个处理列表元素 (比检查 list[0] 更安全，支持混合类型)
    for i, m in enumerate(messages):
        # 情况 A: 字典类型 (加入原 convert_messages_to_dict 的校验逻辑)
        if isinstance(m, dict):
            if "role" not in m or "content" not in m:
                raise ValueError(f"Message dictionary at index {i} must contain both 'role' and 'content' fields")
            msgs_dict.append(m)

        # 情况 B: Message 对象 (使用 duck typing，兼容所有具备 role/content 属性的对象)
        elif hasattr(m, 'role') and hasattr(m, 'content'):
            msgs_dict.append({"role": m.role, "content": m.content})

        # 情况 C: 未知类型
        else:
            raise TypeError(f"Item at index {i} is neither a valid dictionary nor a Message object: {type(m)}")

    return msgs_dict

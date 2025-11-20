from typing import List, Dict, Union
from GDesigner.llm.format import Message


def convert_messages_to_dict(messages: Union[List[Message], List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Unify conversion of Message object list or dictionary list to standard dictionary format ({role, content})

    Args:
        messages: Input messages which can be either a list of Message objects or a list of dictionaries
                  Each Message object must have 'role' and 'content' attributes
                  Each dictionary must contain 'role' and 'content' keys

    Returns:
        List of dictionaries in standard format with 'role' and 'content' keys

    Raises:
        ValueError: If input dictionaries lack 'role' or 'content' fields
        TypeError: If input is not a list of Message objects or dictionaries
    """
    if not messages:
        return []
    # Handle list of Message objects
    if isinstance(messages[0], Message):
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    # Handle list of dictionaries (validate format)
    elif isinstance(messages[0], dict):
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Message dictionary must contain both 'role' and 'content' fields")
        return messages
    else:
        raise TypeError("Messages must be a list of Message objects or dictionaries")
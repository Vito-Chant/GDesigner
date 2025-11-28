import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig  # 新增引入


class ActionAdapter(nn.Module):
    def __init__(self, llm_name=None, input_dim=None, hidden_dim=128, action_dim=5, constraint_prompt=None):
        super().__init__()

        # 自动获取 input_dim 的逻辑
        if input_dim is None:
            if llm_name is not None:
                try:
                    # trust_remote_code=True 对于某些新模型（如 Qwen）是必须的
                    config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
                    input_dim = config.hidden_size
                    print(f"[ActionAdapter] Detected input_dim={input_dim} for {llm_name}")
                except Exception as e:
                    raise ValueError(f"无法根据 llm_name '{llm_name}' 自动获取 input_dim，请手动指定。错误: {e}")
            else:
                raise ValueError("初始化 ActionAdapter 时必须提供 llm_name 或 input_dim 其中之一。")

        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # self.constraint_prompt = {
        #     0: None,
        #     1: None,
        #     2: "The final result is",
        #     3: "In short, the answer is",
        #     4: "Let's analyze this step by step:\n"
        # }
        if constraint_prompt is not None:
            self.constraint_prompt = constraint_prompt
        else:
            self.constraint_prompt = {
                0: None,
                1: "",
                2: "\nConstraint: Answer strictly with the final result only.",
                3: "\nConstraint: Be concise.",
                4: "\nConstraint: Think step-by-step and provide detailed reasoning."
            }

    def forward(self, embedding_tensor):
        # 确保输入也在正确的设备上
        device = next(self.parameters()).device
        embedding_tensor = embedding_tensor.to(device)
        return self.net(embedding_tensor)

    def sample(self, embedding_tensor):
        logits = self.forward(embedding_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

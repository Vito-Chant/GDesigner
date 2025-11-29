# experiments/train_magrpo_mmlu.py

import torch
from typing import Iterator, List, Tuple
import pandas as pd
import numpy as np
import time
import asyncio
import copy

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.agents.analyze_agent import AnalyzeAgent


async def train(graph: Graph,
                dataset,
                num_iters: int = 100,
                num_rounds: int = 1,
                lr: float = 5e-5,  # Typically smaller LR for LLM adapters
                batch_size: int = 1,  # Keep small due to num_generations expansion
                num_generations: int = 8,  # Number of samples per prompt (Group size)
                efficiency_weight: int = 0.5,
                wandb_run=None
                ) -> None:
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                record = dataset[idx.item()]
                yield record

    loader = infinite_data_loader()
    max_idx_sum = graph.num_nodes * (list(graph.nodes.values())[0].adapter.action_dim - 1)

    # 1. Collect trainable parameters from Adapters in AnalyzeAgents
    trainable_params = []
    # Assuming graph has a way to iterate nodes, e.g., graph.nodes.values()
    # You might need to adjust 'graph.nodes.values()' based on your Graph implementation
    agents_found = 0
    for node in graph.nodes.values():
        if isinstance(node, AnalyzeAgent) and hasattr(node, 'adapter') and node.adapter is not None:
            trainable_params.extend(node.adapter.parameters())
            node.adapter.train()  # Set to train mode
            agents_found += 1

    if not trainable_params:
        print("Warning: No adapters found to train!")
        return

    print(f"Training Adapters for {agents_found} AnalyzeAgents...")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * '-')
        start_ts = time.time()

        batch_loss = 0.0
        optimizer.zero_grad()

        # We iterate over the batch. 
        # For GRPO, each "item" in the batch becomes a group of 'num_generations' samples.
        for _ in range(batch_size):
            record = next(loader)
            input_dict = dataset.record_to_input(record)
            correct_answer = dataset.record_to_target_answer(record)

            # 2. Parallel Generation (Rollout)
            # We need to run the graph 'num_generations' times for the SAME input
            tasks = []
            for _ in range(num_generations):
                # Deepcopy to ensure graph state/history doesn't leak between generations
                realized_graph = copy.deepcopy(graph)
                for node_id, realized_node in realized_graph.nodes.items():
                    original_node = graph.nodes[node_id]
                    # 检查是否是 AnalyzeAgent 且拥有 adapter
                    if hasattr(realized_node, 'adapter') and hasattr(original_node, 'adapter'):
                        # 直接引用原对象的 adapter
                        realized_node.adapter = original_node.adapter
                # Ensure shared components (like GCN or MLP if used) are referenced, 
                # but for AnalyzeAgent adapters, the parameters are shared via Pytorch references.
                # Crucially, we assume realized_graph.arun now returns (answer, log_prob)
                # because we updated AnalyzeAgent._async_execute.
                # Note: You need to ensure Graph.arun propagates the tuple return from the agent.
                tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

            # Wait for all generations to complete
            raw_results = await asyncio.gather(*tasks)
            # raw_results structure expected: List[(answer_str, log_prob_tensor)]

            # 3. Process Results & Compute Rewards
            rewards = []
            log_probs = []
            idx_sum = []

            for raw_res in raw_results:
                # Unpack result. 
                # If your Graph.arun returns a list of results (one per agent), handle appropriately.
                # Here assuming single output or aggregated output.
                if isinstance(raw_res, tuple) and len(raw_res) == 3:
                    ans, lp, idx = raw_res
                else:
                    # Fallback if arun wrapper modification is tricky, assumes raw_res is string
                    # But then we lose gradients. This path is invalid for training.
                    raise ValueError("Graph.arun must return (answer, log_prob) tuple for training.")

                processed_ans = dataset.postprocess_answer(ans)

                # Calculate Utility (Reward)
                accuracy = Accuracy()
                accuracy.update(processed_ans, correct_answer)
                acc_reward = accuracy.get()  # 1.0 or 0.0 usually
                eff_reward = efficiency_weight * (max_idx_sum - idx) / max_idx_sum
                total_reward = acc_reward + eff_reward

                rewards.append(total_reward)
                log_probs.append(lp)
                idx_sum.append(idx)

            # Convert to tensors
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            # Stack log_probs (ensure they are on the same device)
            if len(log_probs) > 0:
                # Ensure all log_probs are tensors, handle cases where no action was taken (0.0)
                device = log_probs[0].device if isinstance(log_probs[0], torch.Tensor) else 'cpu'
                log_probs_tensor = torch.stack([
                    lp if isinstance(lp, torch.Tensor) else torch.tensor(0.0, device=device, requires_grad=True)
                    for lp in log_probs
                ])
            else:
                continue

            # 4. Compute GRPO Advantages
            # Baseline is the mean reward of the group
            baseline = rewards_tensor.mean()
            advantages = (rewards_tensor.to(log_probs_tensor.device) - baseline.to(log_probs_tensor.device)) / (
                        rewards_tensor.std() + 1e-8)

            # 5. Compute Loss
            # Loss = - Advantage * LogProb
            # We average over the generations (group)
            single_sample_loss = - (advantages * log_probs_tensor).mean()

            # Accumulate loss (if batch_size > 1)
            # We divide by batch_size here effectively by accumulating and backwarding once
            (single_sample_loss / batch_size).backward()

            batch_loss += single_sample_loss.item()

            print(f"  > Record Answers: {[dataset.postprocess_answer(r[0]) for r in raw_results]}")
            print(f"  > Correct: {correct_answer} | Advantages: {advantages}")
            print(f"  > idx_sum: {sum(idx_sum)}")

            if wandb_run is not None:
                wandb_run.log({"train/sample_loss": single_sample_loss.item()})
                wandb_run.log({"train/sample_idx_sum": sum(idx_sum)})

        # 6. Update Parameters
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        print(f"Batch time {time.time() - start_ts:.3f}")
        print("Iter Loss:", batch_loss)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value / 1000} k")
        print(f"CompletionTokens {CompletionTokens.instance().value / 1000} k")

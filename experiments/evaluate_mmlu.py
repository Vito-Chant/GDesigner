import os
import json
import math
import time
import asyncio
from typing import Union, Literal, Optional, Iterator, List, Any, Dict
from tqdm import tqdm
import copy

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens


async def evaluate(
        graph: Graph,
        dataset,
        num_rounds: int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        **kwargs
) -> float:
    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    # [修改点1] 删除了 graph.gcn.eval()，因为 gcn 已经不存在了

    accuracy = Accuracy()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        start_ts = time.time()
        answer_log_probs = []

        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            # [修改点2] 删除了 realized_graph.gcn = graph.gcn 和 realized_graph.mlp = graph.mlp
            # 因为 Graph 中已经移除了这两个属性

            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)

        # arun 现在返回 (final_answers, total_agent_log_prob)
        # 这里的解包逻辑依然适用，raw_answers 获取第一个元素（答案列表），log_probs 获取第二个元素（log概率）
        # 虽然 evaluate 阶段我们主要关心 accuracy，log_probs 在这里暂时用不到，但为了兼容性保留解包
        raw_answers, log_probs, idx_sum = zip(*raw_results)

        for raw_answer, record in zip(raw_answers, record_batch):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            accuracy.update(answer, correct_answer)

        accuracy.print()
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value / 1000} k")
        print(f"CompletionTokens {CompletionTokens.instance().value / 1000} k")

    accuracy.print()
    print("Done!")

    # 稍微加了个 get 判断防止 key error，原逻辑保持不变
    if kwargs.get('wandb_run') is not None:
        kwargs['wandb_run'].log({"Ptok": PromptTokens.instance().value / 1000})
        kwargs['wandb_run'].log({"Ctok": CompletionTokens.instance().value / 1000})

    return accuracy.get()


# dump_eval_results 保持不变
def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
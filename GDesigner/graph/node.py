import shortuuid
from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import warnings
import asyncio
from GDesigner.llm.adapter import ActionAdapter
import torch


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self,
                 id: Optional[str],
                 agent_name: str = "",
                 domain: str = "",
                 llm_name: str = "",
                 shared_adapter=None,
                 ):
        """
        Initializes a new Node instance.
        """
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str, List[Any]] = {'inputs': [], 'outputs': [], 'raw_inputs': []}
        # if shared_adapter:
        #     self.adapter = shared_adapter
        # else:
        #     self.adapter = ActionAdapter(llm_name=llm_name)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.adapter.to(device=device)

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st == 'temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st == 'temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []

    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def get_spatial_info(self) -> Dict[str, Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs) == 0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role": predecessor.role, "output": predecessor_output}

        return spatial_info

    def get_temporal_info(self) -> Dict[str, Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs) == 0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role": predecessor.role, "output": predecessor_output}

        return temporal_info

    def execute(self, input: Any, **kwargs):
        self.outputs = []
        spatial_info: Dict[str, Dict] = self.get_spatial_info()
        temporal_info: Dict[str, Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    async def async_execute(self, input: Any, **kwargs):

        self.outputs = []
        spatial_info: Dict[str, Any] = self.get_spatial_info()
        temporal_info: Dict[str, Any] = self.get_temporal_info()
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    # async def async_execute(self, input: Any, **kwargs):
    #     self.outputs = []
    #     spatial_info: Dict[str, Any] = self.get_spatial_info()
    #     temporal_info: Dict[str, Any] = self.get_temporal_info()
    #
    #     # 默认模式 (如果 Adapter 不存在或出错，兜底使用详细模式)
    #     mode = "Level 3"
    #
    #     # === [新增] Adapter 介入逻辑 (CTDE Decentralized Execution) ===
    #     # 只有当 Agent 拥有 adapter 且连接了支持 embedding 的 LLM 时才触发
    #     if hasattr(self, 'adapter') and self.adapter is not None:
    #         try:
    #             # 1. 构建观察状态 (Observation)
    #             # 简单的状态表征 = 当前任务 (Task) + 邻居的发言 (Context)
    #             system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info)
    #             message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    #
    #             # 2. 获取特征 (调用 vLLM 服务端计算 Embedding)
    #             # 注意：这里调用的是我们新加的 get_embedding 接口
    #             embedding_list = await self.llm.get_embedding(full_state_prompt)
    #
    #             # 3. Adapter 决策 (本地 PyTorch 计算)
    #             # 确保 Tensor 在 Adapter 所在的设备上 (GPU)
    #             device = next(self.adapter.parameters()).device
    #             emb_tensor = torch.tensor(embedding_list, dtype=torch.float32, device=device).unsqueeze(0)
    #
    #             # 采样动作 (Sample Action)
    #             action_idx, log_prob = self.adapter.sample(emb_tensor)
    #
    #             # 4. 记录训练数据 (用于后续 MARL 更新)
    #             self.current_log_prob = log_prob
    #             self.current_action = action_idx.item()
    #
    #             # 5. 动作映射 (Action Mapping)
    #             # 动作空间定义: 0->Silence, 1->Label Only, 2->Concise, 3->Detailed
    #             if self.current_action == 0:
    #                 # [关键] Silence 动作：直接软剪枝
    #                 # 不调用 _async_execute，不消耗 LLM 生成资源
    #                 # 返回空字符串以保持图拓扑连通性，但没有任何信息量
    #                 self.outputs = [""]
    #                 return self.outputs
    #
    #             mode_map = {1: "Level 1", 2: "Level 2", 3: "Level 3"}
    #             mode = mode_map.get(self.current_action, "Level 3")
    #
    #         except Exception as e:
    #             # 容错处理：如果 Adapter 挂了（比如显存 OOM），回退到默认执行
    #             logging.warning(f"Adapter failed for node {self.id}: {e}. Fallback to default execution.")
    #
    #     # 将决策得到的 mode 传入 kwargs
    #     # 子类 (如 MathSolver) 的 _async_execute 需要从 kwargs 中读取 mode 并据此构建 Prompt
    #     kwargs['mode'] = mode
    #
    #     # 执行具体的节点逻辑 (LLM Generation)
    #     tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
    #     results = await asyncio.gather(*tasks, return_exceptions=False)
    #     for result in results:
    #         if not isinstance(result, list):
    #             result = [result]
    #         self.outputs.extend(result)
    #     return self.outputs

    @abstractmethod
    def _execute(self, input: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                             **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                        **kwargs) -> List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """

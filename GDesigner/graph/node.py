import shortuuid
from typing import List, Any, Optional, Dict, Union, Tuple
from abc import ABC, abstractmethod
import warnings
import asyncio
from GDesigner.llm.adapter import ActionAdapter
import torch


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    v4.3 更新: 支持 KV Cache 复用
    - async_execute 可返回 str 或 (str, List[Dict])
    - 子类可通过返回对话历史来启用 KV Cache
    """

    def __init__(
            self,
            id: Optional[str],
            agent_name: str = "",
            domain: str = "",
            llm_name: str = "",
            shared_adapter=None,
    ):
        """初始化节点"""
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
        """Return a dict that maps id to info."""
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

    async def async_execute(self, input: Any, **kwargs) -> Union[List[Any], Tuple[List[Any], Any]]:
        """
        异步执行节点 (v4.3 KV Cache 兼容版)

        **返回值兼容性**:
        - 如果子类的 _async_execute 返回 str: 返回 [str]
        - 如果子类的 _async_execute 返回 (str, messages): 返回 ([str], messages)

        Returns:
            - List[Any]: 传统模式（向后兼容）
            - Tuple[List[Any], Any]: KV Cache 模式 (outputs, extra_data)
        """
        self.outputs = []
        spatial_info: Dict[str, Any] = self.get_spatial_info()
        temporal_info: Dict[str, Any] = self.get_temporal_info()

        # 创建异步任务
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # **关键逻辑：检测返回值类型**
        extra_data = None
        for result in results:
            # 情况1: 子类返回元组 (content, extra_data)
            if isinstance(result, tuple) and len(result) == 2:
                content, extra_data = result
                if not isinstance(content, list):
                    content = [content]
                self.outputs.extend(content)
            # 情况2: 传统单值返回
            else:
                if not isinstance(result, list):
                    result = [result]
                self.outputs.extend(result)

        # **返回值适配**
        if extra_data is not None:
            return self.outputs, extra_data  # KV Cache 模式
        else:
            return self.outputs  # 传统模式

    @abstractmethod
    def _execute(self, input: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        """To be overriden by the descendant class"""
        """Use the processed input to get the result"""

    @abstractmethod
    async def _async_execute(
            self,
            input: List[Any],
            spatial_info: Dict[str, Any],
            temporal_info: Dict[str, Any],
            **kwargs
    ) -> Union[str, Tuple[str, Any]]:
        """
        To be overriden by the descendant class

        **v4.3 返回值约定**:
        - 传统模式: return str
        - KV Cache 模式: return (str, messages)
        """

    @abstractmethod
    def _process_inputs(self, raw_inputs: List[Any], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any],
                        **kwargs) -> List[Any]:
        """To be overriden by the descendant class"""
        """Process the raw_inputs(most of the time is a List[Dict])"""
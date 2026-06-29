from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast

from pydantic import ConfigDict, Field, field_serializer, field_validator
from pydantic.alias_generators import to_camel

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology, TopologySnapshot
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    NodeIdentity,
    NodeNetworkInfo,
    NodeRdmaCtlStatus,
    NodeThunderboltInfo,
    SystemPerformanceProfile,
    ThunderboltBridgeStatus,
)
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.downloads import DownloadProgress
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.pydantic_ext import FrozenModel


class State(FrozenModel):
    """此說明已翻譯為繁體中文。

    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    此說明已翻譯為繁體中文。
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        extra="forbid",
        # 已翻譯註解。
        strict=True,
        arbitrary_types_allowed=True,
    )
    instances: Mapping[InstanceId, Instance] = {}
    runners: Mapping[RunnerId, RunnerStatus] = {}
    downloads: Mapping[NodeId, Sequence[DownloadProgress]] = {}
    tasks: Mapping[TaskId, Task] = {}
    last_seen: Mapping[NodeId, datetime] = {}
    topology: Topology = Field(default_factory=Topology)
    last_event_applied_idx: int = Field(default=-1, ge=-1)

    # 已翻譯註解。
    node_identities: Mapping[NodeId, NodeIdentity] = {}
    node_memory: Mapping[NodeId, MemoryUsage] = {}
    node_disk: Mapping[NodeId, DiskUsage] = {}
    node_system: Mapping[NodeId, SystemPerformanceProfile] = {}
    node_network: Mapping[NodeId, NodeNetworkInfo] = {}
    node_thunderbolt: Mapping[NodeId, NodeThunderboltInfo] = {}
    node_thunderbolt_bridge: Mapping[NodeId, ThunderboltBridgeStatus] = {}
    node_rdma_ctl: Mapping[NodeId, NodeRdmaCtlStatus] = {}
    node_backends: Mapping[NodeId, list[Backend]] = {}

    # 已翻譯註解。
    thunderbolt_bridge_cycles: Sequence[Sequence[NodeId]] = []

    instance_links: Mapping[InstanceLinkId, InstanceLink] = {}
    prefill_server_ports: Mapping[RunnerId, int] = {}

    # 已翻譯註解。
    custom_model_cards: Mapping[ModelId, ModelCard] = {}

    @field_serializer("topology", mode="plain")
    def _encode_topology(self, value: Topology) -> TopologySnapshot:
        return value.to_snapshot()

    @field_validator("topology", mode="before")
    @classmethod
    def _deserialize_topology(cls, value: object) -> Topology:  # noqa: D401 – Pydantic validator signature
        """此說明已翻譯為繁體中文。

        此說明已翻譯為繁體中文。
        此說明已翻譯為繁體中文。
        """

        if isinstance(value, Topology):
            return value

        if isinstance(value, Mapping):  # 已翻譯註解。
            snapshot = TopologySnapshot(**cast(dict[str, Any], value))  # type: ignore[arg-type]
            return Topology.from_snapshot(snapshot)

        raise TypeError("Invalid representation for Topology field in State")

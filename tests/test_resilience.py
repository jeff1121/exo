# type: ignore
"""韌性測試：斷線/重連節點並驗證叢集恢復能力。

執行方式：
    uv run pytest tests/test_resilience.py -v
"""

from __future__ import annotations

import pytest
from exo_tools.cluster import Thunderbolt
from exo_tools.harness import Comm, Sharding, cleanup_all_instances, place_instance

from .framework import DEFAULT_MODEL, InstanceSpec


@pytest.mark.cluster(count=2, thunderbolt=Thunderbolt.A2A)
@pytest.mark.instance(
    DEFAULT_MODEL, sharding=Sharding.PIPELINE, comm=Comm.RING, min_nodes=2
)
def test_node_recovery(session):
    """完整的斷線/重連循環。

    1. Place a 2-node instance, verify inference
    2. Disconnect one node
    3. Place a 1-node instance on remaining node, verify inference
    4. Reconnect the stopped node, wait for the cluster to reform
    5. Place a 2-node instance again, verify inference
    """
    # --- 階段 1：2 節點推論 ---
    resp = session.chat("Hello")
    assert len(resp) > 0

    # --- 階段 2：中斷一個節點 ---
    session.disconnect_node(1)
    session.wait_ready(60)

    # 清理目前已失效的 2 節點 instance
    cleanup_all_instances(session.client)

    # --- 階段 3：在剩餘節點上進行 1 節點推論 ---
    place_instance(session.client, DEFAULT_MODEL, min_nodes=1)
    session.instance_spec = InstanceSpec(model_id=DEFAULT_MODEL, min_nodes=1)
    resp = session.chat("Hello")
    assert len(resp) > 0

    # --- 階段 4：重連並恢復 2 節點叢集 ---
    cleanup_all_instances(session.client)
    session.reconnect_node(1)
    session.wait_ready(60)

    # --- 階段 5：再次進行 2 節點推論 ---
    place_instance(session.client, DEFAULT_MODEL, min_nodes=2)
    session.instance_spec = InstanceSpec(model_id=DEFAULT_MODEL, min_nodes=2)
    resp = session.chat("Hello again")
    assert len(resp) > 0

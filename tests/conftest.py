# type: ignore
"""使用 marker 驅動的 exo 整合測試 Pytest 設定。

測試作者可透過 marker 宣告需求：

    @pytest.mark.cluster(count=2, thunderbolt='a2a')
    @pytest.mark.instance('mlx-community/Llama-3.2-1B-Instruct-4bit',
                          sharding='tensor', comm='jaccl')
    def test_jaccl_inference(session):
        resp = session.chat('What is 2+2?')
        assert '4' in resp

叢集會以 `ClusterSpec` 快取；相同 cluster_spec 的測試會共用部署。
每個測試會建立自己的 instance（對應 `@pytest.mark.instance`），
並在測試後清理。

執行方式：
    uv run pytest tests/ -v
    uv run pytest tests/ -v --hosts s2,s4,s9,s10
"""

from __future__ import annotations

import contextlib
import json

import pytest
from exo_tools.cluster import ClusterInfo, EcoSession
from exo_tools.harness import cleanup_all_instances, place_instance

from .framework import (
    ClusterSpec,
    Session,
    parse_cluster_marker,
    parse_instance_marker,
)

# 整個測試程序共用單一 eco session。
eco = EcoSession(user_prefix="test")

# 以 ClusterSpec 為鍵的叢集快取——相同規格的測試共用同一個部署。
# 於 session teardown 時清空。
_cluster_cache: dict[ClusterSpec, ClusterInfo] = {}


def pytest_addoption(parser):
    parser.addoption(
        "--hosts",
        default=None,
        help="Comma-separated list of hosts (e.g. s2,s4,s9,s10). "
        "Overrides constraint-based reservation.",
    )


def pytest_configure(config):
    """註冊自訂 markers。"""
    config.addinivalue_line(
        "markers",
        "cluster(count=N, thunderbolt=Thunderbolt|None, min_memory=GB, chip=PATTERN): "
        "declare cluster requirements for a test",
    )
    config.addinivalue_line(
        "markers",
        "instance(model_id, sharding=Sharding, comm=Comm, min_nodes=N): "
        "declare instance placement for a test",
    )


def pytest_report_header(config):
    """顯示本次測試 session 使用的 eco 使用者與主機。"""
    hosts = config.getoption("--hosts")
    lines = [f"eco user: {eco.user}"]
    if hosts:
        lines.append(f"hosts override: {hosts}")
    return lines


@pytest.fixture(scope="session")
def _host_pool(request) -> list[str] | None:
    raw = request.config.getoption("--hosts")
    if raw:
        return [h.strip() for h in raw.split(",") if h.strip()]
    return None


@pytest.fixture
def session(request, _host_pool) -> Session:
    """每個測試的 fixture，提供符合該測試 markers 的 Session。

    Reads @pytest.mark.cluster and @pytest.mark.instance from the test, deploys
    a matching cluster (cached across tests with the same spec), places the
    model, and yields a Session for the test to interact with. Cleans up the
    instance after the test, and invalidates the cluster cache if the test
    left nodes disconnected.
    """
    cluster_marker = request.node.get_closest_marker("cluster")
    instance_marker = request.node.get_closest_marker("instance")

    cluster_spec = parse_cluster_marker(cluster_marker)
    instance_spec = parse_instance_marker(instance_marker)

    # 部署或重用符合規格的叢集
    cluster = _cluster_cache.get(cluster_spec)
    if cluster is None:
        if _host_pool:
            cluster = eco.start_deploy(
                hosts=_host_pool[: cluster_spec.count], wait=True
            )
        else:
            cluster = eco.start_deploy(
                count=cluster_spec.count,
                thunderbolt=cluster_spec.thunderbolt,
                chip=cluster_spec.chip,
                min_memory_gb=cluster_spec.min_memory_gb,
                wait=True,
            )
        _cluster_cache[cluster_spec] = cluster

    # 若測試有指定 instance，則為此測試建立一個 instance
    instance_id = None
    if instance_spec is not None:
        client = cluster.make_client()
        instance_id = place_instance(
            client,
            instance_spec.model_id,
            sharding=instance_spec.sharding,
            comm=instance_spec.comm,
            min_nodes=instance_spec.min_nodes,
        )

    sess = Session(
        cluster=cluster,
        eco=eco,
        instance_spec=instance_spec,
        instance_id=instance_id,
    )

    yield sess

    # ---- 收尾階段 ----

    # 若測試結束後仍有節點中斷，則使叢集快取失效並
    # 停止叢集，讓下一個測試重新部署。
    if sess._stopped_hosts:
        _cluster_cache.pop(cluster_spec, None)
        with contextlib.suppress(Exception):
            eco.stop(sess.cluster.hosts)
        return

    # 否則，清理測試期間建立的所有 instances
    with contextlib.suppress(Exception):
        cleanup_all_instances(sess.client)


# ---------------------------------------------------------------------------
# Session 層級收尾——停止所有快取的叢集
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _teardown_clusters():
    yield
    for cluster in _cluster_cache.values():
        with contextlib.suppress(Exception):
            eco.stop(cluster.hosts)
    _cluster_cache.clear()


def pytest_runtest_makereport(item, call):
    """測試失敗時，將叢集日誌附加到測試報告。"""
    if call.when != "call" or call.excinfo is None:
        return

    sess = item.funcargs.get("session")
    if sess is None:
        return
    try:
        logs = eco.logs(sess.cluster.hosts, lines=200)
        item.add_report_section("call", "Cluster Logs", json.dumps(logs, indent=2))
    except Exception:
        pass

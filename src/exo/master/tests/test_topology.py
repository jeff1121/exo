import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.topology import Connection, SocketConnection


@pytest.fixture
def topology() -> Topology:
    return Topology()


@pytest.fixture
def socket_connection() -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/1235"),
    )


def test_add_node(topology: Topology):
    # 已翻譯註解。
    node_id = NodeId()

    # 已翻譯註解。
    topology.add_node(node_id)

    # 已翻譯註解。
    assert topology.node_is_leaf(node_id)


def test_add_connection(topology: Topology, socket_connection: SocketConnection):
    # 已翻譯註解。
    node_a = NodeId()
    node_b = NodeId()
    connection = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(connection)

    # 已翻譯註解。
    data = list(topology.list_connections())

    # 已翻譯註解。
    assert data == [connection]

    assert topology.node_is_leaf(node_a)
    assert topology.node_is_leaf(node_b)


def test_remove_connection_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    # 已翻譯註解。
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)

    # 已翻譯註解。
    topology.remove_connection(conn)

    # 已翻譯註解。
    assert list(topology.get_all_connections_between(node_a, node_b)) == []


def test_remove_node_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    # 已翻譯註解。
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    # 已翻譯註解。
    topology.remove_node(node_b)

    # 已翻譯註解。
    assert list(topology.out_edges(node_a)) == []


def test_list_nodes(topology: Topology, socket_connection: SocketConnection):
    # 已翻譯註解。
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    # 已翻譯註解。
    nodes = list(topology.list_nodes())

    # 已翻譯註解。
    assert len(nodes) == 2
    assert all(isinstance(node, NodeId) for node in nodes)
    assert set(node for node in nodes) == set([node_a, node_b])

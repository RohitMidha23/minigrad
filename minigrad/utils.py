from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root):
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        graph.node(
            name=uid, label="{%s | data %.3f}" % (n.label, n.data), shape="record"
        )
        if n._op:
            graph.node(name=uid + n._op, label=n._op)
            graph.edge(id + n._op, n._op)
    for v1, v2 in edges:
        graph.edge(str(id(v1)), str(id(v2)) + v2._op)

    return graph


def topological_sort(root):
    topo, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(root)
    return topo

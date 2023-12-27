import math
from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Optional

from pyspark import RDD, SparkContext

from common import (
    Edge,
    Vertex,
    edges_csv_path,
    edges_to_csv,
    get_spark_context,
    mst_csv_path,
    vertices_csv_path,
)


def parse_vertex_line(csv_line: str) -> Vertex:
    x, y = map(float, csv_line.split(","))
    return (x, y)


def parse_edge_line(csv_line: str) -> Edge:
    p1x, p1y, p2x, p2y, w = map(float, csv_line.split(","))
    return (((p1x, p1y), (p2x, p2y)), w)


def get_vertices(sc: SparkContext, dataset: str) -> RDD:
    vertices_path = vertices_csv_path(dataset)
    return sc.textFile(vertices_path).map(parse_vertex_line)


def get_edges(sc: SparkContext, dataset: str) -> RDD:
    edges_path = edges_csv_path(dataset)
    return sc.textFile(edges_path).map(parse_edge_line)


def kruskal(vertices: list[Vertex], edges: list[Edge]) -> list[Edge]:
    edges = sorted(edges, key=lambda e: e[1])

    parent: dict[Vertex, Vertex] = {}
    rank: dict[Vertex, int] = {}

    def make_set(vertex: Vertex):
        parent[vertex] = vertex
        rank[vertex] = 0

    def find_set(vertex: Vertex) -> Vertex:
        while parent[vertex] != vertex:
            vertex, parent[vertex] = parent[vertex], parent[parent[vertex]]

        return vertex

    def union_set(x: Vertex, y: Vertex):
        x = find_set(x)
        y = find_set(y)

        if x == y:
            return

        if rank[x] < rank[y]:
            x, y = y, x

        parent[y] = x

        if rank[x] == rank[y]:
            rank[x] = rank[x] + 1

    result = []

    for vertex in vertices:
        make_set(vertex)

    for edge in edges:
        (u, v), _ = edge
        x = find_set(u)
        y = find_set(v)

        if x != y:
            result.append(edge)
            union_set(x, y)

    return result


def compute_coreset(vertices: list[Vertex]):
    def _compute_coreset(edges: Iterable[Edge]) -> list[Edge]:
        # Convert edges to a list to prevent
        # the iterable from being consumed multiple times.
        edges = list(edges)

        # We use Kruskal's algorithm to compute the MST on each machine
        # since it has the nice property that if the graph is not connected
        # (which is might be the case)
        # it computes the minimum spanning forest.
        return kruskal(vertices, edges)

    return _compute_coreset


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-n", type=int)
    parser.add_argument("-m", type=int)
    args = parser.parse_args()

    dataset: str = args.dataset
    num_vertices: Optional[int] = args.n
    num_edges: Optional[int] = args.m

    sc = get_spark_context("MST Edge Sampling")
    vertices = get_vertices(sc, dataset)

    # The user can give n to speed up the process. Otherwise it is computed.
    if num_vertices is None:
        num_vertices = vertices.count()

    edges = get_edges(sc, dataset)

    # Same as for num_vertices. If not given it is computed
    if num_edges is None:
        num_edges = edges.count()

    # Derive the memory per machine from the initial number of partitions spark creates.
    num_machines = edges.getNumPartitions()
    memory_per_machine = math.ceil(num_edges / num_machines)

    print(f"{num_vertices=}, {num_edges=}, {num_machines=}, {memory_per_machine=}")

    # Give each node a read-only copy of the vertices.
    vertices = vertices.collect()
    vertices = sc.broadcast(vertices)

    first = True

    while num_edges > memory_per_machine:
        # In the first iteration we can skip shuffling the data.
        # This is already done when reading in the RDD.
        if not first:
            num_machines = math.ceil(num_edges / memory_per_machine)

            # Reduce the number of partitions and
            # shuffle the edges between the partitions.
            edges = edges.coalesce(num_machines, shuffle=True)
        else:
            first = False

        # On each machine a.k.a. partition compute the MST.
        edges = edges.mapPartitions(compute_coreset(vertices.value))
        num_edges = edges.count()

    # All the edges fit on one machine so collect them and compute the final MST.
    edges = edges.collect()
    result = kruskal(vertices.value, edges)

    edges_to_csv(mst_csv_path(dataset), result)

    print(len(result))


if __name__ == "__main__":
    main()

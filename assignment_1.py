from collections.abc import Iterable
from pathlib import Path
from pprint import pprint
from typing import Final

from pyspark import RDD, SparkContext

from util import get_spark_context

Point = tuple[float, float]
Vertex = tuple[Point, float]
Edge = tuple[tuple[Point, Point], float]

DATA_DIR: Final[Path] = Path("data")
VERTICES_CSV: Final[str] = "v.csv"
EDGES_CSV: Final[str] = "e.csv"


def parse_float(s: str) -> float:
    """The values are in the form '"123,45"' but need to be in the form '123,45' in order to be parsed by `float`"""
    return float(s[1:-1])


def parse_2d_vertices(csv_line: str) -> Vertex:
    x, y, r = map(parse_float, csv_line.split(","))
    return ((x, y), r)


def parse_2d_edges(csv_line: str) -> Edge:
    p1x, p1y, p2x, p2y, w = map(parse_float, csv_line.split(","))
    return (((p1x, p1y), (p2x, p2y)), w)


def get_vertices_and_edges(sc: SparkContext, dataset: str) -> tuple[RDD, RDD]:
    v = sc.textFile(str(DATA_DIR / dataset / VERTICES_CSV)).map(parse_2d_vertices)
    e = sc.textFile(str(DATA_DIR / dataset / EDGES_CSV)).map(parse_2d_edges)

    return v, e


def compute_mst(vertices: dict[Point, float], edges: dict[Edge, float]):
    ...


def compute_local_mst(vertices: dict[Point, float]):
    def _compute_mst(edges: Iterable[Edge]) -> Iterable[Edge]:
        # The edges can form multiple disjunct subgraphs.
        # Thus we find all connected components and compute the MSTs for all of them.
        # Then we return the union of the MSTs.
        ...

    return _compute_mst


def find_connected_components(edges: Iterable[Edge]) -> list[list[Edge]]:
    # We want to only iterate over the iterator once
    edges: dict[tuple[Point, Point], Edge] = {e[0]: e for e in edges}
    adj = {v: [] for e in edges for v in e}
    for edge in edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])

    visited = set()

    def dfs(v: Vertex, component: set[Edge]):
        visited.add(v)

        for n in adj[v]:
            edge = edges.get((v, n)) or edges[(n, v)]
            component.add(edge)

            if n not in visited:
                dfs(n, component)

    components = []

    for v in adj.keys():
        if v not in visited:
            component = set()
            dfs(v, component)
            components.append(list(component))

    return components


def main():
    sc = get_spark_context("assignment 1")
    v, e = get_vertices_and_edges(sc, "material")

    # Give each node a read-only copy of the vertices.
    vertices = v.collectAsMap()
    vertices = sc.broadcast(vertices)

    # edges = e.collect()

    # pprint(find_connected_components(edges))

    # pprint(e.glom().collect())

    pprint(e.mapPartitions(find_connected_components).collect())

    # num_partitions = e.getNumPartitions()

    # first = True

    # while e.count() > 10:
    #     # In the first iteration we can skip shuffling
    #     if not first:
    #         e = e.repartition(num_partitions)
    #     else:
    #         first = False

    #     e = e.mapPartitions(test)  # Compute local MSTs

    # Collect edges
    # Compute final MST


if __name__ == "__main__":
    main()

import csv
import math
from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Optional

from pyspark import RDD, SparkContext

from util import get_spark_context

Point = tuple[float, float]
Vertex = tuple[Point, float]
Edge = tuple[tuple[Point, Point], float]

DATA_DIR: Final[Path] = Path("data")
VERTICES_CSV: Final[str] = "v.csv"
EDGES_CSV: Final[str] = "e.csv"


def parse_float(s: str) -> float:
    # The values are in the form '"123,45"'
    # but need to be in the form '123,45'
    # in order to be parsed by `float`.
    return float(s[1:-1])


def parse_2d_vertices(csv_line: str) -> Vertex:
    x, y, r = map(parse_float, csv_line.split(","))
    return ((x, y), r)


def parse_2d_edges(csv_line: str) -> Edge:
    p1x, p1y, p2x, p2y, w = map(parse_float, csv_line.split(","))
    return (((p1x, p1y), (p2x, p2y)), w)


def get_vertices(sc: SparkContext, dataset: str) -> RDD:
    return sc.textFile(str(DATA_DIR / dataset / VERTICES_CSV)).map(parse_2d_vertices)


def get_edges(sc: SparkContext, dataset: str, num_partitions: int) -> RDD:
    return (
        sc.textFile(str(DATA_DIR / dataset / EDGES_CSV))
        .map(parse_2d_edges)
        .repartition(num_partitions)
    )


def kruskal(vertices: dict[Point, float], edges: list[Edge]) -> list[Edge]:
    edges = sorted(edges, key=lambda e: e[1])

    parent: dict[Point, Point] = {}
    rank: dict[Point, int] = {}

    def make_set(vertex: Point):
        parent[vertex] = vertex
        rank[vertex] = 0

    def find_set(vertex: Point) -> Point:
        while parent[vertex] != vertex:
            vertex, parent[vertex] = parent[vertex], parent[parent[vertex]]

        return vertex

    def union_set(x: Point, y: Point):
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


def compute_mst(vertices: dict[Point, float]):
    def _compute_mst(edges: Iterable[Edge]) -> list[Edge]:
        # Convert edges to a list to prevent
        # the iterable from being consumed multiple times.
        edges = list(edges)

        # We use Kruskal's algorithm to compute the MST on each machine
        # since it has the nice property that if the graph is not connected
        # (which is might be the case)
        # it computes the minimum spanning forest.
        return kruskal(vertices, edges)

    return _compute_mst


def update_weights(vertices: dict[Point, float]):
    def _update_weights(edge: Edge) -> Edge:
        # Since each vertex is actually a ball, the weight of the edge gets smaller
        # by the radii of the two endpoints.
        (u, v), w = edge
        w = w - vertices[u] - vertices[v]
        # However, if two balls overlap the distance between them is 0 and not negative.
        w = max(w, 0)

        return ((u, v), w)

    return _update_weights


def scale_radius(vertex: Vertex):
    p, r = vertex
    r = r 
    return (p, r)


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-e", type=float, default=0.2)
    parser.add_argument("-n", type=int)
    parser.add_argument("-m", type=int)
    args = parser.parse_args()

    dataset: str = args.dataset
    epsilon: float = args.e
    num_vertices: Optional[int] = args.n
    num_edges: Optional[int] = args.m

    sc = get_spark_context("assignment 1")
    vertices = get_vertices(sc, dataset).map(scale_radius)

    # The user can give n to speed up the process. Otherwise it is computed.
    if num_vertices is None:
        num_vertices = vertices.count()

    memory_per_machine = math.ceil(num_vertices ** (1 + epsilon))

    # Same as for num_vertices but if not given we assume that the graph is a clique.
    if num_edges is None:
        num_edges = (num_vertices * (num_vertices - 1)) // 2

    # Number of machines needed.
    num_machines = math.ceil(num_edges / memory_per_machine)

    edges = get_edges(sc, dataset, num_machines)

    print(f"{num_vertices=}, {num_edges=}, {num_machines=}, {memory_per_machine=}")

    # Give each node a read-only copy of the vertices.
    vertices = vertices.collectAsMap()
    vertices = sc.broadcast(vertices)

    # We first update all the edge weights depending on the radii of the endpoints
    edges = edges.map(update_weights(vertices.value))

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
        edges = edges.mapPartitions(compute_mst(vertices.value))
        num_edges = edges.count()

    # All the edges fit on one machine so collect them and compute the final MST.
    edges = edges.collect()
    result = kruskal(vertices.value, edges)

    dest_dir = Path(f"data/{args.dataset}")

    with open(dest_dir / "mst.csv", "w+", newline="") as mst_csv:
        mst_csv_writer = csv.writer(mst_csv, dialect="unix")
        for row in result:
            if row[1] != 0:
                print("found a distance different than 0", row[1])

            v1_x, v1_y = row[0][0][0], row[0][0][1]
            v2_x, v2_y = row[0][1][0], row[0][1][1]
            dist = row[1]

            mst_csv_writer.writerow([v1_x, v1_y, v2_x, v2_y, dist])

    print(len(result))


if __name__ == "__main__":
    main()

import math
import random
from argparse import ArgumentParser
from itertools import batched, product
from pprint import pprint
from typing import Optional

from common import Edge, Vertex, get_edges, get_spark_context, get_vertices, kruskal


def partition(vertices: list[Vertex], k: int) -> list[tuple[set[Vertex], set[Vertex]]]:
    # Don't modify original copy
    vertices = vertices.copy()
    n = len(vertices)
    random.shuffle(vertices)
    v = vertices.copy()
    random.shuffle(vertices)
    u = vertices.copy()

    batch_size = math.ceil(n / k)

    v = map(set, batched(v, batch_size))
    u = map(set, batched(u, batch_size))

    return list(product(v, u))


def induced(pair: tuple[tuple[tuple[Vertex], tuple[Vertex]], Edge]) -> bool:
    (v, u), ((x, y), _) = pair
    return x in v and y in u  # or (y in v and x in u)


def compute_coreset(
    pair: tuple[tuple[tuple[Vertex], tuple[Vertex]], list[Edge]]
) -> list[Edge]:
    (v, u), edges = pair
    vertices = list(set(v + u))
    return kruskal(vertices, edges)


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

    sc = get_spark_context("MST Vertex Sampling")

    vertices = get_vertices(sc, dataset)

    # The user can give m to speed up the process. Otherwise it is computed.
    if num_vertices is None:
        num_vertices = vertices.count()

    memory_per_machine = math.ceil(num_vertices ** (1 + epsilon))

    edges = get_edges(sc, dataset)

    # The user can give m to speed up the process. Otherwise it is computed.
    if num_edges is None:
        num_edges = edges.count()

    c = math.ceil(math.log(num_edges / num_vertices, num_vertices))

    print(f"{num_vertices=}, {num_edges=}, {c=}, {memory_per_machine=}")

    while num_edges > memory_per_machine:
        num_partitions = math.ceil(num_vertices ** ((c - epsilon) / 2))

        vertices = vertices.repartition(num_partitions)
        v = vertices.glom().map(tuple)
        vertices = vertices.repartition(num_partitions)
        u = vertices.glom().map(tuple)

        pairs = v.cartesian(u)
        pairs = pairs.cartesian(edges)
        pairs = pairs.filter(induced)
        pairs = pairs.groupByKey()
        pairs = pairs.mapValues(list)
        edges = pairs.flatMap(compute_coreset)

        c = (c - epsilon) / 2
        num_edges = edges.count()

    vertices = vertices.collect()
    edges = edges.collect()
    result = kruskal(vertices, edges)
    pprint(len(result))


if __name__ == "__main__":
    main()

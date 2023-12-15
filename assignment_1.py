from pathlib import Path
from typing import Final

from pyspark import RDD, SparkContext

from util import get_spark_context

DATA_DIR: Final[Path] = Path("data")
VERTICES_CSV: Final[str] = "v.csv"
EDGES_CSV: Final[str] = "e.csv"


def parse_float(s: str) -> float:
    """The values are in the form '"123,45"' but need to be in the form '123,45' in order to be parsed by `float`"""
    return float(s[1:-1])


def parse_2d_vertices(csv_line: str) -> tuple[tuple[float, float], float]:
    x, y, r = map(parse_float, csv_line.split(","))
    return ((x, y), r)


def parse_2d_edges(
    csv_line: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], float]:
    p1x, p1y, p2x, p2y, w = map(parse_float, csv_line.split(","))
    return (((p1x, p1y), (p2x, p2y)), w)


def get_vertices_and_edges(sc: SparkContext, dataset: str) -> tuple[RDD, RDD]:
    v = sc.textFile(str(DATA_DIR / dataset / VERTICES_CSV)).map(parse_2d_vertices)
    e = sc.textFile(str(DATA_DIR / dataset / EDGES_CSV)).map(parse_2d_edges)

    return v, e


def main():
    sc = get_spark_context("assignment 1")
    v, e = get_vertices_and_edges(sc, "circles-10000")

    # Give each node a read-only copy of the vertices.
    vertices = v.collectAsMap()
    vertices = sc.broadcast(vertices)

    while e.count() > 10:
        e = e.mapPartitions()  # Compute local MSTs
        # do shuffle

    # Collect edges
    # Compute final MST


if __name__ == "__main__":
    main()

import csv
from pathlib import Path
from typing import Final

from pyspark import SparkConf, SparkContext

DATA_DIR: Final[Path] = Path("data")
VERTICES_CSV: Final[Path] = Path("v.csv")
EDGES_CSV: Final[Path] = Path("e.csv")
MST_CSV: Final[Path] = Path("mst.csv")

Vertex = tuple[float, float]
Edge = tuple[tuple[Vertex, Vertex], float]


def vertices_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / VERTICES_CSV)


def edges_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / EDGES_CSV)


def mst_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / MST_CSV)


def flatten(edge: Edge) -> tuple[float, float, float, float, float]:
    (((p1x, p1y), (p2x, p2y)), w) = edge
    return (p1x, p1y, p2x, p2y, w)


def get_spark_context(app_name: str) -> SparkContext:
    conf = (
        SparkConf()
        .setAppName(app_name)
        .setMaster("local[*]")
        .set("spark.eventLog.enabled", "true")
    )

    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")

    return sc


def edges_to_csv(path: str, edges: list[Edge]):
    edges = map(flatten, edges)

    with open(path, "w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(edges)

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pyspark import SparkConf, SparkContext

DATA_DIR: Final[Path] = Path("data")
VERTICES_CSV: Final[Path] = Path("v.csv")
EDGES_CSV: Final[Path] = Path("e.csv")
MST_CSV: Final[Path] = Path("mst.csv")


def vertices_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / VERTICES_CSV)


def edges_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / EDGES_CSV)


def mst_csv_path(dataset: str) -> str:
    return str(DATA_DIR / dataset / MST_CSV)


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


@contextmanager
def csv_writer(path: str | Path):
    with open(path, "w+", newline="") as file:
        yield csv.writer(file)

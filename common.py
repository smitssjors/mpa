import csv
from contextlib import contextmanager
from pathlib import Path

from pyspark import SparkConf, SparkContext


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

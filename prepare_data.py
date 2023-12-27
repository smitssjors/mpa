import os

import numpy as np
import pandas as pd
from pyspark import SparkContext
from sklearn.datasets import fetch_california_housing

from common import Vertex, get_spark_context


def parse_vertex(line: str) -> Vertex:
    lat, long = map(float, line.split(","))
    return (lat, long)


def make_edge(pair: tuple[Vertex, Vertex]) -> str:
    p1, p2 = map(np.array, pair)
    weight = np.linalg.norm(p1 - p2)
    return ",".join(map(str, (pair[0][0], pair[0][1], pair[1][0], pair[1][1], weight)))


def generate_edges(sc: SparkContext, vertices_csv: str, out: str):
    vertices = sc.textFile(vertices_csv).map(parse_vertex)
    pairs = vertices.cartesian(vertices)
    pairs = pairs.filter(lambda pair: pair[0] < pair[1])
    pairs = pairs.repartition(24)
    edges = pairs.map(make_edge)
    edges.saveAsTextFile("data/temp")
    os.system(f"cat data/temp/part* > {out}")
    os.system("rm -rf data/temp")


def parse_latlong(row) -> pd.Series:
    c: str = row["geometry.coordinates"]
    lon, lat = map(float, c[1:-1].split(","))
    return pd.Series([lat, lon], index=["lat", "lon"])


def prepare_mcdonalds():
    df = pd.read_csv("data/mcdonalds/raw.csv")
    df = df.loc[df["properties.subDivision"] == "CA"].apply(parse_latlong, axis=1)
    df.to_csv("data/mcdonalds/v.csv", header=False, index=False)


def prepare_housing():
    dataset = fetch_california_housing(data_home="data/housing", as_frame=True)
    df: pd.DataFrame = dataset["frame"]
    df = df[["Latitude", "Longitude"]]
    df = df.drop_duplicates()
    df.to_csv("data/housing/v.csv", header=False, index=False)


def main():
    prepare_mcdonalds()
    prepare_housing()
    sc = get_spark_context("prepare")
    generate_edges(sc, "data/mcdonalds/v.csv", "data/mcdonalds/e.csv")
    generate_edges(sc, "data/housing/v.csv", "data/housing/e.csv")


if __name__ == "__main__":
    main()

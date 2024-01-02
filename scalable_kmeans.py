from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from pyspark import RDD, SparkContext

from common import get_spark_context, vertices_csv_path


def parse_point(csv_line: str) -> np.ndarray:
    return np.fromstring(csv_line, dtype=np.float64, sep=",", count=2)


def get_points(sc: SparkContext, dataset: str) -> RDD[np.ndarray]:
    path = vertices_csv_path(dataset)
    return sc.textFile(path).map(parse_point)


def closest_center(centers: np.ndarray):
    def closest_center(point: np.ndarray) -> tuple[int, np.ndarray]:
        distances = np.linalg.norm(centers - point, axis=1)
        i = np.argmin(distances)
        return i, point

    return closest_center


def temp(point: np.ndarray) -> np.ndarray:
    return np.append(point, 1)

def seq(agg: np.ndarray, point: np.ndarray) -> np.ndarray:
    agg += np.append(point, 1)
    return agg


def comb(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x += y
    return x


def main():
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    dataset: str = args.dataset
    k: int = args.k

    sc = get_spark_context("Scalable k-means")
    points = get_points(sc, dataset).repartition(12)

    centers = np.vstack(points.takeSample(False, k, 9))
    pprint(centers)
    changed = True

    while changed:
        centers = sc.broadcast(centers)
        closest = points.map(closest_center(centers.value))
        # combineByKey
        # aggByKey
        # mapPartitions with np

        closest = closest.combineByKey(temp, seq, comb)
        closest = closest.mapValues(lambda x: x[:2] / x[2])

        # sort by key
        # collect + sort
        closest = closest.collect()
        closest = sorted(closest, key=lambda x: x[0])
        closest = list(map(lambda x: x[1], closest))
        closest = np.vstack(closest)

        changed = not np.array_equal(centers.value, closest)
        centers.destroy()
        centers = closest
    
    pprint(centers)


if __name__ == "__main__":
    main()

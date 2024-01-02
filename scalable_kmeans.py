from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from pyspark import RDD, SparkContext

from common import get_spark_context, vertices_csv_path

Point = np.ndarray
PartialMean = tuple[Point, int]


def parse_point(csv_line: str) -> Point:
    return np.fromstring(csv_line, dtype=np.float64, sep=",", count=2)


def get_points(sc: SparkContext, dataset: str) -> RDD[Point]:
    path = vertices_csv_path(dataset)
    return sc.textFile(path).map(parse_point)


def closest_center(centers: np.ndarray):
    def closest_center(point: Point) -> tuple[int, Point]:
        distances = np.linalg.norm(centers - point, axis=1)
        i = np.argmin(distances)
        return i, point

    return closest_center


def seq(partial: PartialMean, point: Point) -> PartialMean:
    return partial[0] + point, partial[1] + 1


def comb(x: PartialMean, y: PartialMean) -> PartialMean:
    return x[0] + y[0], x[1] + y[1]


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

        closest = closest.combineByKey((np.zeros(2), 0), seq, comb)
        closest = closest.mapValues(lambda x: x[0] / x[1])

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

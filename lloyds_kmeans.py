import csv
from argparse import ArgumentParser

import numpy as np
from pyspark import RDD, SparkContext

from common import centers_csv_path, get_spark_context, vertices_csv_path

Point = np.ndarray


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


def save_centers(dataset: str, centers: np.ndarray):
    path = centers_csv_path(dataset)
    np.savetxt(path, centers, delimiter=",")


def main():
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("k", type=int)
    args = parser.parse_args()

    dataset: str = args.dataset
    k: int = args.k

    sc = get_spark_context("Lloyd's k-means")
    points = get_points(sc, dataset).repartition(12)

    centers = np.vstack(points.takeSample(False, k))
    changed = True

    while changed:
        # Compute the closest center for each point.
        # Point -> (index of closest center, point) pair.
        closest = points.map(closest_center(centers))

        # Compute the partial means for the new centers
        partial_mean = closest.aggregateByKey(
            (np.zeros(2), 0),
            lambda x, p: (x[0] + p, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
        )

        # Compute the actual new centers
        new_centers = partial_mean.mapValues(lambda x: x[0] / x[1])

        new_centers = new_centers.collect()
        new_centers = sorted(new_centers, key=lambda x: x[0])
        new_centers = list(map(lambda x: x[1], new_centers))
        new_centers = np.vstack(new_centers)

        changed = not np.array_equal(centers, new_centers)
        centers = new_centers

    save_centers(dataset, centers)


if __name__ == "__main__":
    main()

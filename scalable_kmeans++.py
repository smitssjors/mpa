from argparse import ArgumentParser
from typing import Iterable

import numpy as np
from pyspark import RDD, SparkContext

from common import get_spark_context, vertices_csv_path

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


def distance(centers: np.ndarray):
    def distance(point: Point) -> float:
        return np.min(np.square(np.linalg.norm(centers - point)))

    return distance


def compute_cost(points: RDD[Point], centers: np.ndarray) -> float:
    return points.map(distance(centers)).sum()


def sample(
    points: RDD[Point], centers: np.ndarray, l: float, cost: float
) -> np.ndarray:
    rng = np.random.default_rng()
    dist = distance(centers)

    def test(point: Point):
        return rng.random() < ((l * dist(point)) / cost)

    return np.vstack(points.filter(test).collect())


def k_meansbb(points: RDD[Point], k: int, l: float) -> np.ndarray:
    # Pick an initial center at random
    centers = np.vstack(points.takeSample(False, 1))
    cost = compute_cost(points, centers)

    for _ in range(int(np.log2(cost))):
        new_centers = sample(points, centers, l, cost)

        centers = np.concatenate([centers, new_centers])
        centers = np.unique(centers, axis=0)

        cost = compute_cost(points, centers)

    print(len(centers))


def lloyds(points: RDD[Point], centers: np.ndarray, k: int) -> np.ndarray:
    pass


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("k", type=int)
    parser.add_argument("-l", type=float, default=0.5)
    args = parser.parse_args()

    dataset: str = args.dataset
    k: int = args.k
    l: float = args.l

    # Oversampling factor
    l = k * l

    sc = get_spark_context("k-means|| -> k-means++ -> lloyd's")
    points = get_points(sc, dataset).repartition(12)

    initial_centers = k_meansbb(points, k, l)
    centers = lloyds(points, initial_centers, k)

    return

    initial_center = points.takeSample(False, 1)
    initial_cost = points.aggregate(
        0.0,
        lambda l, p: l + np.square(np.linalg.norm(initial_center - p)),
        lambda x, y: x + y,
    )

    print(initial_cost)

    return
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


if __name__ == "__main__":
    main()

import csv
import sys
import time
from argparse import ArgumentParser
from typing import Iterable, Optional

import numpy as np
from pyspark import RDD, SparkContext
from sklearn.cluster import kmeans_plusplus

from common import centers_csv_path, get_spark_context, vertices_csv_path

Point = np.ndarray
PointWithDist = tuple[Point, np.float64]


def parse_point(csv_line: str) -> Point:
    return np.fromstring(csv_line, dtype=np.float64, sep=",", count=2)


def get_points(sc: SparkContext, dataset: str) -> RDD[Point]:
    path = vertices_csv_path(dataset)
    return sc.textFile(path).map(parse_point)


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([a, b]), axis=0)


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("k", type=int)
    parser.add_argument("-l", type=float, default=0.5)
    parser.add_argument("-s", type=int)
    args = parser.parse_args()

    dataset: str = args.dataset
    k: int = args.k
    l: float = args.l
    seed: Optional[int] = args.s

    # Oversampling factor.
    l = k * l

    rng = np.random.default_rng(seed)

    sc = get_spark_context("k-means|| -> k-means++ -> lloyd's")
    points = get_points(sc, dataset).repartition(12)

    start_time = time.perf_counter_ns()

    ### Start of k-means||
    # Pick and initial center at random.
    centers = np.vstack(points.takeSample(False, 1, seed))

    def update_distances() -> RDD[PointWithDist]:
        def distance(point: Point) -> PointWithDist:
            dist = np.min(np.square(np.linalg.norm(centers - point)))
            return point, dist

        return points.map(distance)

    points_with_dist = update_distances()

    def compute_cost() -> float:
        return points_with_dist.values().sum()

    cost = compute_cost()

    def sample() -> np.ndarray:
        _seed = rng.integers(0, sys.maxsize)

        def filter(split: int, points: Iterable[PointWithDist]) -> Iterable[Point]:
            _rng = np.random.default_rng([split, _seed])

            for point in points:
                if _rng.random() < ((l * point[1]) / cost):
                    yield point[0]

        new_centers = points_with_dist.mapPartitionsWithIndex(filter)
        new_centers = new_centers.collect()

        # It can happen that we sample 0 centers.
        if not new_centers:
            return np.array([]).reshape((0, 2))

        return np.vstack(new_centers)

    for i in range(int(np.log2(cost))):
        # Skip this in the first iteration since it has already been done.
        if i != 0:
            points_with_dist = update_distances()
            cost = compute_cost()

        new_centers = sample()
        centers = union(centers, new_centers)

    # Compute the weights for each center
    def representative(point: Point) -> tuple[int]:
        distances = np.linalg.norm(centers - point, axis=1)
        return (np.argmin(distances),)

    weights = points.map(representative).countByKey()
    weights = np.fromiter(map(lambda x: x[1], sorted(weights.items())), dtype=int)

    ### k-means++ to cluster the points from k-means|| to k
    centers, _ = kmeans_plusplus(centers, k, random_state=seed, sample_weight=weights)

    ### Lloyds
    def closest_center(point: Point) -> tuple[int, Point]:
        distances = np.linalg.norm(centers - point, axis=1)
        i = np.argmin(distances)
        return i, point

    changed = True

    while changed:
        # Compute the closest center for each point.
        # Point -> (index of closest center, point) pair.
        closest = points.map(closest_center)

        # Compute the partial means for the new centers
        partial_mean = closest.aggregateByKey(
            (np.zeros(2), 0),
            lambda x, p: (x[0] + p, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
        )

        # Compute the actual new centers
        new_centers = partial_mean.mapValues(lambda x: x[0] / x[1])

        new_centers = new_centers.collect()
        new_centers = sorted(new_centers)
        new_centers = list(map(lambda x: x[1], new_centers))
        new_centers = np.vstack(new_centers)

        changed = not np.array_equal(centers, new_centers)
        centers = new_centers

    final_time = time.perf_counter_ns() - start_time

    with open("data/stats.csv", "a", newline="") as stats:
        writer = csv.writer(stats)
        dataset, n = dataset.split("-")
        writer.writerow((dataset, n, final_time, k))

    print(final_time)

    # path = centers_csv_path(dataset)
    # np.savetxt(path, centers, delimiter=",")


if __name__ == "__main__":
    main()

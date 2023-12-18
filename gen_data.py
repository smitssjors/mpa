import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Final

import numpy as np
from sklearn.datasets import make_circles, make_blobs, make_moons

DATASETS: Final[dict[str, Callable]] = {"circles": make_circles, "blobs": make_blobs, "moons": make_moons}


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("-n", "--n-samples", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng()

    generator = DATASETS[args.dataset]

    dest_dir = Path(f"data/{args.dataset}-{args.n_samples}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    vertices, _ = generator(args.n_samples)

    with open(dest_dir / "v.csv", "w+", newline="") as vertices_csv:
        vertices_csv_writer = csv.writer(vertices_csv, dialect="unix")
        with open(dest_dir / "e.csv", "w+", newline="") as edges_csv:
            edges_csv_writer = csv.writer(edges_csv, dialect="unix")

            for i in range(len(vertices)):
                radius = rng.random()
                vertices_csv_writer.writerow(np.concatenate((vertices[i], [radius])))
                for j in range(i + 1, len(vertices)):
                    dist = np.linalg.norm(vertices[i] - vertices[j])
                    edges_csv_writer.writerow(
                        np.concatenate((vertices[i], vertices[j], [dist]))
                    )


if __name__ == "__main__":
    main()

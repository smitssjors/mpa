import pandas as pd
from sklearn.datasets import fetch_california_housing


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
    prepare_housing()


if __name__ == "__main__":
    main()

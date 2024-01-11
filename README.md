# Massively Parallel Algorithms Assignments

This repo contains the code for both the MPA assignments.

## Getting started

First install the dependencies. Also make sure you have Spark installed.
```sh
pip install -r requirements.txt
```

Download the [McDonalds dataset](https://www.kaggle.com/datasets/ben1989/mcdonalds-locations). Then extract the CSV file into `data/mcdonalds/v.csv`.

Then run
```sh
spark-submit prepare_data.py
```
to prepare the data and download the other datasets.

Finally you can run the clustering with for example
```sh
spark-submit scalable_kmeans++.py housing 100
```
to run the k-mean algorithm on the housing dataset with k=100

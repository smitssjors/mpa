repeat 10 {
    for size (500 1000 5000 10000 50000 100000 500000 1000000) do
        for dataset (blobs circles moons) do
            for k (50 100 250) do
                spark-submit scalable_kmeans++.py $dataset-$size $k -l 2
            done
        done
    done
}
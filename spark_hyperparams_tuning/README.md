## 10–1000x faster Parallelized Randomized Search with Pyspark

Recently I was working on tuning hyperparameters for a really huge model. 
Manual tuning was not an option since I had to tweak a lot of parameters. Hyperopt was also not an option as it works in a serialized manner. That is at a time only a single model is being built. So it was taking up a lot of time to train each model and I was short on time.
I had to come up with another approach if I were to meet the deadline. So I thought of the one thing that helps in many such scenarios - Parallelization.
Can I parallelize my model hyperparameter search process?

Here is my approach in the [blogpost](https://towardsdatascience.com/10-1000x-faster-parallelized-randomized-search-with-pyspark-4de19e44f5e6)


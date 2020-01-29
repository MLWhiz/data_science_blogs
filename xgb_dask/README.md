## Time Series using Gradient Boosting

XGBoost is one of the most used libraries fora data science.
At the time XGBoost came into existence, it was lightning fast compared to its nearest rival Python’s Scikit-learn GBM. But as the times have progressed, it has been rivaled by some awesome libraries like LightGBM and Catboost, both on speed as well as accuracy.
I, for one, use LightGBM for most of the use cases where I have just got CPU for training. But when I have a GPU or multiple GPUs at my disposal, I still love to train with XGBoost.
Why?
So I could make use of the excellent GPU Capabilities provided by XGBoost in conjunction with Dask to use XGBoost in both single and multi-GPU mode.
How?
This post is about running XGBoost on Multi-GPU machines.

Here is my approach in the [blogpost](https://towardsdatascience.com/lightning-fast-xgboost-on-multiple-gpus-32710815c7c3)


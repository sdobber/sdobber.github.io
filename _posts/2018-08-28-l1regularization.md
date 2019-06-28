---
layout: post
title:  "L1-Regularization"
date:   2018-08-28 19:46:15 +0200
categories: julia tensorflow regularization sparsity
---




The next [programming exercise](https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=l1regularization-colab&hl=en) in the [machine learning crash course](https://developers.google.com/machine-learning/crash-course/) is about L1-regularization and sparsity.

In principle, one can add a regularization term to the `train_linear_classifier_model`-function from the previous file:

```julia
    y=feature_columns*m + b
    loss = -reduce_mean(log(y+ϵ).*target_columns + log(1-y+ϵ).*(1-target_columns))
    regularization = regularization_strength*reduce_sum(abs(m))
    optimizer_function=loss+regularization
```

Unfortunately, with this setup, all optimizers that are implemented in Tensorflow.jl still produce a non-sparse model. This is due to the fact that gradient descent algorithms basically never produce weights that are exactly zero.


To obtain a sparse set of weights, special classes of optimizers need to be used. In the original exercise, the FTRL Optimizer ("Follow the Regularized Leader") is used. This optimizer has been suggested in [this paper](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf) and shows good results for driving weights to zero with a good model accuracy.


I am not that familiar with implementing optimizers myself in TensorFlow.jl. If you have any suggestions on how to do that, I would be very interested - just leave a comment!

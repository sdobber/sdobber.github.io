---
layout: post
title:  "Introduction to TensorFlow.jl"
date:   2018-08-06 17:07:15 +0200
categories: julia tensorflow
---



The first exercise is about getting a general idea of TensorFlow. We run through the following number of steps:

1. Load the necessary packages and start a new session.
2. Load the data.
3. Define the features and targets as placeholders in 4. TensorFlow. Define the variables of the model and
put them together to give a linear regressor model.
4. Create some functions that help with feeding the input features to the model.
5. Train the model and have a look at the result.


One interesting difference between the Python version and Julia is how to implement gradient clipping.

Tensorflow.jl does not expose `tf.contrib.estimator.clip_gradients_by_norm` (to my knowledge),

so instead of `tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)` we use the construction

```julia
my_optimizer=(train.GradientDescentOptimizer(learning_rate))
gvs = train.compute_gradients(my_optimizer, loss)
capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
my_optimizer = train.apply_gradients(my_optimizer,capped_gvs)
```

The Jupyter Notebook can be downloaded [here](https://github.com/sdobber/MLCrashCourse/blob/master/1.%20First%20steps%20with%20Tensorflow%20Julia.ipynb). It is also displayed below.

***


This file is based on the file [First Steps with TensorFlow](https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=firststeps-colab&hl=en), which is part of Google's [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/).


```julia
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# First Steps with TensorFlow.jl

**Learning Objectives:**
  * Learn fundamental TensorFlow concepts
  * Building a linear regressor in TensorFlow to predict median housing price, at the granularity of city blocks, based on one input feature
  * Evaluate the accuracy of a model's predictions using Root Mean Squared Error (RMSE)
  * Improve the accuracy of a model by tuning its hyperparameters

The [data](https://developers.google.com/machine-learning/crash-course/california-housing-data-description) is based on 1990 census data from California. The training data can be downloaded [here](https://storage.googleapis.com/mledu-datasets/california_housing_train.csv).

## Setup
In this first cell, we'll load the necessary libraries.


```julia
using Plots
gr(fmt=:png)
using DataFrames
using TensorFlow
import CSV
using Random
using Statistics
#using PyCall
```

Start a new TensorFlow session.


```julia
sess=Session()
```

Next, we'll load our data set.


```julia
california_housing_dataframe = CSV.read("california_housing_train.csv", delim=",");
```

We'll randomize the data, just to be sure not to get any pathological ordering effects that might harm the performance of Stochastic Gradient Descent. Additionally, we'll scale `median_house_value` to be in units of thousands, so it can be learned a little more easily with learning rates in a range that we usually use.


```julia
california_housing_dataframe = california_housing_dataframe[Random.shuffle(1:size(california_housing_dataframe, 1)),:];
california_housing_dataframe[:median_house_value] /= 1000.0
california_housing_dataframe
```




<table class="data-frame"><thead><tr><th></th><th>longitude</th><th>latitude</th><th>housing_median_age</th><th>total_rooms</th><th>total_bedrooms</th><th>population</th><th>households</th><th>median_income</th><th>median_house_value</th></tr><tr><th></th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64</th></tr></thead><tbody><p>17,000 rows × 9 columns</p><tr><th>1</th><td>-122.43</td><td>37.66</td><td>29.0</td><td>3541.0</td><td>786.0</td><td>2259.0</td><td>770.0</td><td>4.3039</td><td>278.4</td></tr><tr><th>2</th><td>-117.99</td><td>33.89</td><td>23.0</td><td>2111.0</td><td>306.0</td><td>979.0</td><td>288.0</td><td>8.5621</td><td>347.8</td></tr><tr><th>3</th><td>-118.42</td><td>34.27</td><td>33.0</td><td>1209.0</td><td>341.0</td><td>1097.0</td><td>283.0</td><td>1.6295</td><td>134.3</td></tr><tr><th>4</th><td>-121.26</td><td>38.13</td><td>38.0</td><td>1419.0</td><td>411.0</td><td>1226.0</td><td>397.0</td><td>2.2188</td><td>68.8</td></tr><tr><th>5</th><td>-118.22</td><td>34.05</td><td>34.0</td><td>1113.0</td><td>313.0</td><td>928.0</td><td>290.0</td><td>3.1654</td><td>155.0</td></tr><tr><th>6</th><td>-121.98</td><td>37.37</td><td>36.0</td><td>1651.0</td><td>344.0</td><td>1062.0</td><td>331.0</td><td>4.575</td><td>215.4</td></tr><tr><th>7</th><td>-118.44</td><td>34.02</td><td>37.0</td><td>1592.0</td><td>308.0</td><td>783.0</td><td>321.0</td><td>6.2583</td><td>386.0</td></tr><tr><th>8</th><td>-121.88</td><td>37.35</td><td>49.0</td><td>1728.0</td><td>350.0</td><td>1146.0</td><td>391.0</td><td>3.5781</td><td>193.0</td></tr><tr><th>9</th><td>-119.64</td><td>37.31</td><td>15.0</td><td>2654.0</td><td>530.0</td><td>1267.0</td><td>489.0</td><td>2.8393</td><td>104.4</td></tr><tr><th>10</th><td>-118.61</td><td>35.47</td><td>13.0</td><td>2267.0</td><td>601.0</td><td>756.0</td><td>276.0</td><td>2.5474</td><td>78.4</td></tr><tr><th>11</th><td>-119.64</td><td>36.35</td><td>23.0</td><td>3182.0</td><td>563.0</td><td>1525.0</td><td>585.0</td><td>3.8108</td><td>90.4</td></tr><tr><th>12</th><td>-122.04</td><td>36.97</td><td>45.0</td><td>1302.0</td><td>245.0</td><td>621.0</td><td>258.0</td><td>5.1806</td><td>266.4</td></tr><tr><th>13</th><td>-121.99</td><td>37.36</td><td>33.0</td><td>2321.0</td><td>480.0</td><td>1230.0</td><td>451.0</td><td>4.9091</td><td>270.3</td></tr><tr><th>14</th><td>-123.75</td><td>39.37</td><td>16.0</td><td>1377.0</td><td>296.0</td><td>830.0</td><td>279.0</td><td>3.25</td><td>151.4</td></tr><tr><th>15</th><td>-121.83</td><td>37.38</td><td>31.0</td><td>3633.0</td><td>843.0</td><td>2677.0</td><td>797.0</td><td>3.2222</td><td>184.8</td></tr><tr><th>16</th><td>-117.2</td><td>34.48</td><td>7.0</td><td>4998.0</td><td>953.0</td><td>2764.0</td><td>891.0</td><td>3.205</td><td>101.9</td></tr><tr><th>17</th><td>-122.41</td><td>37.76</td><td>52.0</td><td>1427.0</td><td>281.0</td><td>620.0</td><td>236.0</td><td>1.9944</td><td>262.5</td></tr><tr><th>18</th><td>-122.26</td><td>38.0</td><td>6.0</td><td>678.0</td><td>104.0</td><td>318.0</td><td>91.0</td><td>5.2375</td><td>246.4</td></tr><tr><th>19</th><td>-122.09</td><td>37.7</td><td>30.0</td><td>1751.0</td><td>269.0</td><td>731.0</td><td>263.0</td><td>6.005</td><td>263.9</td></tr><tr><th>20</th><td>-124.17</td><td>41.76</td><td>20.0</td><td>2673.0</td><td>538.0</td><td>1282.0</td><td>514.0</td><td>2.4605</td><td>105.9</td></tr><tr><th>21</th><td>-121.91</td><td>36.59</td><td>17.0</td><td>5039.0</td><td>833.0</td><td>1678.0</td><td>710.0</td><td>6.2323</td><td>339.1</td></tr><tr><th>22</th><td>-117.29</td><td>34.11</td><td>35.0</td><td>2426.0</td><td>715.0</td><td>1920.0</td><td>586.0</td><td>1.5561</td><td>68.0</td></tr><tr><th>23</th><td>-117.33</td><td>34.08</td><td>35.0</td><td>2240.0</td><td>423.0</td><td>1394.0</td><td>396.0</td><td>3.1799</td><td>86.7</td></tr><tr><th>24</th><td>-118.08</td><td>33.88</td><td>27.0</td><td>3065.0</td><td>736.0</td><td>1840.0</td><td>719.0</td><td>3.6417</td><td>208.1</td></tr><tr><th>25</th><td>-117.81</td><td>33.81</td><td>19.0</td><td>3154.0</td><td>390.0</td><td>1404.0</td><td>384.0</td><td>8.9257</td><td>431.8</td></tr><tr><th>26</th><td>-117.17</td><td>32.78</td><td>42.0</td><td>1104.0</td><td>305.0</td><td>892.0</td><td>270.0</td><td>2.2768</td><td>145.2</td></tr><tr><th>27</th><td>-118.13</td><td>34.09</td><td>21.0</td><td>3862.0</td><td>1186.0</td><td>2773.0</td><td>1102.0</td><td>2.7816</td><td>188.2</td></tr><tr><th>28</th><td>-118.17</td><td>33.91</td><td>37.0</td><td>1499.0</td><td>288.0</td><td>1237.0</td><td>344.0</td><td>3.9333</td><td>162.3</td></tr><tr><th>29</th><td>-118.32</td><td>33.94</td><td>37.0</td><td>2740.0</td><td>504.0</td><td>1468.0</td><td>479.0</td><td>4.5368</td><td>168.8</td></tr><tr><th>30</th><td>-119.12</td><td>36.54</td><td>30.0</td><td>2747.0</td><td>515.0</td><td>1368.0</td><td>453.0</td><td>2.9828</td><td>85.2</td></tr><tr><th>&vellip;</th><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td></tr></tbody></table>



## Examine the Data

It's a good idea to get to know your data a little bit before you work with it.

We'll print out a quick summary of a few useful statistics on each column: count of examples, mean, standard deviation, max, min, and various quantiles.


```julia
describe(california_housing_dataframe)
```




<table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>min</th><th>median</th><th>max</th><th>nunique</th><th>nmissing</th><th>eltype</th></tr><tr><th></th><th>Symbol</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Nothing</th><th>Union…</th><th>DataType</th></tr></thead><tbody><p>9 rows × 8 columns</p><tr><th>1</th><td>longitude</td><td>-119.562</td><td>-124.35</td><td>-118.49</td><td>-114.31</td><td></td><td>0</td><td>Float64</td></tr><tr><th>2</th><td>latitude</td><td>35.6252</td><td>32.54</td><td>34.25</td><td>41.95</td><td></td><td>0</td><td>Float64</td></tr><tr><th>3</th><td>housing_median_age</td><td>28.5894</td><td>1.0</td><td>29.0</td><td>52.0</td><td></td><td>0</td><td>Float64</td></tr><tr><th>4</th><td>total_rooms</td><td>2643.66</td><td>2.0</td><td>2127.0</td><td>37937.0</td><td></td><td>0</td><td>Float64</td></tr><tr><th>5</th><td>total_bedrooms</td><td>539.411</td><td>1.0</td><td>434.0</td><td>6445.0</td><td></td><td>0</td><td>Float64</td></tr><tr><th>6</th><td>population</td><td>1429.57</td><td>3.0</td><td>1167.0</td><td>35682.0</td><td></td><td>0</td><td>Float64</td></tr><tr><th>7</th><td>households</td><td>501.222</td><td>1.0</td><td>409.0</td><td>6082.0</td><td></td><td>0</td><td>Float64</td></tr><tr><th>8</th><td>median_income</td><td>3.88358</td><td>0.4999</td><td>3.5446</td><td>15.0001</td><td></td><td>0</td><td>Float64</td></tr><tr><th>9</th><td>median_house_value</td><td>207.301</td><td>14.999</td><td>180.4</td><td>500.001</td><td></td><td></td><td>Float64</td></tr></tbody></table>



## Build the First Model

In this exercise, we'll try to predict `median_house_value`, which will be our label (sometimes also called a target). We'll use `total_rooms` as our input feature.

**NOTE:** Our data is at the city block level, so this feature represents the total number of rooms in that block.

To train our model, we'll set up a linear regressor model.

### Step 1: Define Features and Configure Feature Columns

In order to import our training data into TensorFlow, we need to specify what type of data each feature contains. There are two main types of data we'll use in this and future exercises:

* **Categorical Data**: Data that is textual. In this exercise, our housing data set does not contain any categorical features, but examples you might see would be the home style, the words in a real-estate ad.

* **Numerical Data**: Data that is a number (integer or float) and that you want to treat as a number. As we will discuss more later sometimes you might want to treat numerical data (e.g., a postal code) as if it were categorical.

To start, we're going to use just one numeric input feature, `total_rooms`. The following code pulls the `total_rooms` data from our `california_housing_dataframe` and defines a feature and targer column:


```julia
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[:total_rooms]

# Configure a numeric feature column for total_rooms.
feature_columns = placeholder(Float32)
target_columns = placeholder(Float32)
```

### Step 2: Define the Target

Next, we'll define our target, which is `median_house_value`. Again, we can pull it from our `california_housing_dataframe`:


```julia
# Define the label.
targets = california_housing_dataframe[:median_house_value];
```

### Step 3: Configure the LinearRegressor

Next, we'll configure a linear regression model using LinearRegressor. We'll train this model using the `GradientDescentOptimizer`, which implements Mini-Batch Stochastic Gradient Descent (SGD). The `learning_rate` argument controls the size of the gradient step.

**NOTE:** To be safe, we also apply [gradient clipping](https://developers.google.com/machine-learning/glossary/#gradient_clipping) to our optimizer via `clip_gradients_by_norm`. Gradient clipping ensures the magnitude of the gradients do not become too large during training, which can cause gradient descent to fail.


```julia
# Configure the linear regression model with our feature columns and optimizer.
m=Variable(0.05)
b=Variable(0.0)
y=m.*feature_columns+b
loss=reduce_sum((target_columns - y).^2)

# Use gradient descent as the optimizer for training the model.
# Set a learning rate of 0.0000001 for Gradient Descent.
learning_rate=0.0000001;
my_optimizer=train.minimize(train.GradientDescentOptimizer(learning_rate), loss)
#gvs = train.compute_gradients(my_optimizer, loss)
#capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
#my_optimizer = train.apply_gradients(my_optimizer,capped_gvs)
```


### Step 4: Define the Input Function

To import our California housing data into our linear regressor model, we need to define an input function, which instructs TensorFlow how to preprocess
the data, as well as how to batch, shuffle, and repeat it during model training.

First, we'll convert our *DataFrame* feature data into an array. We can then construct a dataset object from our data, and then break our data into batches of `batch_size`, to be repeated for the specified number of epochs (num_epochs).

**NOTE:** When the default value of `num_epochs=None` is passed to `repeat()`, the input data will be repeated indefinitely.

Next, if `shuffle` is set to `True`, we'll shuffle the data so that it's passed to the model randomly during training. The `buffer_size` argument specifies
the size of the dataset from which `shuffle` will randomly sample.

Finally, our input function constructs an iterator for the dataset and returns the next batch of data to the linear regressor.


```julia
function create_batches(features, targets, steps, batch_size=5, num_epochs=0)

    if(num_epochs==0)
        num_epochs=ceil(batch_size*steps/length(features))
    end

    features_batches=Union{Float64, Missings.Missing}[]
    target_batches=Union{Float64, Missings.Missing}[]

    for i=1:num_epochs
        select=Random.shuffle(1:length(features))
        append!(features_batches, features[select])
        append!(target_batches, targets[select])
    end

    return features_batches, target_batches
end
```


```julia
function next_batch(features_batches, targets_batches, batch_size, iter)

    select=mod((iter-1)*batch_size+1, length(features_batches)):mod(iter*batch_size, length(features_batches));

    ds=features_batches[select];
    target=targets_batches[select];

    return ds, target
end

```


```julia
function my_input_fn(features_batches, targets_batches, iter, batch_size=5, shuffle_flag=1)
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """


    # Construct a dataset, and configure batching/repeating.
    ds, target = next_batch(features_batches, targets_batches, batch_size, iter)

    # Shuffle the data, if specified.
    if shuffle_flag==1
      select=Random.shuffle(1:size(ds, 1));
        ds = ds[select,:]
        target = target[select, :]
    end

    # Return the next batch of data.
    return convert(Matrix{Float64},ds), convert(Matrix{Float64},target)
end
```


**NOTE:** We'll continue to use this same input function in later exercises.

### Step 5: Train the Model

We can now call `train()` on our `my_optimizer` to train the model. To start, we'll train for 100 steps.


```julia
steps=100;
batch_size=5;
run(sess, global_variables_initializer())
features_batches, targets_batches = create_batches(my_feature, targets, steps, batch_size)

for i=1:steps
    features, labels = my_input_fn(features_batches, targets_batches, i, batch_size)
    run(sess, my_optimizer, Dict(feature_columns=>features, target_columns=>labels))
end
```

We can assess the values for the weight and bias variables:


```julia
weight = run(sess,m)
```
    -2.1643431694151094e86


```julia
bias = run(sess,b)
```
    -1.184116157861517e83



### Step 6: Evaluate the Model

Let's make predictions on that training data, to see how well our model fit it during training.

**NOTE:** Training error measures how well your model fits the training data, but it **_does not_** measure how well your model **_generalizes to new data_**. In later exercises, you'll explore how to split your data to evaluate your model's ability to generalize.



```julia
# Run the TF session on the data to make predictions.
predictions = run(sess, y, Dict(feature_columns=>convert.(Float64, my_feature)));

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = mean((predictions- targets).^2);
root_mean_squared_error = sqrt(mean_squared_error);
println("Mean Squared Error (on training data): ", mean_squared_error)
println("Root Mean Squared Error (on training data): ", root_mean_squared_error)
```

    Mean Squared Error (on training data): 5.499863808321752e179
    Root Mean Squared Error (on training data): 7.416106666116495e89


Is this a good model? How would you judge how large this error is?

Mean Squared Error (MSE) can be hard to interpret, so we often look at Root Mean Squared Error (RMSE)
instead.  A nice property of RMSE is that it can be interpreted on the same scale as the original targets.

Let's compare the RMSE to the difference of the min and max of our targets:


```julia
min_house_value = minimum(california_housing_dataframe[:median_house_value])
max_house_value = maximum(california_housing_dataframe[:median_house_value])
min_max_difference = max_house_value - min_house_value

println("Min. Median House Value: " , min_house_value)
println("Max. Median House Value: " , max_house_value)
println("Difference between Min. and Max.: " , min_max_difference)
println("Root Mean Squared Error: " , root_mean_squared_error)
```

    Min. Median House Value: 14.999
    Max. Median House Value: 500.001
    Difference between Min. and Max.: 485.00199999999995
    Root Mean Squared Error: 7.416106666116495e89


Our error spans nearly half the range of the target values. Can we do better?

This is the question that nags at every model developer. Let's develop some basic strategies to reduce model error.

The first thing we can do is take a look at how well our predictions match our targets, in terms of overall summary statistics.


```julia
calibration_data = DataFrame();
calibration_data[:predictions] = predictions;
calibration_data[:targets] = targets;
describe(calibration_data)
```




<table class="data-frame"><thead><tr><th></th><th>variable</th><th>mean</th><th>min</th><th>median</th><th>max</th><th>nunique</th><th>nmissing</th><th>eltype</th></tr><tr><th></th><th>Symbol</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Nothing</th><th>Nothing</th><th>DataType</th></tr></thead><tbody><p>2 rows × 8 columns</p><tr><th>1</th><td>predictions</td><td>-5.7218e89</td><td>-8.21087e90</td><td>-4.60356e89</td><td>-4.32987e86</td><td></td><td></td><td>Float64</td></tr><tr><th>2</th><td>targets</td><td>207.301</td><td>14.999</td><td>180.4</td><td>500.001</td><td></td><td></td><td>Float64</td></tr></tbody></table>



Okay, maybe this information is helpful. How does the mean value compare to the model's RMSE? How about the various quantiles?

We can also visualize the data and the line we've learned.  Recall that linear regression on a single feature can be drawn as a line mapping input *x* to output *y*.

First, we'll get a uniform random sample of the data so we can make a readable scatter plot.


```julia
sample = california_housing_dataframe[rand(1:size(california_housing_dataframe,1), 300),:];
```

Next, we'll plot the line we've learned, drawing from the model's bias term and feature weight, together with the scatter plot. The line will show up red.


```julia
# Get the min and max total_rooms values.
x_0 = minimum(sample[:total_rooms])
x_1 = maximum(sample[:total_rooms])

# Retrieve the final weight and bias generated during training.
weight = run(sess,m)
bias = run(sess,b)

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plot([x_0, x_1], [y_0, y_1], c=:red, ylabel="median_house_value", xlabel="total_rooms", label="Fit line")

# Plot a scatter plot from our data sample.
scatter!(sample[:total_rooms], sample[:median_house_value], c=:blue, label="Data")
```

![png](/images/introductiontotensorflow/output_47_0.png)


```julia
input_feature=:total_rooms
```



This initial line looks way off.  See if you can look back at the summary stats and see the same information encoded there.

Together, these initial sanity checks suggest we may be able to find a much better line.

## Tweak the Model Hyperparameters
For this exercise, we've put all the above code in a single function for convenience. You can call the function with different parameters to see the effect.

In this function, we'll proceed in 10 evenly divided periods so that we can observe the model improvement at each period.

For each period, we'll compute and graph training loss.  This may help you judge when a model is converged, or if it needs more iterations.

We'll also plot the feature weight and bias term values learned by the model over time.  This is another way to see how things converge.


```julia
function train_model(learning_rate, steps, batch_size, input_feature=:total_rooms)
  """Trains a linear regression model of one feature.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """

  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = convert.(Float32,california_housing_dataframe[my_feature])
  my_label = :median_house_value
  targets = convert.(Float32,california_housing_dataframe[my_label])

  # Create feature columns.
  feature_columns = placeholder(Float32)
  target_columns = placeholder(Float32)

  # Create a linear regressor object.
  # Configure the linear regression model with our feature columns and optimizer.
  m=Variable(0.0)
  b=Variable(0.0)
  y=m.*feature_columns .+ b
  loss=reduce_sum((target_columns - y).^2)
  features_batches, targets_batches = create_batches(my_feature_data, targets, steps, batch_size)

  # Use gradient descent as the optimizer for training the model.
  my_optimizer=(train.GradientDescentOptimizer(learning_rate))
  gvs = train.compute_gradients(my_optimizer, loss)
  capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
  my_optimizer = train.apply_gradients(my_optimizer,capped_gvs)

  run(sess, global_variables_initializer())

  # Set up to plot the state of our model's line each period.
  sample = california_housing_dataframe[rand(1:size(california_housing_dataframe,1), 300),:];
  p1=scatter(sample[my_feature], sample[my_label], title="Learned Line by Period", ylabel=my_label, xlabel=my_feature,color=:coolwarm)
  colors= [ColorGradient(:coolwarm)[i] for i in range(0,stop=1, length=periods+1)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  println("Training model...")
  println("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in 1:periods
    # Train the model, starting from the prior state.
   for i=1:steps_per_period
    features, labels = my_input_fn(features_batches, targets_batches, convert(Int,(period-1)*steps_per_period+i), batch_size)
    run(sess, my_optimizer, Dict(feature_columns=>features, target_columns=>labels))
   end
    # Take a break and compute predictions.
    predictions = run(sess, y, Dict(feature_columns=>convert.(Float64, my_feature_data)));

    # Compute loss.
     mean_squared_error = mean((predictions- targets).^2)
     root_mean_squared_error = sqrt(mean_squared_error)
    # Occasionally print the current loss.
    println("  period ", period, ": ", root_mean_squared_error)
    # Add the loss metrics from this period to our list.
    push!(root_mean_squared_errors, root_mean_squared_error)
    # Finally, track the weights and biases over time.

    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = [0 maximum(sample[my_label])]

    weight = run(sess,m)
    bias = run(sess,b)

    x_extents = (y_extents .- bias) / weight    
    x_extents = max.(min.(x_extents, maximum(sample[my_feature])),
                           minimum(sample[my_feature]))    
    y_extents = weight .* x_extents .+ bias

    p1=plot!(x_extents', y_extents', color=colors[period], linewidth=2)
 end

  println("Model training finished.")

  # Output a graph of loss metrics over periods.
  p2=plot(root_mean_squared_errors, title="Root Mean Squared Error vs. Periods", ylabel="RMSE", xlabel="Periods")

  # Output a table with calibration data.
  calibration_data = DataFrame()
  calibration_data[:predictions] = predictions
  calibration_data[:targets] = targets
  describe(calibration_data)

  println("Final RMSE (on training data): ", root_mean_squared_errors[end])
  println("Final Weight (on training data): ", weight)
  println("Final Bias (on training data): ", bias)

  return p1, p2      
end
```



## Task 1:  Achieve an RMSE of 180 or Below

Tweak the model hyperparameters to improve loss and better match the target distribution.
If, after 5 minutes or so, you're having trouble beating a RMSE of 180, check the solution for a possible combination.


```julia
sess=Session()
p1, p2= train_model(
    0.0001, # learning rate
    20, # steps
    5 # batch size
)
```

    Training model...
    RMSE (on training data):
      period 1: 235.10452754834785
      period 2: 232.69434701416756
      period 3: 230.30994696291228
      period 4: 227.95213639459652
      period 5: 225.62174891342735
      period 6: 223.31964301811482
      period 7: 221.04670233191547
      period 8: 218.80383576386936
      period 9: 216.59197759206555
      period 10: 214.4120874591565
    Model training finished.
    Final RMSE (on training data): 214.4120874591565
    Final Weight (on training data): -2.1643431694151094e86
    Final Bias (on training data): -1.184116157861517e83





```julia
plot(p1, p2, layout=(1,2), legend=false)
```




![png](/images/introductiontotensorflow/output_54_0.png)



### Solution

Click below for one possible solution.


```julia
p1, p2= train_model(
    0.001, # learning rate
    20, # steps
    5 # batch size
)
```

    Training model...
    RMSE (on training data):
      period 1: 214.4120874591565
      period 2: 194.600111798902
      period 3: 179.20684050800634
      period 4: 169.44086805355906
      period 5: 169.434877106273
      period 6: 166.2920638633921
      period 7: 170.13862388370202
      period 8: 180.52860974790764
      period 9: 170.13354759178128
      period 10: 180.5261550915958
    Model training finished.
    Final RMSE (on training data): 180.5261550915958
    Final Weight (on training data): -2.1643431694151094e86
    Final Bias (on training data): -1.184116157861517e83


```julia
plot(p1, p2, layout=(1,2), legend=false)
```




![png](/images/introductiontotensorflow/output_57_0.png)



This is just one possible configuration; there may be other combinations of settings that also give good results. Note that in general, this exercise isn't about finding the *one best* setting, but to help build your intutions about how tweaking the model configuration affects prediction quality.

### Is There a Standard Heuristic for Model Tuning?

This is a commonly asked question. The short answer is that the effects of different hyperparameters are data dependent. So there are no hard-and-fast rules; you'll need to test on your data.

That said, here are a few rules of thumb that may help guide you:

 * Training error should steadily decrease, steeply at first, and should eventually plateau as training converges.
 * If the training has not converged, try running it for longer.
 * If the training error decreases too slowly, increasing the learning rate may help it decrease faster.
   * But sometimes the exact opposite may happen if the learning rate is too high.
 * If the training error varies wildly, try decreasing the learning rate.
   * Lower learning rate plus larger number of steps or larger batch size is often a good combination.
 * Very small batch sizes can also cause instability.  First try larger values like 100 or 1000, and decrease until you see degradation.

Again, never go strictly by these rules of thumb, because the effects are data dependent.  Always experiment and verify.

## Task 2: Try a Different Feature

See if you can do any better by replacing the `total_rooms` feature with the `population` feature.


```julia
# YOUR CODE HERE
p1, p2= train_model(
    0.0001, # learning rate
    300, # steps
    5, # batch size
    :population #feature
)
```

    Training model...
    RMSE (on training data):
      period 1: 219.9919658470582
      period 2: 204.6500256564945
      period 3: 194.31631437316608
      period 4: 185.43549863077988
      period 5: 179.30763919257313
      period 6: 176.34328658344398
      period 7: 175.86576786417197
      period 8: 176.0115967111658
      period 9: 176.62469585550392
      period 10: 176.460123907991
    Model training finished.
    Final RMSE (on training data): 176.460123907991
    Final Weight (on training data): -2.1643431694151094e86
    Final Bias (on training data): -1.184116157861517e83

```julia
plot(p1, p2, layout=(1,2), legend=false)
```




![png](/images/introductiontotensorflow/output_62_0.png)

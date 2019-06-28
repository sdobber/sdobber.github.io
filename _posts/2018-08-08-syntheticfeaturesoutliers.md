---
layout: post
title:  "Synthetic Features and Outliers"
date:   2018-08-08 19:33:15 +0200
categories: julia tensorflow features outliers
---

In this second part, we create a synthetic feature and remove some outliers from the data set.


The Jupyter notebook can be downloaded [here](https://github.com/sdobber/MLCrashCourse/blob/master/2.%20Synthetic%20Features%20and%20Outliers%20Julia.ipynb).

***


This notebook is based on the file [Synthetic Features and Outliers](https://colab.research.google.com/notebooks/mlcc/synthetic_features_and_outliers.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=syntheticfeatures-colab&hl=en), which is part of Google's [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/).


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

# Synthetic Features and Outliers

**Learning Objectives:**
  * Create a synthetic feature that is the ratio of two other features
  * Use this new feature as an input to a linear regression model
  * Improve the effectiveness of the model by identifying and clipping (removing) outliers out of the input data

Let's revisit our model from the previous First Steps with TensorFlow exercise.

First, we'll import the California housing data into `DataFrame`:

## Setup


```julia
using Plots
gr(fmt=:png)
using DataFrames
using TensorFlow
import CSV
using Random
using Statistics

sess=Session()

california_housing_dataframe = CSV.read("california_housing_train.csv", delim=",");
california_housing_dataframe[:median_house_value] /= 1000.0
california_housing_dataframe
```



<table class="data-frame"><thead><tr><th></th><th>longitude</th><th>latitude</th><th>housing_median_age</th><th>total_rooms</th><th>total_bedrooms</th><th>population</th><th>households</th><th>median_income</th><th>median_house_value</th></tr><tr><th></th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64⍰</th><th>Float64</th></tr></thead><tbody><p>17,000 rows × 9 columns</p><tr><th>1</th><td>-114.31</td><td>34.19</td><td>15.0</td><td>5612.0</td><td>1283.0</td><td>1015.0</td><td>472.0</td><td>1.4936</td><td>66.9</td></tr><tr><th>2</th><td>-114.47</td><td>34.4</td><td>19.0</td><td>7650.0</td><td>1901.0</td><td>1129.0</td><td>463.0</td><td>1.82</td><td>80.1</td></tr><tr><th>3</th><td>-114.56</td><td>33.69</td><td>17.0</td><td>720.0</td><td>174.0</td><td>333.0</td><td>117.0</td><td>1.6509</td><td>85.7</td></tr><tr><th>4</th><td>-114.57</td><td>33.64</td><td>14.0</td><td>1501.0</td><td>337.0</td><td>515.0</td><td>226.0</td><td>3.1917</td><td>73.4</td></tr><tr><th>5</th><td>-114.57</td><td>33.57</td><td>20.0</td><td>1454.0</td><td>326.0</td><td>624.0</td><td>262.0</td><td>1.925</td><td>65.5</td></tr><tr><th>6</th><td>-114.58</td><td>33.63</td><td>29.0</td><td>1387.0</td><td>236.0</td><td>671.0</td><td>239.0</td><td>3.3438</td><td>74.0</td></tr><tr><th>7</th><td>-114.58</td><td>33.61</td><td>25.0</td><td>2907.0</td><td>680.0</td><td>1841.0</td><td>633.0</td><td>2.6768</td><td>82.4</td></tr><tr><th>8</th><td>-114.59</td><td>34.83</td><td>41.0</td><td>812.0</td><td>168.0</td><td>375.0</td><td>158.0</td><td>1.7083</td><td>48.5</td></tr><tr><th>9</th><td>-114.59</td><td>33.61</td><td>34.0</td><td>4789.0</td><td>1175.0</td><td>3134.0</td><td>1056.0</td><td>2.1782</td><td>58.4</td></tr><tr><th>10</th><td>-114.6</td><td>34.83</td><td>46.0</td><td>1497.0</td><td>309.0</td><td>787.0</td><td>271.0</td><td>2.1908</td><td>48.1</td></tr><tr><th>11</th><td>-114.6</td><td>33.62</td><td>16.0</td><td>3741.0</td><td>801.0</td><td>2434.0</td><td>824.0</td><td>2.6797</td><td>86.5</td></tr><tr><th>12</th><td>-114.6</td><td>33.6</td><td>21.0</td><td>1988.0</td><td>483.0</td><td>1182.0</td><td>437.0</td><td>1.625</td><td>62.0</td></tr><tr><th>13</th><td>-114.61</td><td>34.84</td><td>48.0</td><td>1291.0</td><td>248.0</td><td>580.0</td><td>211.0</td><td>2.1571</td><td>48.6</td></tr><tr><th>14</th><td>-114.61</td><td>34.83</td><td>31.0</td><td>2478.0</td><td>464.0</td><td>1346.0</td><td>479.0</td><td>3.212</td><td>70.4</td></tr><tr><th>15</th><td>-114.63</td><td>32.76</td><td>15.0</td><td>1448.0</td><td>378.0</td><td>949.0</td><td>300.0</td><td>0.8585</td><td>45.0</td></tr><tr><th>16</th><td>-114.65</td><td>34.89</td><td>17.0</td><td>2556.0</td><td>587.0</td><td>1005.0</td><td>401.0</td><td>1.6991</td><td>69.1</td></tr><tr><th>17</th><td>-114.65</td><td>33.6</td><td>28.0</td><td>1678.0</td><td>322.0</td><td>666.0</td><td>256.0</td><td>2.9653</td><td>94.9</td></tr><tr><th>18</th><td>-114.65</td><td>32.79</td><td>21.0</td><td>44.0</td><td>33.0</td><td>64.0</td><td>27.0</td><td>0.8571</td><td>25.0</td></tr><tr><th>19</th><td>-114.66</td><td>32.74</td><td>17.0</td><td>1388.0</td><td>386.0</td><td>775.0</td><td>320.0</td><td>1.2049</td><td>44.0</td></tr><tr><th>20</th><td>-114.67</td><td>33.92</td><td>17.0</td><td>97.0</td><td>24.0</td><td>29.0</td><td>15.0</td><td>1.2656</td><td>27.5</td></tr><tr><th>21</th><td>-114.68</td><td>33.49</td><td>20.0</td><td>1491.0</td><td>360.0</td><td>1135.0</td><td>303.0</td><td>1.6395</td><td>44.4</td></tr><tr><th>22</th><td>-114.73</td><td>33.43</td><td>24.0</td><td>796.0</td><td>243.0</td><td>227.0</td><td>139.0</td><td>0.8964</td><td>59.2</td></tr><tr><th>23</th><td>-114.94</td><td>34.55</td><td>20.0</td><td>350.0</td><td>95.0</td><td>119.0</td><td>58.0</td><td>1.625</td><td>50.0</td></tr><tr><th>24</th><td>-114.98</td><td>33.82</td><td>15.0</td><td>644.0</td><td>129.0</td><td>137.0</td><td>52.0</td><td>3.2097</td><td>71.3</td></tr><tr><th>25</th><td>-115.22</td><td>33.54</td><td>18.0</td><td>1706.0</td><td>397.0</td><td>3424.0</td><td>283.0</td><td>1.625</td><td>53.5</td></tr><tr><th>26</th><td>-115.32</td><td>32.82</td><td>34.0</td><td>591.0</td><td>139.0</td><td>327.0</td><td>89.0</td><td>3.6528</td><td>100.0</td></tr><tr><th>27</th><td>-115.37</td><td>32.82</td><td>30.0</td><td>1602.0</td><td>322.0</td><td>1130.0</td><td>335.0</td><td>3.5735</td><td>71.1</td></tr><tr><th>28</th><td>-115.37</td><td>32.82</td><td>14.0</td><td>1276.0</td><td>270.0</td><td>867.0</td><td>261.0</td><td>1.9375</td><td>80.9</td></tr><tr><th>29</th><td>-115.37</td><td>32.81</td><td>32.0</td><td>741.0</td><td>191.0</td><td>623.0</td><td>169.0</td><td>1.7604</td><td>68.6</td></tr><tr><th>30</th><td>-115.37</td><td>32.81</td><td>23.0</td><td>1458.0</td><td>294.0</td><td>866.0</td><td>275.0</td><td>2.3594</td><td>74.3</td></tr><tr><th>&vellip;</th><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td></tr></tbody></table>



Next, we'll set up our input functions, and define the function for model training:


```julia
function create_batches(features, targets, steps, batch_size=5, num_epochs=0)

    if(num_epochs==0)
        num_epochs=ceil(batch_size*steps/length(features))
    end

    features_batches=Union{Float64, Missings.Missing}[]
    target_batches=Union{Float64, Missings.Missing}[]

    for i=1:num_epochs

        select=shuffle(1:length(features))

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
      features: DataFrame of features
      targets: DataFrame of targets
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


```julia
function train_model(learning_rate, steps, batch_size, input_feature=:total_rooms)
  """Trains a linear regression model of one feature.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `symbol` specifying a column from `california_housing_dataframe`
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
  m=Variable(0.0)
  b=Variable(0.0)
  y=m.*feature_columns .+ b
  loss=reduce_sum((target_columns - y).^2)
  run(sess, global_variables_initializer())
  features_batches, targets_batches = create_batches(my_feature_data, targets, steps, batch_size)

  # Use gradient descent as the optimizer for training the model.
  #my_optimizer=train.minimize(train.GradientDescentOptimizer(learning_rate), loss)
  my_optimizer=(train.GradientDescentOptimizer(learning_rate))
  gvs = train.compute_gradients(my_optimizer, loss)
  capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
  my_optimizer = train.apply_gradients(my_optimizer,capped_gvs)

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
    predictions = run(sess, y, Dict(feature_columns=> my_feature_data));    

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

  predictions = run(sess, y, Dict(feature_columns=> my_feature_data));
  weight = run(sess,m)
  bias = run(sess,b)

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

  return p1, p2, calibration_data   
end
```

## Task 1: Try a Synthetic Feature

Both the `total_rooms` and `population` features count totals for a given city block.

But what if one city block were more densely populated than another? We can explore how block density relates to median house value by creating a synthetic feature that's a ratio of `total_rooms` and `population`.

In the cell below, we create a feature called `rooms_per_person`, and use that as the `input_feature` to `train_model()`.


```julia
california_housing_dataframe[:rooms_per_person] =(
    california_housing_dataframe[:total_rooms] ./ california_housing_dataframe[:population]);
```


```julia
p1, p2, calibration_data= train_model(
    0.05, # learning rate
    1000, # steps
    5, # batch size
    :rooms_per_person #feature
)
```

    Training model...
    RMSE (on training data):
      period 1: 174.73499015754794
      period 2: 135.18378014936647
      period 3: 124.81483763650894
      period 4: 124.99348861666715
      period 5: 128.0855648925441
      period 6: 129.86065434272652
      period 7: 128.06995380520772
      period 8: 129.3628149624267
      period 9: 129.9533622545398
      period 10: 129.8691255721607
    Model training finished.
    Final RMSE (on training data): 129.8691255721607
    Final Weight (on training data): 74.18756812501896
    Final Bias (on training data): 67.78876122300292




```julia
plot(p1, p2, layout=(1,2), legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_15_0.png)



## Task 2: Identify Outliers

We can visualize the performance of our model by creating a scatter plot of predictions vs. target values.  Ideally, these would lie on a perfectly correlated diagonal line.

We use `scatter` to create a scatter plot of predictions vs. targets, using the rooms-per-person model you trained in Task 1.

Do you see any oddities?  Trace these back to the source data by looking at the distribution of values in `rooms_per_person`.


```julia
scatter(calibration_data[:predictions], calibration_data[:targets], legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_17_0.png)



The calibration data shows most scatter points aligned to a line. The line is almost vertical, but we'll come back to that later. Right now let's focus on the ones that deviate from the line. We notice that they are relatively few in number.

If we plot a histogram of `rooms_per_person`, we find that we have a few outliers in our input data:


```julia
histogram(california_housing_dataframe[:rooms_per_person], nbins=20, legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_19_0.png)



## Task 3: Clip Outliers

We see if we can further improve the model fit by setting the outlier values of `rooms_per_person` to some reasonable minimum or maximum.

The histogram we created in Task 2 shows that the majority of values are less than `5`. Let's clip `rooms_per_person` to 5, and plot a histogram to double-check the results.


```julia
california_housing_dataframe[:rooms_per_person] = min.(
    california_housing_dataframe[:rooms_per_person],5)

histogram(california_housing_dataframe[:rooms_per_person], nbins=20, legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_22_0.png)



To verify that clipping worked, let's train again and print the calibration data once more:


```julia
p1, p2, calibration_data= train_model(
    0.05, # learning rate
    500, # steps
    10, # batch size
    :rooms_per_person #feature
)
```

    Training model...
    RMSE (on training data):
      period 1: 204.65393150901195
      period 2: 173.7183427312223
      period 3: 145.97809305428905
      period 4: 125.39350453104828
      period 5: 113.5851230428793
      period 6: 108.94376856469054
      period 7: 107.51132608903492
      period 8: 107.37501891367756
      period 9: 107.35720127223883
      period 10: 107.36959825293329
    Model training finished.
    Final RMSE (on training data): 107.36959825293329
    Final Weight (on training data): 70.5
    Final Bias (on training data): 71.72122192382814



```julia
plot(p1, p2, layout=(1,2), legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_25_0.png)




```julia
scatter(calibration_data[:predictions], calibration_data[:targets], legend=false)
```




![png](/images/syntheticfeaturesoutliers/output_26_0.png)

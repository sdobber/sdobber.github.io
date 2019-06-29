---
layout: post
title:  "Introduction to Neural Nets"
date:   2018-09-02 21:38:15 +0200
categories: julia tensorflow neuralnet machinelearning
---


In this exercise, we construct a neural network with a given structure of the hidden layer.

The main part consists of first creating the network parameters
```
  # Network parameters
  push!(hidden_units,size(training_targets,2)) #create an output node that fits to the size of the targets
  activation_functions = Vector{Function}(size(hidden_units,1))
  activation_functions[1:end-1]=z->nn.dropout(nn.relu(z), keep_probability)
  activation_functions[end] = identity #Last function should be idenity as we need the logits  
```

and then put the network together by using
```
  # create network
  Zs = [feature_columns]
  for (ii,(hlsize, actfun)) in enumerate(zip(hidden_units, activation_functions))
        Wii = get_variable("W_$ii"*randstring(4), [get_shape(Zs[end], 2), hlsize], Float32)
        bii = get_variable("b_$ii"*randstring(4), [hlsize], Float32)
        Zii = actfun(Zs[end]*Wii + bii)
        push!(Zs, Zii)
  end
  y=Zs[end]
```

This approach was inspired by the blog posts that can be found [here](https://www.juliabloggers.com/intro-to-machine-learning-with-tensorflow-jl/) and [here](https://white.ucc.asn.au/2017/08/02/Intro-to-Machine-Learning-with-TensorFlow.jl.html).


When running the model several times, identical names for the nodes create error messages - that is why we added a random string to each variable name as in `"W_$ii"*randstring(4)`.  A simpler approach without using names would be
```
    # Create network
    Zs = [feature_columns]
    for (ii,(hlsize, actfun)) in enumerate(zip(hidden_units, activation_functions))
        Wii = Variable(zeros(get_shape(Zs[end], 2), hlsize)  )
        bii = Variable(zeros( 1, hlsize) )
        Zii = actfun(Zs[end]*Wii + bii)
        push!(Zs, Zii)
    end
```
Due to some unknown reason, this network is not able to be fitted correctly - I basically always end up with the same final RMSE, no matter how I choose the hyperparameters. Any ideas on why this happens are appreciated!

The Jupyter notebook can be downloaded [here](https://github.com/sdobber/MLCrashCourse/blob/master/8.%20Intro%20to%20Neural%20Nets%20Julia.ipynb).


***



This notebook is based on the file [Intro to Neural Nets programming exercise](https://colab.research.google.com/notebooks/mlcc/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=introneuralnets-colab&hl=en), which is part of Google's [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/).


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

# Intro to Neural Networks

**Learning Objectives:**
  * Define a neural network (NN) and its hidden layers
  * Train a neural network to learn nonlinearities in a dataset and achieve better performance than a linear regression model

In the previous exercises, we used synthetic features to help our model incorporate nonlinearities.

One important set of nonlinearities was around latitude and longitude, but there may be others.

We'll also switch back, for now, to a standard regression task, rather than the logistic regression task from the previous exercise. That is, we'll be predicting `median_house_value` directly.

## Setup

First, let's load and prepare the data.


```julia
using Plots
using Distributions
gr(fmt=:png)
using DataFrames
using TensorFlow
import CSV
import StatsBase
using PyCall
using Random
using Statistics

sess=Session(Graph())
california_housing_dataframe = CSV.read("california_housing_train.csv", delim=",");
california_housing_dataframe = california_housing_dataframe[shuffle(1:size(california_housing_dataframe, 1)),:];
```

```julia
function preprocess_features(california_housing_dataframe)
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = california_housing_dataframe[
    [:latitude,
     :longitude,
     :housing_median_age,
     :total_rooms,
     :total_bedrooms,
     :population,
     :households,
     :median_income]]
  processed_features = selected_features
  # Create a synthetic feature.
  processed_features[:rooms_per_person] = (
    california_housing_dataframe[:total_rooms] ./
    california_housing_dataframe[:population])
  return processed_features
end

function preprocess_targets(california_housing_dataframe)
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets[:median_house_value] = (
    california_housing_dataframe[:median_house_value] ./ 1000.0)
  return output_targets
end
```




```julia
# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(first(california_housing_dataframe,12000))
training_targets = preprocess_targets(first(california_housing_dataframe,12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(last(california_housing_dataframe,5000))
validation_targets = preprocess_targets(last(california_housing_dataframe,5000))

# Double-check that we've done the right thing.
println("Training examples summary:")
describe(training_examples)
println("Validation examples summary:")
describe(validation_examples)

println("Training targets summary:")
describe(training_targets)
println("Validation targets summary:")
describe(validation_targets)
```




## Building a Neural Network

Use **`hidden_units`** to define the structure of the NN.  The `hidden_units` argument provides a list of ints, where each int corresponds to a hidden layer and indicates the number of nodes in it.  For example, consider the following assignment:

`hidden_units=[3,10]`

The preceding assignment specifies a neural net with two hidden layers:

* The first hidden layer contains 3 nodes.
* The second hidden layer contains 10 nodes.

If we wanted to add more layers, we'd add more ints to the list. For example, `hidden_units=[10,20,30,40]` would create four layers with ten, twenty, thirty, and forty units, respectively.

By default, all hidden layers will use ReLu activation and will be fully connected.


```julia
function construct_columns(input_features)
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: DataFrame of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  out=convert(Matrix, input_features[:,:])
  return convert(Matrix{Float64},out)

end
```


```julia
function create_batches(features, targets, steps, batch_size=5, num_epochs=0)
  """Create batches.

  Args:
    features: Input features.
    targets: Target column.
    steps: Number of steps.
    batch_size: Batch size.
    num_epochs: Number of epochs, 0 will let TF automatically calculate the correct number
  Returns:
    An extended set of feature and target columns from which batches can be extracted.
  """  

    if(num_epochs==0)
        num_epochs=ceil(batch_size*steps/size(features,1))
    end

    names_features=names(features);
    names_targets=names(targets);

    features_batches=copy(features)
    target_batches=copy(targets)

    for i=1:num_epochs

        select=shuffle(1:size(features,1))

        if i==1
            features_batches=(features[select,:])
            target_batches=(targets[select,:])
        else

            append!(features_batches, features[select,:])
            append!(target_batches, targets[select,:])
        end
    end
    return features_batches, target_batches
end
```




```julia
function next_batch(features_batches, targets_batches, batch_size, iter)
  """Next batch.

  Args:
    features_batches: Features batches from create_batches.
    targets_batches: Target batches from create_batches.
    batch_size: Batch size.
    iter: Number of the current iteration
  Returns:
    An extended set of feature and target columns from which batches can be extracted.
  """
    select=mod((iter-1)*batch_size+1, size(features_batches,1)):mod(iter*batch_size, size(features_batches,1));

    ds=features_batches[select,:];
    target=targets_batches[select,:];

    return ds, target
end
```


```julia
function my_input_fn(features_batches, targets_batches, iter, batch_size=5, shuffle_flag=1)
    """Prepares a batch of features and labels for model training.

    Args:
      features_batches: Features batches from create_batches.
      targets_batches: Target batches from create_batches.
      iter: Number of the current iteration
      batch_size: Batch size.
      shuffle_flag: Determines wether data is shuffled before being returned
    Returns:
      Tuple of (features, labels) for next data batch
    """                                          

    # Construct a dataset, and configure batching/repeating.
    ds, target = next_batch(features_batches, targets_batches, batch_size, iter)

    # Shuffle the data, if specified.
    if shuffle_flag==1
      select=shuffle(1:size(ds, 1));
        ds = ds[select,:]
        target = target[select, :]
    end

    # Return the next batch of data.
    return ds, target
end
```




```julia
function train_nn_regression_model(learning_rate,
                     steps,
                     batch_size,
                     hidden_units,
                     keep_probability,
                     training_examples,
                     training_targets,
                     validation_examples,
                     validation_targets)
  """Trains a neural network model of one feature.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A vector describing the layout of the neural network
    keep_probability: A `float`, the probability of keeping a node active during one training step.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create feature columns.
  feature_columns = placeholder(Float32, shape=[-1, size(construct_columns(training_examples),2)])
  target_columns = placeholder(Float32, shape=[-1, size(construct_columns(training_targets),2)])

  # Network parameters
  push!(hidden_units,size(training_targets,2)) #create an output node that fits to the size of the targets
  activation_functions = Vector{Function}(undef, size(hidden_units,1))
  activation_functions[1:end-1] .= z->nn.dropout(nn.relu(z), keep_probability)
  activation_functions[end] = identity #Last function should be idenity as we need the logits  

  # create network - professional template
  Zs = [feature_columns]
  for (ii,(hlsize, actfun)) in enumerate(zip(hidden_units, activation_functions))
        Wii = get_variable("W_$ii"*randstring(4), [get_shape(Zs[end], 2), hlsize], Float32)
        bii = get_variable("b_$ii"*randstring(4), [hlsize], Float32)
        Zii = actfun(Zs[end]*Wii + bii)
        push!(Zs, Zii)
  end

  y=Zs[end]
  loss=reduce_sum((target_columns - y).^2)

  features_batches, targets_batches = create_batches(training_examples, training_targets, steps, batch_size)

  # Advanced gradient decent with gradient clipping
  my_optimizer=(train.AdamOptimizer(learning_rate))
  gvs = train.compute_gradients(my_optimizer, loss)
  capped_gvs = [(clip_by_norm(grad, 5.), var) for (grad, var) in gvs]
  my_optimizer = train.apply_gradients(my_optimizer,capped_gvs)

  run(sess, global_variables_initializer())

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  println("Training model...")
  println("RMSE (on training data):")
  training_rmse = []
  validation_rmse=[]

  for period in 1:periods
    # Train the model, starting from the prior state.
   for i=1:steps_per_period
    features, labels = my_input_fn(features_batches, targets_batches, convert(Int,(period-1)*steps_per_period+i), batch_size)
    run(sess, my_optimizer, Dict(feature_columns=>construct_columns(features), target_columns=>construct_columns(labels)))
   end
    # Take a break and compute predictions.
    training_predictions = run(sess, y, Dict(feature_columns=> construct_columns(training_examples)));    
    validation_predictions = run(sess, y, Dict(feature_columns=> construct_columns(validation_examples)));  

    # Compute loss.
     training_mean_squared_error = mean((training_predictions- construct_columns(training_targets)).^2)
     training_root_mean_squared_error = sqrt(training_mean_squared_error)
     validation_mean_squared_error = mean((validation_predictions- construct_columns(validation_targets)).^2)
     validation_root_mean_squared_error = sqrt(validation_mean_squared_error)
    # Occasionally print the current loss.
    println("  period ", period, ": ", training_root_mean_squared_error)
    # Add the loss metrics from this period to our list.
    push!(training_rmse, training_root_mean_squared_error)
    push!(validation_rmse, validation_root_mean_squared_error)
 end

  println("Model training finished.")

  # Output a graph of loss metrics over periods.
  p1=plot(training_rmse, label="training", title="Root Mean Squared Error vs. Periods", ylabel="RMSE", xlabel="Periods")
  p1=plot!(validation_rmse, label="validation")

  #
  println("Final RMSE (on training data): ", training_rmse[end])
  println("Final RMSE (on validation data): ", validation_rmse[end])

  return y, feature_columns, p1
end
```



## Task 1: Train a NN Model

**Adjust hyperparameters, aiming to drop RMSE below 110.**

Run the following block to train a NN model.  

Recall that in the linear regression exercise with many features, an RMSE of 110 or so was pretty good.  We'll aim to beat that.

Your task here is to modify various learning settings to improve accuracy on validation data.

Overfitting is a real potential hazard for NNs.  You can look at the gap between loss on training data and loss on validation data to help judge if your model is starting to overfit. If the gap starts to grow, that is usually a sure sign of overfitting.

Because of the number of different possible settings, it's strongly recommended that you take notes on each trial to help guide your development process.

Also, when you get a good setting, try running it multiple times and see how repeatable your result is. NN weights are typically initialized to small random values, so you should see differences from run to run.



```julia
 output_function, output_columns, p1 = train_nn_regression_model(
    # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
    0.001, #learning rate
    2000, #steps
    100, #batch_size
    [10, 10], #hidden_units
    1.0, # keep probability
    training_examples,
    training_targets,
    validation_examples,
    validation_targets)
```

    Training model...
    RMSE (on training data):
      period 1: 159.5003388798359
      period 2: 146.17210524272318
      period 3: 115.86667616689647
      period 4: 103.41430171920487
      period 5: 101.759607509385
      period 6: 100.03529275063775
      period 7: 99.1696604548445
      period 8: 97.63196027126274
      period 9: 96.23186697214885
      period 10: 96.03693581788262
    Model training finished.
    Final RMSE (on training data): 96.03693581788262
    Final RMSE (on validation data): 95.03425708133081



```julia
plot(p1)
```




![png](/images/introductiontoneuralnets/output_17_0.png)



## Task 2: Evaluate on Test Data

**Confirm that your validation performance results hold up on test data.**

Once you have a model you're happy with, evaluate it on test data to compare that to validation performance.

Reminder, the test data set is located [here](https://storage.googleapis.com/mledu-datasets/california_housing_test.csv).

Similar to what the code at the top does, we just need to load the appropriate data file, preprocess it and call predict and mean_squared_error.

Note that we don't have to randomize the test data, since we will use all records.


```julia
california_housing_test_data  = CSV.read("california_housing_test.csv", delim=",");
test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

test_predictions  = run(sess, output_function, Dict(output_columns=> construct_columns(test_examples)));
test_mean_squared_error = mean((test_predictions- construct_columns(test_targets)).^2)
test_root_mean_squared_error = sqrt(test_mean_squared_error)

print("Final RMSE (on test data): ", test_root_mean_squared_error)
```

    Final RMSE (on test data): 94.71009531508615

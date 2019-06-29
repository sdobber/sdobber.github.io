---
layout: post
title:  "Conversion of Movie-review data to one-hot encoding"
date:   2018-09-18 19:10:15 +0200
categories: julia dataextraction moviereview tfrecord imdb
---



In the last post, we obtained the files `test_data.h5` and `train_data.h5`, containing text data from movie reviews (from the [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/)). In the next exercise, we need to access a one-hot encoded version of these files, based on a large vocabulary. The following code converts the data and stores it on disk for later use. It takes about two hours to run on my laptop and uses 13GB of storage for the converted file.


The Jupyter notebook can be downloaded [here](https://github.com/sdobber/MLCrashCourse/blob/master/Conversion%20of%20Movie-review%20data%20to%20one-hot%20encoding.ipynb).


***



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

# Conversion of Movie-review data to one-hot encoding

The final exercise of Google's [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) uses the [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/)) to train a Neural Network on movie review data. At this step, we are not concerned with building an input pipeline or implementing an effective handling and storage of the data.  

The following code converts the movie review data we extracted from a ``.tfrecord``-file in the [previous step](https://github.com/sdobber/MLCrashCourse/blob/master/TFrecord%20Extraction.ipynb) to a one-hot encoded matrix and stores it on the disk for later use:


```julia
using HDF5
using JLD
```

The following function handles the conversion to a one-hot encoding:


```julia
# function for creating categorial colum from vocabulary list in one hot encoding
function create_data_columns(data, informative_terms)
   onehotmat=zeros(length(data), length(informative_terms))

    for i=1:length(data)
        string=data[i]
        for j=1:length(informative_terms)
            if occursin(informative_terms[j], string)
                onehotmat[i,j]=1
            end
        end
    end
    return onehotmat
end
```


Let's load the data from disk:


```julia
c = h5open("train_data.h5", "r") do file
   global train_labels=read(file, "output_labels")
   global train_features=read(file, "output_features")
end
c = h5open("test_data.h5", "r") do file
   global test_labels=read(file, "output_labels")
   global test_features=read(file, "output_features")
end
train_labels=train_labels'
test_labels=test_labels';
```

We will use the full vocabulary file, which can be obtained [here](https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt). Put it in the same folder as the Jupyter-file and open it using


```julia
vocabulary=Array{String}(undef, 0)
open("terms.txt") do file
    for ln in eachline(file)
        push!(vocabulary, ln)
    end
end
```

We will now create the test and training features matrices based on the full vocabulary file. This code does not create sparse matrices and takes a long time to run (about 2h on my laptop).


```julia
# This takes a looong time. Only run it once and save the result
train_features_full=create_data_columns(train_features, vocabulary)
test_features_full=create_data_columns(test_features, vocabulary);
```



Save the data to disk. The data takes about 13GB of memory in uncompressed state.


```julia
save("IMDB_fullmatrix_datacolumns.jld", "train_features_full", train_features_full, "test_features_full", test_features_full)
```

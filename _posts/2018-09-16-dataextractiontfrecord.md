---
layout: post
title:  "Data Extraction from TFrecord-files"
date:   2018-09-16 22:07:15 +0200
categories: python tensorflow machinelearning dataextraction
---

The last exercise of the [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) uses text data from movie reviews (from the [ACL 2011 IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/)). The data has been processed as a `tf.Example-format` and can be downloaded as a `.tfrecord-`file from Google's servers.


Tensorflow.jl does not support this file type, so in order to follow the exercise, we need to extract the data from the tfrecord-dataset. This [Jupyter-notebook](https://github.com/sdobber/MLCrashCourse/blob/master/TFrecord%20Extraction.ipynb) contains Python code to access the data, store it as an HDF5 file, and upload it to Google Drive. It can be run directly in [Google's Colaboratory](https://colab.research.google.com/drive/1QFN8BCJJKVraUOk-lmR9fF3bRXRlz9FI#forceEdit=true&sandboxMode=true) Platform without installing Python. We obtain the files `test_data.h5` and `train_data.h5`, which will be used in the next post.


***




```python
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

# TFrecord Extraction


We will load a tfrecord dataset and get the data out to use them with some other framework, for example TensorFlow on Julia.

## Prepare Packages and Parse Function


```python
from __future__ import print_function

import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)
```


```python
def _parse_function(record):
  """Extracts features and labels.

  Args:
    record: File path to a TFRecord file    
  Returns:
    A `tuple` `(labels, features)`:
      features: A dict of tensors representing the features
      labels: A tensor with the corresponding labels.
  """
  features = {
    "terms": tf.VarLenFeature(dtype=tf.string), # terms are strings of varying lengths
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32) # labels are 0 or 1
  }

  parsed_features = tf.parse_single_example(record, features)

  terms = parsed_features['terms'].values
  labels = parsed_features['labels']

  return  {'terms':terms}, labels
```

## Training Data

We start with the training data.


```python
# Create the Dataset object.
ds = tf.data.TFRecordDataset(train_path)
# Map features and labels with the parse function.
ds = ds.map(_parse_function)
```


```python
# Make a one shot iterator
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
```

Direct meta-information on the number of datasets in a ``tfrecord`` file is unfortunately not available. We use the following nice hack to get the total number of entries by iterating over the whole dataset.


```python
sum(1 for _ in tf.python_io.tf_record_iterator(train_path))
```




    25000



Now, we create two vectors to store the output labels and features. Looping over the ``tfrecord``-dataset extracts the entries.


```python
output_features=[]
output_labels=[]

for i in range(0,24999):
  value=sess.run(n)
  output_features.append(value[0]['terms'])
  output_labels.append(value[1])
```

### Export to File

We create a file to export using the h5py package.


```python
import h5py
```


```python
dt = h5py.special_dtype(vlen=str)

h5f = h5py.File('train_data.h5', 'w')
h5f.create_dataset('output_features', data=output_features, dtype=dt)
h5f.create_dataset('output_labels', data=output_labels)
h5f.close()
```

## Test Data

We do a similar action on the test data.


```python
# Create the Dataset object.
ds = tf.data.TFRecordDataset(test_path)
# Map features and labels with the parse function.
ds = ds.map(_parse_function)
```


```python
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
```

The total number of datasets is


```python
sum(1 for _ in tf.python_io.tf_record_iterator(test_path))
```




    25000




```python
output_features=[]
output_labels=[]

for i in range(0,24999):
  value=sess.run(n)
  output_features.append(value[0]['terms'])
  output_labels.append(value[1])
```

### Export to file


```python
dt = h5py.special_dtype(vlen=str)

h5f = h5py.File('test_data.h5', 'w')
h5f.create_dataset('output_features', data=output_features, dtype=dt)
h5f.create_dataset('output_labels', data=output_labels)
h5f.close()
```

## Google Drive Export

Finally, we export the two files containing the training and test data to Google Drive. If necessary, intall the PyDrive package using ``!pip install -U -q PyDrive``. The folder-id is the string of letters and numbers that can be seen in your browser URL after ``https://drive.google.com/drive/u/0/folders/`` when accessing the desired folder.


```python
!pip install -U -q PyDrive
```


```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# PyDrive reference:
# https://googledrive.github.io/PyDrive/docs/build/html/index.html
```


```python
# Adjust the id to the folder of your choice in Google Drive
# Use `file = drive.CreateFile()` to write to root directory
file = drive.CreateFile({'parents':[{"id": "insert_folder_id"}]})
file.SetContentFile('train_data.h5')
file.Upload()
```


```python
# Adjust the id to the folder of your choice in Google Drive
# Use `file = drive.CreateFile()` to write to root directory
file = drive.CreateFile({'parents':[{"id": "insert_folder_id"}]})
file.SetContentFile('test_data.h5')
file.Upload()
```

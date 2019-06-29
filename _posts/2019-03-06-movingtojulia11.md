---
layout: post
title:  "Moving from Julia 0.6 to 1.1"
date:   2019-03-06 21:02:15 +0200
categories: julia tensorflow machinelearning
---


Finally, all files in the [GitHub repository](https://github.com/sdobber/MLCrashCourse) have been updated to be able to run on Julia 1.1. In order to be able to run them (at the time of writing), the developmental versions of the Tensorflow.jl and PyCall.jl packages need to be installed. Some notable changes are listed below:



# General
* The `shuffle` command has been moved to the Random package as `Random.shuffle()`.
* `linspace` has been replaced with `range(start, stop=..., length=...)`.
* To determine the length of a matrix, use `size(...,2)` instead of `length()`.
* For accessing the first and last part of datasets, `head()`  as been replaced with `first()`, and `tail()` with `last()`.
* For arithmetic calculations involving constants and arrays, the better syntax `const .- array` needs to be used instead of `const - array`. In connection with this, a space might be required before the operator; i.e. to avoid confusion with number types use `1 .- array`, and not `1.-array`.
* Conversion of a dataframe `df` to a matrix can be done via `convert(Matrix, df)` instead of `convert(Array, df)`.

# Exercise 8
* The creation of an initially undefined vector etc. now requires the `undef` keyword as in `activation_functions = Vector{Function}(undef, size(hidden_units,1))`.
* An assignment of a function to multiple entries of a vector requires the dot-operator: `activation_functions[1:end-1] .= z->nn.dropout(nn.relu(z), keep_probability)`

# Exercise 10
* For this exercise to work, the MNIST.jl package needs an update. A quick fix can be found in the repository together with the exercise notebooks.
* Instead of `flipdim(A, d)`, use `reverse(A, dims=d)`.
* `indmax` has been replaced by `argmax`.
* Parsing expressions has been moved to the Meta package and can be done by `Meta.parse(...)`.

# Exercise 11
* The behaviour of taking the transpose of a vector has been changed - it now creates an abstract linear algebra object. We use `collect()` on the transpose for functions that cannot handle the new type.
* To test if a string contains a certain word, `contains()` has been replaced by `occursin()`.

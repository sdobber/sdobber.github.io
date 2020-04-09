---
layout: post
title:  "Where to go from here? Announcing `FluxArchitectures`"
date:   2020-04-07 21:02:15 +0200
categories: julia flux tensorflow
---

It's been a while since anything happened on this blog. So what happened in the meantime? Well, for the first I abandoned `Tensorflow.jl` in favour of `Flux.jl`. It seems to be the package that most people are using these days. It has a nice way of setting up models, and is nicely integrated into other parts of the Julia ecosystem as well (say, for example by combining it with differential equations to give [scientific machine learning](https://github.com/SciML/DiffEqFlux.jl)).


So how to proceed from here for improving one's data science skills? As [Casey Kneale](https://github.com/caseykneale) put [it](https://www.linkedin.com/feed/update/urn:li:activity:6650467784071479296?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A6650467784071479296%2C6650555371180093440%29)
> Focus on your analytical reasoning. Learn/brush up on basic statistics, think about them. Learn the limitations of the statistics you learned, otherwise they are useless.  Learn basics of experimental science (experimental design!, the basics of the scientific method) and practice it :). Learn how to collect(scrape, parse, clean, organize, store), and curate data(quality, quantity, how to set up small scale infrastructure). Study some core algorithms, nothing fancy unless your math skills are strong. Try tweaking an algorithm that you think is cool to do something better for a dataset/problem, and study how well it does. Learn statistical methods for comparing experimental outcomes without bias. Make a ton of mistakes, even intentionally.

My point of interest is applying neural network methods to medical data, while brushing up on and learning more about these things. More specifically, I want to try out for predicting future blood glucose levels for patients suffering from diabetes. Here, even a 30 to 45 minute window of a reasonable forecast could provide with an opportunity to mitigate adverse effects.

All of this is centered around modeling of time series. When I started out, I was lacking good examples of slightly more complex models for `Flux.jl` than the standard examples from the documentation. To give a place to some of the models, I started the  [FluxArchitectures](https://github.com/sdobber/FluxArchitectures) repository.

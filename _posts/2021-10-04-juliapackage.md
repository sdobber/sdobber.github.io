---
layout: post
title:  "Tips and tricks to register your first Julia package"
date:   2021-10-04 18:00:00 +0100
categories: julia pkg package registrator documenter
---

[`FluxArchitectures`](https://github.com/sdobber/FluxArchitectures.jl) has finally been published as a Julia package! Installation is now as easy as typing `]` to activate the package manager, and then
```julia
add FluxArchitectures
```
at the REPL prompt. 

While preparing the package, I noticed that a lot of good and useful information about some of the details of registering a package is spread out in different documentations and not easily accessible. In this post, I try to give a walkthrough of the different steps. There is also an older (but highly recommended) video from Chris Rackauckas explaining some of the process:

<iframe width="810" height="405" src="https://www.youtube.com/embed/QVmU29rCjaA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Package Template

Start out by creating a package using the [PkgTemplate.jl](https://github.com/invenia/PkgTemplates.jl)-package. This package takes care of most of the internals for setting everything up. I decided to have automatic workflows for running tests and preparing the documentation done on Github, so the code for creating the template was
```julia
using PkgTemplates

t = Template(; 
    user="UserName",
    dir="~/Code/PkgLocation/",
    julia=v"1.6",
    plugins=[
        Git(; manifest=true, ssh=true),
        GitHubActions(; x86=true),
        Codecov(),
        Documenter{GitHubActions}(),
    ],
)

t("PkgName")
```
Of course, change `UserName` to your Github user name, `PkgLocation` and `PkgName` to the locaction and package name of your choice. A good starting point is [this place](https://invenia.github.io/PkgTemplates.jl/stable/user/#A-More-Complicated-Example-1) of the `PkgTemplates.jl` documentation. There are a lot of options available, which are described in the documentation.

Depending on which Julia version you want to run your tests, you might need to edit the `.github/workflows/CI.yml` file. For example, to run tests also on the latest Julia nightly, edit the `matrix` section:
```yml
matrix:
        version:
          - '1.6'
          - 'nightly'
```
To allow for errors in the nightly version, edit the `steps` section to include a `continue-on-error` statement:
```yml
    - uses: julia-actions/julia-runtest@v1
      continue-on-error: {% raw %}${{ matrix.version == 'nightly' }}{% endraw %}
```
(The full file can be found [here](https://github.com/sdobber/FluxArchitectures.jl/blob/master/.github/workflows/CI.yml).)


## Develop Code and Tests

The next step is of course to fill the template with Julia code (which goes into the `src` folder) and tests for your code (residing in the `test` folder).


## Documentation

A package should have some documentation describing its features. It goes into the `docs` folder in the form of Markdown files. The `make.jl` file takes care of preparing everything, though we need to tell it the desired sidebar structure of our documentation. This goes into the `pages` keyword:
```julia
pages=[
        "Home" => "index.md",
        "Examples" => "examples/examples.md",
        "Exported Functions" => "functions.md",
        "Models" =>
                    ["Model 1" => "models/model1.md",
                     "Model 2" => "models/model2.md"],
        "Reference" => "reference.md",
    ]
```
All the files on the right side of the pairs need to exist in the `docs/src`-directory (or subfolders thereof). It is strongly advised to run the `make.jl` file locally and see if everything works.

See the [`Documenter.jl`-documentation](https://juliadocs.github.io/Documenter.jl/stable/man/guide/#Adding-Some-Docstrings) on how to automatically add docstrings from the package source code, add links to other sections etc. One can also learn a lot by looking at the code for other packages' documentations.


## External Data

In case your package includes some larger files with example data etc., it is a good idea to include them via Julia's Artifact system. This consists of the following steps:

* Create a `.tar.gz` archive of your files.
* Upload the files to an accessible location (e.g. in a separate GitHub repository). For FluxArchitectures, I used a [FA_data  repository](https://github.com/sdobber/FA_data).
* Create an `Artifacts.toml` file in your package folder containing information about the files to download.
* Access your files in Julia by finding their location through the Artifact system - Julia will automatically take care of downloading them, storing them and making them accesible. This works like
    ```julia
    using Pkg.Artifacts
    rootpath = artifact"DatasetName"
    ```

These steps are described in detail in the [`Pkg`-documentation](https://pkgdocs.julialang.org/v1/artifacts/#Basic-Usage). For the `toml`-file, a file SHA and git tree SHA are needed. They can be produced by Julia itself - see the linked documentation. If one adds the `lazy = true` keyword to the section containing the git tree SHA, the data is only downloaded when the user requests it for the first time.

A common mistake is to "just" copy-paste the GitHub URL to the archive into the `Artifacts.toml` file, which will not work. Make sure to use a link to the "raw" data, which usually can be obtained by inserting a `raw` into the GitHub URL, for example as in `https://github.com/sdobber/FA_data/raw/main/data.tar.gz`.


## Check Requirements

It is a good idea to check the requirements for new packages, which can be found in the [`RegistryCI.jl`-documentation](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/). This documents gives some hints about proper naming etc. 


## Publish to GitHub & Add JuliaRegistrator

If not done already, make sure that your package is available on GitHub.

Click on the "install app" button on [`JuliaRegistrator`'s  page](https://github.com/JuliaRegistries/Registrator.jl/blob/master/README.md#install-registrator) and allow the bot to access your package repository.


## Stable vs Dev Documentation

The `CI.yml` file created in the package template contains a workflow that will build your documentation and make it available in your repository. The documentation is pushed to a new branch called `gh-pages`. You might need to tell GitHub to use this branch as a "Github Pages" Site. Follow the instructions in this [GitHub documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), and set the "publishing source" to the `gh-pages` branch.

With the default settings, only documentation for the current development version of the package will be created. If you also want to create and keep documentation for each tagged version, you need to create and add a key pair to the `DOCUMENTER_KEY` secret of the GitHub repository. The easiest way I found for producing those is to install the package called `DocumenterTools.jl` and run
```julia
using DocumenterTools
DocumenterTools.genkeys()
```
in the Julia REPL.

The REPL-output will present you with two strings that need to be pasted into different places:
* The first key needs to be added as a public key to your repository, see [this documentation](https://docs.github.com/en/developers/overview/managing-deploy-keys#setup-2) and start at step 2.
* The second key needs to be added as a repository secret. Follow [this document](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository). The name of the secret needs to be `DOCUMENTER_KEY`, and the value is the output string from Julia.


## Register Package

On GitHub, open a new issue in your repository, and write `@JuliaRegistrator register` in the comment area. The JuliaRegistrator bot will pick this up, and after a 3 day waiting period, your new package hopefully gets added to the general Julia registry. 
# ttpca

*TTPCA* is a dimensionality reduction and visualization method for scalar functions that is based on the [tensor train (TT)](https://dl.acm.org/citation.cfm?id=2079149) decomposition.

<img src="https://github.com/rballester/ttpca/raw/master/images/hyperslices.jpg" width="800" title="Plot matrix">

The method is described in the paper [*Visualization of High-dimensional Scalar Functions Using Principal Parameterizations*](https://arxiv.org/abs/1809.03618):

```
@ARTICLE{BP:18,
   author = {{Ballester-Ripoll}, R. and {Pajarola}, R.},
    title = "{Visualization of High-dimensional Scalar Functions Using Principal Parameterizations}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1809.03618},
 primaryClass = "cs.GR",
 keywords = {Computer Science - Graphics, Computer Science - Machine Learning, Computer Science - Multimedia, Computer Science - Numerical Analysis},
     year = 2018,
    month = sep}
```

## Requirements

*TTPCA* relies on [*ttrecipes*](https://github.com/rballester/ttrecipes/), a library of auxiliary TT functions. Please install *ttrecipes* first as indicated in its [README](https://github.com/rballester/ttrecipes#installation).

After installing *ttrecipes*, you should be able to install *TTPCA* as follows:

```
git clone https://github.com/rballester/ttpca.git
pip install -e ttpca
```

(note: it's always highly recommended to work in a conda or pip *virtualenvironment*!)

## Usage

### Back-end

All numerical functionality is encapsulated in the function `reduce()`. It simply takes a TT and a list of k target variables and produces a TT with (k+1) dimensions (the last dimension has size 3) that contains the principal parameterization (i.e. embedding into a 3D Euclidean space) for those k variables:

```
ttpca.reduce(t, modes=[0, 1])  # This will return a parameterized surface in 3D
```

### Front-end

We provide a visualization front-end using [PyQtGraph](http://pyqtgraph.org/) and *PyOpenGL* that allows 3D navigation of plot matrices, curve arrays, and various interactions as described in [the paper](https://arxiv.org/abs/1809.03618).
# MagPy: Causal Discovery and Effect Estimation Framework

MagPy is a Python framework for causal discovery and effect estimation, aimed at uncovering causal relationships in observational data and estimate the impact of interventinos/counterfactuals.

This is an experimental project, currently under active development.

## Installation

```bash
pip install causal-magpy
```

or

```bash
git clone https://github.com/ergodic-ai/magpy.git
cd magpy
pip install .
```

or

```bash
pip install git+https://github.com/ergodic-ai/magpy.git
```

## 1. AStarSearch

The `AStarSearch` class implements the A\* search algorithm for Bayesian network structure learning. Our goal with this algorithm is to be able to decouple the scoring function from the search algorithm.

A simple graph search using the BIC score:

```python
import pandas as pd
from magpy.search.astar import AStarSearch, bic_score_node

data: pd.DataFrame = ...

astar = AStarSearch(data)
astar.run_scoring(func=bic_score_node, parallel=False)
result = astar.search()
```

Or a more complex search using a polynomial scoring function:

```python
from typing import Optional
import numpy
from sklearn.preprocessing import PolynomialFeatures


def bic_score_node_poly(
    y: numpy.ndarray,
    X: Optional[numpy.ndarray] = None,
    node: str | None = None,
    parent_set: set | None = None,
    degree=3,
    include_bias=True,
):
    n = len(y)

    if X is None:
        residual = numpy.sum(y**2)
        dof = 0

    else:
        Xf = PolynomialFeatures(degree=degree, include_bias=include_bias).fit_transform(
            X
        )
        n, dof = Xf.shape
        _, residual, _, _ = numpy.linalg.lstsq(a=Xf, b=y, rcond=None)

    bic = n * numpy.log(residual / n) + dof * numpy.log(n)
    return bic.item()

astar = AStarSearch(data)
astar.run_scoring(func=bic_score_node, parallel=False)
result = astar.search()
```

The philosophy behind this algorithm is that:

- You know the data! And that should be reflected in your choice of scoring function. As long as it adheres to the above interface, it will work.
- The algorithm executes the search.

## 2. Skeleton Learning

As part of the causal discovery pipeline, we need to learn the skeleton of the graph. This is typically done using the PC algorithm.
The PC algorithm is essentially a wrapper around a conditional independence test, which we'll call an oracle.

An oracle is an object that tests whether a certin variable X is independent of Y given a set of covariates Z: $\(X \perp Y | Z\)$.

We are currently supporting two main oracles:

- `MixedDataOracle`, which can handle continuous, binary, and categorical variables, and assumes linear relationships.
- `BaseOracle`, which is the base class to implement your own oracle.

### 2.1 BaseOracle

Similarly to the AStar algorithm, our objective with the `BaseOracle` is to decouple the underlying "learner" from the hypothesis testing.

```python
from magpy.oracles.oracles import BaseOracle, linear
import numpy

z = numpy.random.randn(1000)
x = z + numpy.random.randn(1000) * 0.1
y = x + numpy.random.randn(1000) * 0.1

df = pandas.DataFrame({"x": x, "y": y, "z": z})


oracle = BaseOracle(df, threshold=0.05, learner=linear)

print("linear: ")
print("independent: ", oracle("y", "x", ["z"]))
print("pvalue: ", oracle._run("y", "x", ["z"]))
```

The learner object is a function that accepts X, y and returns the RSS of a regression and the number of degrees of freedom within the model..

Here's an example of how to implement a learner based on polynomial regression:

```python
from typing import Optional, Union
import pandas
import numpy
from sklearn.preprocessing import PolynomialFeatures


def poly_rss(
    X: Union[pandas.DataFrame, None],
    y: pandas.Series,
    node: Optional[str] = None,
    parent_set: Optional[set] = None,
    degree: int = 3,
):
    """Perform polynomial regression and return residual sum of squares and degrees of freedom.

    Args:
        X (Union[pandas.DataFrame, None]): Feature matrix. If None, only intercept is used.
        y (pandas.Series): Target variable.
        node (Optional[str], optional): Node name, not used but included for API compatibility. Defaults to None.
        parent_set (Optional[set], optional): Parent set, not used but included for API compatibility. Defaults to None.
        degree (int, optional): Degree of polynomial features. Defaults to 3.

    Returns:
        tuple: (rss, p)
            rss (float): Residual sum of squares from polynomial regression
            p (int): Number of parameters (degrees of freedom) in the model
    """

    if X is None:
        X_values = numpy.ones(shape=(y.shape[0], 1))

    else:
        X_values = X.values
        X_values = PolynomialFeatures(degree=degree).fit_transform(X_values)

    y_values: numpy.ndarray = y.values  # type: ignore

    _, [rss], _, _ = numpy.linalg.lstsq(X_values, y_values, rcond=None)

    p = X_values.shape[1]

    return rss, p
```

Using this learner, we can now model more complex relationships:

```python
from magpy.oracles.oracles import BaseOracle, linear
import numpy

z = numpy.random.randn(1000)
x = z**2 + numpy.random.randn(1000) * 0.1
y = x**2 + numpy.random.randn(1000) * 0.1

df = pandas.DataFrame({"x": x, "y": y, "z": z})


oracle = BaseOracle(df, threshold=0.05, learner=linear)

print("linear: ")
print("independent: ", oracle("y", "x", ["z"]))
print("pvalue: ", oracle._run("y", "x", ["z"]))



oracle = BaseOracle(df, threshold=0.05, learner=poly_rss)

print("polynomial: ")
print("independent: ", oracle("y", "x", ["z"]))
print("pvalue: ", oracle._run("y", "x", ["z"]))

```

Again, our philosophy is that you know your data best, and you should be able to implement a learner that best captures the relationship you are interested in.

## 2.2 MixedDataOracle

We developed this oracle because dealing with mixed data types is a pain. One-hot encoding and praying isn't necessarily a good idea, and this provides a quick way to handle this with some science behind it.

This is losely based on the work of [Tsagris et al](https://pmc.ncbi.nlm.nih.gov/articles/PMC6428307/).

Here's a silly example:

```python
from magpy.oracles.mixed import MixedDataOracle
import pandas
import numpy

z = numpy.random.randn(1000)
x = z + numpy.random.randn(1000)
y = z + numpy.random.randn(1000)
y_d = [str(int(elm.clip(-2, 2))) for elm in y]

df = pandas.DataFrame({"x": x, "y": y_d, "z": z})

oracle = MixedDataOracle(df, threshold=0.05)
print("Independent: ", oracle("y", "x", ["z"]))
print("pvalue: ", oracle._run("y", "x", ["z"]))
```

The oracle automatically tags variables as continuous or binary/categorical based on the data. If you want an integer to be treated as categorical, make sure to cast it as a string or object before.

### 2.3 PC Algorithm

We haven't implemented the full PC algorithm yet, our goal is to actually separate it into the various components:

1. Skeleton search
2. V-structures detection
3. Further edge orientation

For now let's stick to skeleton search:

```python
from magpy.search.pcskeleton import pc_skeleton
from magpy.oracles.oracles import BaseOracle, linear, cubic
import pandas
from typing import Callable



def pc_skeleton_magpy(
    X: pandas.DataFrame,
    learner: Callable = linear,
    intersection_or_union: str = "union",
):
    oracle = BaseOracle(X, threshold=0.05, learner=learner)
    skeleton, sepsets = pc_skeleton(
        oracle, X.columns, intersection_or_union=intersection_or_union
    )
    return skeleton

```

There are a number of niceties inside the PC skeleton implementation, we'll update the documentation soon to expose them. If you
are working with continuous data, we strongly recommend composing the PC skeleton algorithm with a direct search method for orientation.

## Composing

Our goal here is to allow for composition of different parts of the causal discovery pipeline. For instance, this is how you will perform a skeleton search using the PC skeleton:

```python
from magpy.search.pcskeleton import pc_skeleton
from magpy.search.astar import AStarSearch, bic_score_node_poly
import pandas
from typing import Callable


def full_composite_search(
    X: pandas.DataFrame,
    learner_pc: Callable = cubic,
    learner_astar: Callable = bic_score_node_poly,
    intersection_or_union: str = "union",
    force=True,
):
    # Fix colinearity
    fix_colinearity(X)

    skeleton = pc_skeleton_magpy(
        X, intersection_or_union=intersection_or_union, learner=learner_pc
    )

    priors = skeleton.copy() * 0
    priors.loc["known_parent", "known_child"] = 1

    astar = AStarSearch(X, super_graph=skeleton, include_graph=priors)
    astar.run_scoring(parallel=False, func=learner_astar, verbose=False)
    y_df = astar.search()

    return y_df

```

## 3. Effect Estimation

Under deep development. The SF and the Diabetes notebooks are good starting points.

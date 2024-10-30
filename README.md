# MagPy: Causal Discovery and Effect Estimation Framework

MagPy is a Python framework for causal discovery and effect estimation, aimed at uncovering causal relationships in observational data and estimate the impact of interventinos/counterfactuals. 

## Project Scope

The primary focus of MagPy is causal discovery, which involves:

1. Learning the structure of causal graphs from observational data
2. Handling mixed data types (continuous, categorical, binary)
3. Implementing constraint-based and score-based algorithms
4. Providing tools for causal inference and intervention analysis

## Installation

```bash
git clone https://github.com/ergodic-ai/magpy.git
cd magpy
pip install .
```

or

```bash
pip install git+https://github.com/ergodic-ai/magpy.git
```

## Main Packages and Usage

### 1. AStarSearch

The `AStarSearch` class implements the A\* search algorithm for Bayesian network structure learning.

```python
import pandas as pd
from magpy.search.astar import AStarSearch, bic_score_node

data: pd.DataFrame = ...

astar = AStarSearch(data)
astar.run_scoring(func=bic_score_node, parallel=True)
result = astar.search()
```

Key features:

- Custom scoring functions
- Parallel processing
- Constraint-based search with super_graph and include_graph


### 2. Oracles

We are introducing a variety of conditional independence tests - oracles. 
As an example, the `MixedDataOracle` class handles conditional independence testing for mixed data types.

```python
from magpy.oracles.mixed import MixedDataOracle

oracle = MixedDataOracle(data, threshold=0.05)
independent = oracle(x, y, Z)
```

  
### 3. PC Skeleton Algorithm

The `pc_skeleton` function implements the PC algorithm for learning the skeleton of a causal graph.

```python
from magpy.search.pcskeleton import pc_skeleton
from magpy.oracles.mixed import MixedDataOracle
import pandas

data: pandas.DataFrame = ... 

nodes = list(data.columns)
oracle = MixedDataOracle(data, threshold=0.05)
skeleton, sepsets = pc_skeleton(oracle, nodes)

```

### 3. Quick Start

For continuous data:

```python
import pandas as pd
from magpy.search.astar import AStarSearch, bic_score_node

data = pd.read_csv('your_data.csv')
astar = AStarSearch(data)
astar.run_scoring(func=bic_score_node, parallel=True)
result = astar.search()


import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_pandas_adjacency(result, create_using=nx.DiGraph)
nx.draw(G, with_labels=True)plt.show()
```

For categorical data, see the `sf.ipynb` notebook.

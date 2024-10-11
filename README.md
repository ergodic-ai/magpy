# MagPy: Causal Discovery Framework

MagPy is a Python framework for causal discovery, aimed at uncovering causal relationships in observational data. It implements various algorithms and techniques for learning the structure of causal graphs from data.

## Project Scope

The primary focus of MagPy is causal discovery, which involves:

1. Learning the structure of causal graphs from observational data
2. Handling mixed data types (continuous, categorical, binary)
3. Implementing constraint-based and score-based algorithms
4. Providing tools for causal inference and intervention analysis

## Main Packages and Usage

### 1. AStarSearch

The `AStarSearch` class implements the A\* search algorithm for Bayesian network structure learning.

```python
from magpy.search.astar import AStarSearch
Initialize with data
astar = AStarSearch(data)
astar.run_scoring(func=my_custom_scoring_function, parallel=True)
result = astar.search()
```

Key features:

- Custom scoring functions
- Parallel processing
- Constraint-based search with super_graph and include_graph

### 2. PC Skeleton Algorithm

The `pc_skeleton` function implements the PC algorithm for learning the skeleton of a causal graph.

```python
from magpy.search.pcskeleton import pc_skeleton
skeleton, sepsets = pc_skeleton(ci_test, nodes)
```

Key features:

- Customizable conditional independence tests
- Parallel execution for improved performance

### 3. MixedDataOracle

The `MixedDataOracle` class handles conditional independence testing for mixed data types.

```python
from magpy.oracles.mixed import MixedDataOracle
Initialize oracle
oracle = MixedDataOracle(data, threshold=0.05)
independent = oracle(x, y, Z)
```

Key features:

- Handles continuous, categorical, and binary data types
- Customizable threshold for conditional independence

### 3. Quick Start

```python
import pandas as pd
from magpy.search.astar import AStarSearch
from magpy.oracles.mixed import MixedDataOracle
Load your data
data = pd.read_csv('your_data.csv')
oracle = MixedDataOracle(data)
astar = AStarSearch(data)
astar.run_scoring(func=oracle, parallel=True)# Perform search
result = astar.search()# Visualize result
import networkx as nx
import matplotlib.pyplot as plt
G = nx.from_pandas_adjacency(result, create_using=nx.DiGraph)
nx.draw(G, with_labels=True)plt.show()
```

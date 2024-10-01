import pandas as pd
import numpy as np
import pytest
from magpy.search.astar import AStarSearch, bic_score_node


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        }
    )
    X["D"] = X["A"] + X["B"] + np.random.randn(100) * 0.1
    return X


def test_astar_initialization(sample_data):
    astar = AStarSearch(sample_data)
    assert astar.nodes == list(sample_data.columns)
    assert astar.max_parents == len(sample_data.columns)
    assert astar.include_graph.shape == (4, 4)
    assert astar.super_graph.shape == (4, 4)


def test_astar_potential_parent_set(sample_data):
    astar = AStarSearch(sample_data)
    potential_parents = astar._potential_parent_set("D")
    assert potential_parents == {"A", "B", "C"}


def test_astar_include_parents(sample_data):
    include_graph = pd.DataFrame(
        np.zeros((4, 4)), columns=sample_data.columns, index=sample_data.columns
    )
    include_graph.loc["A", "D"] = 1
    astar = AStarSearch(sample_data, include_graph=include_graph)
    include_parents = astar._include_parents("D")
    assert include_parents == {"A"}


def test_astar_single_node_score_args(sample_data):
    astar = AStarSearch(sample_data)
    args = astar._single_node_score_args("D")
    assert len(args) == 8  # 2^3 combinations of potential parents
    assert all(
        "X" in arg and "y" in arg and "node" in arg and "parent_set" in arg
        for arg in args
    )


def test_astar_run_scoring(sample_data):
    astar = AStarSearch(sample_data)
    astar.run_scoring(parallel=False)
    assert astar.parent_graph is not None


def test_astar_search(sample_data):
    astar = AStarSearch(sample_data)
    astar.run_scoring(parallel=False)
    result = astar.search()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 4)
    assert set(result.index) == set(sample_data.columns)
    assert set(result.columns) == set(sample_data.columns)


def test_astar_get_adjacency_matrix(sample_data):
    astar = AStarSearch(sample_data)
    parent_dict = {"A": set(), "B": set(), "C": set(), "D": {"A", "B"}}
    adj_matrix = astar.get_adjacency_matrix(parent_dict)
    assert adj_matrix.loc["A", "D"] == 1
    assert adj_matrix.loc["B", "D"] == 1
    assert adj_matrix.loc["C", "D"] == 0
    assert adj_matrix.sum().sum() == 2


def test_bic_score_node():
    y = np.random.randn(100)
    X = np.random.randn(100, 2)
    score = bic_score_node(y, X)
    assert isinstance(score, float)

    score_without_X = bic_score_node(y)
    assert isinstance(score_without_X, float)


if __name__ == "__main__":
    pytest.main()

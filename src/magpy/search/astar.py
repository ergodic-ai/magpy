import pandas
import numpy
from typing import Optional
import networkx
import numpy
import itertools as it
import logging
from joblib import Parallel, delayed
from pydantic import BaseModel
import heapq
from tqdm.auto import tqdm

INF = float("inf")
NEGINF = float("-inf")


class PQItem(BaseModel):
    visited: set
    parent_dict: dict
    heuristic: float
    score: float
    removed: bool = False

    def __lt__(self, other):
        return self.score + self.heuristic < other.score + other.heuristic

    def __le__(self, other):
        return self.score + self.heuristic <= other.score + other.heuristic

    def __gt__(self, other):
        return self.score + self.heuristic > other.score + other.heuristic

    def __ge__(self, other):
        return self.score + self.heuristic >= other.score + other.heuristic


class PriorityQueue:
    def __init__(self):
        self.n = 0
        self.pq = []
        self.entries = {}

    def __len__(self):
        return self.n

    def push(self, item: PQItem, weight: float):
        entry = item  # [weight, item]
        # print(self.pq)
        visited = frozenset(item.visited)
        self.entries[visited] = entry
        heapq.heappush(self.pq, entry)
        self.n += 1

    def get(self, visited: set):
        visited = frozenset(visited)
        return self.entries.get(visited, None)

    def delete(self, visited):
        visited = frozenset(visited)
        entry = self.entries.pop(visited)
        entry.removed = True
        self.n -= 1

    def empty(self):
        return self.n == 0

    def pop(self):
        while not self.empty():
            item = heapq.heappop(self.pq)
            if item.removed == False:
                visited = frozenset(item.visited)
                del self.entries[visited]
                self.n -= 1
                weight = item.score + item.heuristic
                return weight, item
        else:
            raise KeyError("Attempting to pop from an empty priority queue")


class ParentGraph:
    def __init__(self, nodes, results):
        self.nodes = nodes
        self.graph = {}
        for node in nodes:
            node_results = [result for result in results if result["node"] == node]
            node_results = sorted(node_results, key=lambda x: x["score"])
            self.graph[node] = node_results

        # self.trim()

    def _trim_node(self, node: str):
        """We will transverse the list of parents, and for each parent list we will remove the subsequent
        parent lists that have a higher score than the current parent list and include the original parent list.
        For example:
        [ [(a,b,c), -20 ], [(a,b,d), -15], [(a,b,c,e), -10]]

        We will exclude (abcd) since it has worse score than (abc) and it also contains (abc).

        Remember, good scores are lower scores.

        """

        node_results = self.graph[node]
        new_node_results = []
        exclusion = []
        for i in range(len(node_results)):
            parent_set = node_results[i]["parent_set"]

            if i in exclusion:
                continue

            new_node_results.append(node_results[i])
            for j in range(i + 1, len(node_results)):
                if parent_set.issubset(node_results[j]["parent_set"]):
                    logging.info("Excluding parent set")
                    logging.info(f"Parent Set: {node_results[j]['parent_set']}")
                    logging.info(f"Score: {node_results[j]['score']}")
                    logging.info(f"Pivot Set: {parent_set}")
                    logging.info(f"Pivot Score: {node_results[i]['score']}")
                    logging.info("****")
                    exclusion.append(j)

        self.graph[node] = new_node_results

    def trim(self):
        for node in self.graph:
            self._trim_node(node)

    def get_best_parent_set(self, node: str):
        return self.graph[node][0]["parent_set"]

    def query(self, node: str, structure: set = set(), exclude: set = set(), n=1):
        """
        This function will return the best parent set for a node that includes the structure.

        Parameters
        ----------
        node: str
            The node to query
        structure: set
            The structure to query

        Returns
        -------
        dict
            The result of the query
        """

        prev = [None]
        node_results = self.graph[node]
        for result in node_results:
            if result["parent_set"].issubset(structure):
                if not (len(exclude) and exclude.issubset(result["parent_set"])):
                    prev.append(result)
                    if len(prev) > n:
                        return prev[-1]

        return prev[-1]

    def query_exact_structure(self, node: str, structure: set):
        """
        This function will return the score of the exact structure for a node

        Parameters
        ----------
        node: str
            The node to query
        structure: set
            The structure to query

        Returns
        -------
        dict
            The result of the query
        """
        node_results = self.graph[node]
        for result in node_results:
            if result["parent_set"] == structure:
                return result

        return None

    def query_best(self, node: str):
        """
        This function will return the best parent set for a node that includes the structure.

        Parameters
        ----------
        node: str
            The node to query
        structure: set
            The structure to query

        Returns
        -------
        dict
            The result of the query
        """
        node_graph = self.graph[node]
        if len(node_graph) == 0:
            return None
        return node_graph[0]

    def heuristic(self, nodes: set, noise=0):
        """
        This function will return the heuristic score for a set of nodes

        Parameters
        ----------
        nodes: set
            The set of nodes to get the heuristic score for

        Returns
        -------
        float
            The heuristic score for the set of nodes
        """
        add_noise = 1
        if noise > 0:
            add_noise = numpy.random.normal(0, noise)
            add_noise = numpy.exp(add_noise)

        score = 0
        for node in nodes:
            query_result = self.query_best(node)
            if query_result is not None:
                score += query_result["score"]

        return score * add_noise

    def path_extension(self, visited: set, parent_dict: dict, score: float):
        d = len(self.nodes)

        while True:
            extended = False
            additional_nodes = set(self.nodes) - visited
            # set of nodes that are not visited
            for node in additional_nodes:
                # for each node that is not visited, we query their best parent set
                # parents, best_score = query_best_structure(parent_graphs[i], U)
                item = self.query(node, visited)
                if item is None:
                    continue
                parents = item["parent_set"]
                best_score = item["score"]

                # now we ask the question, is the best score for the node conditional on the visited nodes
                # also the best score overall
                queried_best = self.query_best(node)
                if queried_best is None:
                    overall_best_score = INF
                else:
                    overall_best_score = queried_best["score"]

                if best_score == overall_best_score:
                    score += overall_best_score
                    visited.add(node)
                    parent_dict = parent_dict.copy()
                    parent_dict[node] = parents

                    extended = True
                    # print("extended")
                    break
            if not extended:
                break

        return visited, parent_dict, score

    def score_graph(self, mat: pandas.DataFrame):
        """
        This function will score a graph

        Parameters
        ----------
        mat: pandas.DataFrame
            The adjacency matrix of the graph

        Returns
        -------
        float
            The score of the graph
        """
        score = 0
        for node in self.nodes:
            parents = set(list(mat[mat[node] == 1].index))
            query_result = self.query_exact_structure(node, parents)
            if query_result is not None:
                print(node, parents, query_result["score"])
                score += query_result["score"]
            else:
                print(node, parents, "None")
        return score


def bic_score_node(
    y: numpy.ndarray,
    X: Optional[numpy.ndarray] = None,
    node: Optional[str] = None,
    parent_set: Optional[set] = None,
):
    n = len(y)

    if X is None:
        residual = numpy.sum(y**2)
        dof = 0

    else:
        n, dof = X.shape
        _, residual, _, _ = numpy.linalg.lstsq(a=X, b=y, rcond=None)

    bic = n * numpy.log(residual / n) + dof * numpy.log(n)
    return bic.item()


def validate_include_graph(include_graph: pandas.DataFrame, nodes: list):
    assert include_graph.shape == (
        len(nodes),
        len(nodes),
    ), "include_graph must be a square matrix with shape equal to the number of nodes"
    assert all(
        include_graph.columns == nodes
    ), "include_graph must have the same columns as the data"

    # print(include_graph)
    nxgraph = networkx.from_pandas_adjacency(
        include_graph, create_using=networkx.DiGraph
    )
    assert networkx.is_directed_acyclic_graph(
        nxgraph
    ), "include_graph must be a directed acyclic graph"


def validate_super_graph(super_graph: pandas.DataFrame, nodes: list):
    assert super_graph.shape == (
        len(nodes),
        len(nodes),
    ), "super_graph must be a square matrix with shape equal to the number of nodes"
    assert all(
        super_graph.columns == nodes
    ), "super_graph must have the same columns as the data"

    assert all(
        numpy.diag(super_graph) == 0
    ), "super_graph must have zeros on the diagonal"


class AStarSearch:
    """A* Search for Bayesian Network Structure Learning
    Parameters
    ----------
    X: pandas.DataFrame
        The data to learn the structure from
    super_graph: pandas.DataFrame
        The skeleton of the graph to search over
    include_graph: pandas.DataFrame
        The graph that tells us which edges to include in the graph
    max_parents: int
        The maximum number of parents for each node
    """

    def __init__(
        self,
        X: pandas.DataFrame,
        super_graph: Optional[pandas.DataFrame] = None,
        include_graph: Optional[pandas.DataFrame] = None,
        max_parents: Optional[int] = None,
        path_extension: Optional[bool] = True,
    ):
        self.data = X
        self.nodes = list(X.columns)

        n, d = X.shape

        # include graph is a directed acyclic graph that tells us domain expertise about which
        # nodes edges should be included in the graph
        if include_graph is None:
            include_graph_array = numpy.zeros((d, d))
            include_graph = pandas.DataFrame(
                include_graph_array, index=self.nodes, columns=self.nodes
            )
        else:
            validate_include_graph(include_graph, self.nodes)

        self.include_graph = include_graph
        logging.info(f"Include Graph: {include_graph}")

        # super graphp is a skeleton of the graph that we will use to search for the optimal
        # graph structure
        if super_graph is None:
            super_graph_array = numpy.ones((d, d))
            super_graph_array[numpy.diag_indices_from(super_graph_array)] = 0
            super_graph = pandas.DataFrame(
                super_graph_array, index=self.nodes, columns=self.nodes
            )

        else:
            validate_super_graph(super_graph, self.nodes)

        self.super_graph = super_graph
        logging.info(f"Super Graph: {super_graph}")

        if max_parents is None:
            max_parents = d

        self.max_parents = max_parents
        self.parent_graph = None
        self.path_extension = path_extension
        self.parent_dict = None

    def _potential_parent_set(self, node: str):
        """
        Returns the potential parent set for a given node

        Parameters
        ----------
        node: str
            The node to find the potential parent set for

        Returns
        -------
        set
            The potential parent set for the given node
        """
        return set(self.super_graph[self.super_graph[node] == 1].index) - {node}

    def _include_parents(self, node: str):
        """
        Returns the potential parent set for a given node

        Parameters
        ----------
        node: str
            The node to find the potential parent set for

        Returns
        -------
        set
            The potential parent set for the given node
        """
        return set(self.include_graph[self.include_graph[node] == 1].index) - {node}

    def _single_node_score_args(self, node: str):
        """
        Returns the arguments for scoring a single node

        Parameters
        ----------
        node: str
            The node to score

        Returns
        -------
        list
            A list of tuples containing the arguments for scoring the node:
            (node, parent_set, X[node], X[list[parent_set + include_parents]])
        """

        X = self.data

        args = []  # list of tuples

        parent_set = self._potential_parent_set(node)
        logging.info(f"Parent Set for {node}: {parent_set}")

        include_parents = self._include_parents(node)
        logging.info(f"Include Parents for {node}: {include_parents}")

        parents = parent_set - include_parents
        logging.info(f"Parents for {node}: {parents}")

        for j in range(len(parents) + 1):
            if j == 0:
                if len(include_parents) == 0:
                    d = {
                        "X": None,
                        "y": X[[node]].copy().values,
                        "node": node,
                        "parent_set": set(),
                    }
                    args.append(d)
                else:
                    d = {
                        "X": X[list(include_parents)].copy().values,
                        "y": X[[node]].copy().values,
                        "node": node,
                        "parent_set": include_parents,
                    }
                    args.append(d)

            else:
                for p in it.combinations(parents, j):
                    extended_p = include_parents.union(p)
                    d = {
                        "X": X[list(extended_p)].copy().values,
                        "y": X[[node]].copy().values,
                        "node": node,
                        "parent_set": extended_p,
                    }
                    args.append(d)

        return args

    def _full_score_args(self):
        """
        Returns the arguments for scoring all nodes

        Returns
        -------
        list
            A list of tuples containing the arguments for scoring all nodes:
            (node, parent_set, X[node], X[list[parent_set + include_parents]])
        """
        args = []
        for node in self.nodes:
            args.extend(self._single_node_score_args(node))

        return args

    def _score(self, args, parallel=True, n_jobs=4, func=None):
        """
        Scores the nodes

        Parameters
        ----------
        args: list
            A list of tuples containing the arguments for scoring the node:
            (node, parent_set, X[node], X[list[parent_set + include_parents]])

        parallel: bool
            Whether to score in parallel

        Returns
        -------
        list
            A list of tuples containing the arguments for scoring the node:
            (node, parent_set, X[node], X[list[parent_set + include_parents]])
        """
        logging.info(f"Scoring {len(args)} nodes in parallel")

        if func is None:
            func = lambda x, y: bic_score_node(x, y)

        if parallel:
            scores = Parallel(n_jobs=4)(delayed(func)(**arg) for arg in tqdm(args))
        else:
            scores = [func(**arg) for arg in tqdm(args)]
        results = []
        for arg, result in zip(args, scores):
            d = {}
            d["node"] = arg["node"]
            d["parent_set"] = arg["parent_set"]
            d["score"] = result

            results.append(d)

        return results

    def _process_scoring_results(self, results: list[dict]):
        self.parent_graph = ParentGraph(self.nodes, results)

    def run_scoring(self, parallel=True, func=None):
        logging.info("Scoring Nodes")
        args = self._full_score_args()
        results = self._score(args, parallel=parallel, n_jobs=8, func=func)
        logging.info("Processing Results")
        self._process_scoring_results(results)

    def search(self, noise=0):
        assert self.parent_graph is not None, "You must score the nodes first"

        nodes = set(self.nodes)
        heuristic = self.parent_graph.heuristic(nodes, noise=noise)
        pq = PriorityQueue()
        closed = []
        best_scores = {}

        def set_best_score(node_set, score):
            best_scores[frozenset(node_set)] = score

        def get_best_score(node_set):
            return best_scores[frozenset(node_set)]

        def is_closed(node_set):
            for c in closed:
                if c == node_set:
                    return True

            return False

        parent_dict = {}
        for node in nodes:
            parent_dict[node] = set()

        item = PQItem(
            visited=set(), parent_dict=parent_dict, heuristic=heuristic, score=0
        )
        pq.push(item, heuristic)
        set_best_score(item.visited, item.score)
        v = []
        while not pq.empty():
            heuristic, item = pq.pop()
            # print(pq.entries)

            parent_dict = item.parent_dict
            v.append((item.visited, heuristic, parent_dict, item.score))

            visited = item.visited
            item_score = item.score

            if len(item.visited) == len(nodes):
                break

            if is_closed(visited):
                continue
            closed.append(visited)

            remaining = nodes - visited

            for node in sorted(list(remaining)):
                n = 1
                if noise > 0:
                    n = numpy.random.poisson(noise) + 1

                node_result = self.parent_graph.query(node, visited, n=n)
                # print(node, node_result, visited)

                if node_result is None:
                    parent_set = None
                    best_score = INF
                else:
                    parent_set = node_result["parent_set"]
                    best_score = node_result["score"]

                new_visited = visited.union({node})
                new_parent_dict = parent_dict.copy()
                new_parent_dict[node] = parent_set
                new_score = item_score + best_score

                # print(new_heuristic, new_score)
                if self.path_extension:
                    # old_visited = new_visited.copy()
                    # old_parent_dict = new_parent_dict.copy()
                    # old_score = new_score

                    new_visited, new_parent_dict, new_score = (
                        self.parent_graph.path_extension(
                            new_visited, new_parent_dict, new_score
                        )
                    )
                    # if new_score != old_score:
                    #     print("Using path extension for node: ", node)
                    #     print(old_visited, new_visited)
                    #     print(old_parent_dict, new_parent_dict)
                    #     print(old_score, new_score)
                    #     print("****")

                new_remaining = nodes - new_visited
                new_heuristic = self.parent_graph.heuristic(new_remaining)

                new_item = PQItem(
                    visited=new_visited,
                    parent_dict=new_parent_dict,
                    heuristic=new_heuristic,
                    score=new_score,
                )

                full_heuristic = new_score + new_heuristic

                if is_closed(new_visited):
                    current_score = get_best_score(new_visited)
                    if new_score < current_score:
                        # best_scores[new_visited] = new_score
                        set_best_score(new_visited, new_score)
                        closed.remove(new_visited)
                        pq.push(new_item, full_heuristic)

                else:
                    if pq.get(new_visited) is None:
                        pq.push(new_item, full_heuristic)
                        # best_scores[new_visited] = new_score
                        set_best_score(new_visited, new_score)
                    else:
                        if new_score < get_best_score(new_visited):
                            pq.delete(new_visited)
                            pq.push(new_item, full_heuristic)
                            # best_scores[new_visited] = new_score
                            set_best_score(new_visited, new_score)

        self.parent_dict = parent_dict
        self.adjacency_matrix = self.get_adjacency_matrix(parent_dict)
        self.visited = v
        return self.adjacency_matrix

    def get_adjacency_matrix(self, parent_dict):
        d = len(self.nodes)
        # print(parent_dict)
        res_df = pandas.DataFrame(
            numpy.zeros((d, d)), columns=self.nodes, index=self.nodes
        )

        for elm in parent_dict:
            for parent in parent_dict[elm]:
                res_df.loc[parent, elm] = 1

        res_df = res_df.astype(int)
        return res_df


def get_include_graph(columns):
    include_graph = pandas.DataFrame(
        numpy.zeros((len(columns), len(columns))), columns=columns, index=columns
    )
    include_graph.loc["X", "Y"] = 1
    return include_graph
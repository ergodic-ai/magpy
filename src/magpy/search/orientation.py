from itertools import combinations, permutations
import pandas
import numpy as np
import networkx as nx


def orient_v_structures(mat: pandas.DataFrame, sepsets):
    """
    Orient X *--* Z *--* Y as X *--> Z <--* if X and Y are disjoint and Z is not in their separation set
    :param mat: the adjacency matrix
    :param sepsets: Separating sets, as a dict of tuples of nodes as keys and sets of nodes as values
    :return: the oriented matrix
    """

    nodes = mat.index

    results = []

    mat = mat.copy()

    def get_sepset(node_x, node_y):
        return sepsets.get((node_x, node_y), sepsets.get((node_y, node_x), set()))

    # check each node if it can serve as a collider for a disjoint neighbors
    for node_z in nodes:
        # check neighbors
        xy_nodes = set(mat.index[mat.loc[:, node_z] == 1])  # potential drivers of Z

        for node_x, node_y in combinations(xy_nodes, 2):

            if mat.loc[node_x, node_y] == 1:
                continue  # skip this pair as they are connected
            if node_z not in get_sepset(node_x, node_y):
                mat.loc[node_x, node_z] = 1
                mat.loc[node_z, node_x] = 0
                mat.loc[node_y, node_z] = 1
                mat.loc[node_z, node_y] = 0

                d = {}
                d["from_1"] = node_x
                d["from_2"] = node_y
                d["to"] = node_z
                results.append(d)

    return mat, results


def rule_1(mat: pandas.DataFrame):
    """
    [R1] If A *--> B o--* C, and A & C are not connected, then orient A *--> B ---> C
    :return: True if the graph was modified; otherwise False
    """
    nodes = mat.index

    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    def is_connected(node_a, node_b):
        return mat.loc[node_a, node_b] == 1 or mat.loc[node_b, node_a] == 1

    for node_b in nodes:
        a_nodes = direct_parents_of(node_b)
        c_nodes = undirected_neighbors(node_b)

        for node_a in a_nodes:
            for node_c in c_nodes:
                if not is_connected(node_a, node_c):
                    mat.loc[node_b, node_c] = 1
                    mat.loc[node_c, node_b] = 0
                    d = {}
                    d["node_a"] = node_a
                    d["node_b"] = node_b
                    d["node_c"] = node_c
                    results.append(d)

    return mat, results


def rule_2(mat: pandas.DataFrame):
    """
    [R2] If (1) A *--> B ---> C or (2) A ---> B *--> C, and A *--o C, then orient A *--> C
    :return: True if the graph was modified; otherwise False
    """

    nodes = mat.index

    results = []
    mat = mat.copy()

    def potential_parents_of(node):
        return set(mat.index[mat.loc[:, node] == 1])

    def potential_children_of(node):
        return set(mat.index[mat.loc[node] == 1])

    def direct_parents_of(node):
        return potential_parents_of(node) - potential_children_of(node)

    def direct_children_of(node):
        return potential_children_of(node) - potential_parents_of(node)

    def undirected_neighbors(node):
        return potential_parents_of(node) & potential_children_of(node)

    def is_connected(node_a, node_b):
        return mat.loc[node_a, node_b] == 1 or mat.loc[node_b, node_a] == 1

    for node_b in nodes:
        a_nodes = direct_parents_of(node_b)
        c_nodes = direct_children_of(node_b)

        for node_c in c_nodes:
            for node_a in a_nodes:
                if node_c in potential_children_of(node_a):
                    mat.loc[node_a, node_c] = 1
                    mat.loc[node_c, node_a] = 0
                    d = {}
                    d["node_a"] = node_a
                    d["node_b"] = node_b
                    d["node_c"] = node_c
                    results.append(d)

    return mat, results


def is_dag(matrix):
    """Check if the given adjacency matrix has cycles"""
    # first I will remove all undirected edges from the matrix
    matrix = matrix.copy()

    G = nx.DiGraph((matrix * (1 - matrix.T)))
    return nx.is_directed_acyclic_graph(G)


def generate_orientations(mat: pandas.DataFrame, unoriented_edges: list):
    if not unoriented_edges:
        yield mat
        return

    edge = unoriented_edges.pop()
    u, v = edge

    # Two possible orientations
    matrix1 = mat.copy()
    matrix1.loc[u, v] = 0

    matrix2 = mat.copy()
    matrix2.loc[v, u] = 0

    # Check if adding the edge creates a DAG
    if is_dag(matrix1):
        yield from generate_orientations(matrix1, unoriented_edges.copy())

    if is_dag(matrix2):
        yield from generate_orientations(matrix2, unoriented_edges.copy())


def get_unoriented_edges(mat: pandas.DataFrame):
    """gets the edges for which we have a forward and backward path"""

    edges = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[1]):
            iname, jname = mat.index[i], mat.index[j]
            if mat.loc[iname, jname] == 1 and mat.loc[jname, iname] == 1:
                edges.append((iname, jname))
    return edges


def all_dags(init_matrix: pandas.DataFrame):
    """Main function to generate all DAGs from symmetric adjacency matrix"""
    edges = get_unoriented_edges(init_matrix)
    init_matrix = init_matrix.copy()

    dags = list(generate_orientations(init_matrix, edges))
    return dags


def generate_one_dag(init_matrix: pandas.DataFrame):
    """Main function to generate all DAGs from symmetric adjacency matrix"""
    edges = get_unoriented_edges(init_matrix)
    init_matrix = init_matrix.copy()

    try:
        return next(generate_orientations(init_matrix, edges))
    except:
        return None

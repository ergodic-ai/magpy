from itertools import combinations, chain
import pandas
import numpy
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def execute_parallel(func, args):
    """
    Execute a function in parallel
    :param func: a function to execute
    :param args: a list of arguments to pass to the function
    :return: a list of results
    """

    results = []

    r = Parallel(n_jobs=512)(
        delayed(func)(arg["node_i"], arg["node_j"], arg["cond_set"])
        for arg in tqdm(args)
    )

    for i, arg in enumerate(args):
        arg["independent"] = r[i]
        results.append(arg)

    return results


def unique_element_iterator(chained_iterators):
    """
    return the unique instances of the chained iterators
    :param an iterator with possibly repeating elements, e.g., chained_iterators: chain(iter_a, iter_b)
    :return: an iterator with unique (unordered) elements
    """
    seen = set()
    for e in chained_iterators:
        if e in seen:
            continue

        seen.add(e)
        yield e


def pc_skeleton(
    ci_test,
    nodes,
    max_sepset_size=-1,
    existing=[],
    forbidden=[],
    mat=None,
    intersection_or_union="union",
    skip_sepsets=[],
    verbose=False,
):
    """
    Find the skeleton of a DAG using the PC algorithm.
    :param ci_test: a conditional independence test
    :param nodes: a list of nodes in the graph
    :param existing: a list of edges that are known to be in the graph
    :param forbidden: a list of edges that are known to not be in the graph
    :param verbose: if True, print the graph after each step
    :return: a networkx graph representing the skeleton of the DAG
    """

    sepsets = dict()

    # initialize the dataframe with ones
    if mat is None:
        mat = pandas.DataFrame(
            numpy.ones((len(nodes), len(nodes))), index=nodes, columns=nodes
        )

    # remove the ones that are forbidden
    for edge in forbidden:
        mat.loc[edge[0], edge[1]] = 0
        mat.loc[edge[1], edge[0]] = 0

    for node in nodes:
        mat.loc[node, node] = 0

    def exit_cond(cond_set_size):
        if cond_set_size > len(nodes) - 2:
            return True

        if max_sepset_size > 0 and cond_set_size > max_sepset_size:
            return True

        # if there are no nodes with more parents than the cond_set_size, we can stop
        for node in nodes:
            if mat[node].sum() > cond_set_size:
                return False

        return True

    cond_set_size = 0
    while not exit_cond(cond_set_size):
        if verbose:
            print(f"Conditioning set of size: {cond_set_size}")

        if cond_set_size in skip_sepsets:
            cond_set_size += 1
            continue

        for node_i, node_j in tqdm(combinations(nodes, 2)):
            if mat.loc[node_i, node_j] == 0:
                continue

            for edge in existing:
                if node_i in edge and node_j in edge:
                    break

            pot_parents_i = set(mat.index[mat.loc[node_i] == 1]) - {node_j}
            pot_parents_j = set(mat.index[mat.loc[node_j] == 1]) - {node_i}

            if intersection_or_union == "intersection":
                pot_parents_ij = pot_parents_i.intersection(pot_parents_j)
                cond_set_ij = combinations(pot_parents_ij, cond_set_size)
                cond_sets = unique_element_iterator(cond_set_ij)

            else:
                cond_sets_i = combinations(pot_parents_i, cond_set_size)
                cond_sets_j = combinations(pot_parents_j, cond_set_size)

                cond_sets = unique_element_iterator(chain(cond_sets_i, cond_sets_j))

            for cond_set in cond_sets:
                if ci_test(node_i, node_j, cond_set):
                    mat.loc[node_i, node_j] = 0
                    mat.loc[node_j, node_i] = 0
                    sepsets[(node_i, node_j)] = cond_set

                    if verbose:
                        print(
                            f"Removed edge {node_i} -- {node_j} based on the conditioning set: {cond_set} // { pot_parents_i } // { pot_parents_j }"
                        )
                    break

        cond_set_size += 1

    return mat, sepsets


def parallel_pc_skeleton(
    ci_test,
    nodes,
    max_sepset_size=-1,
    existing=[],
    forbidden=[],
    mat=None,
    intersection_or_union="union",
    verbose=False,
):
    """
    Find the skeleton of a DAG using the PC algorithm.
    :param ci_test: a conditional independence test
    :param nodes: a list of nodes in the graph
    :param existing: a list of edges that are known to be in the graph
    :param forbidden: a list of edges that are known to not be in the graph
    :param verbose: if True, print the graph after each step
    :return: a networkx graph representing the skeleton of the DAG
    """

    sepsets = dict()

    # initialize the dataframe with ones
    if mat is None:
        mat = pandas.DataFrame(
            numpy.ones((len(nodes), len(nodes))), index=nodes, columns=nodes
        )

    # remove the ones that are forbidden
    for edge in forbidden:
        mat.loc[edge[0], edge[1]] = 0
        mat.loc[edge[1], edge[0]] = 0

    for node in nodes:
        mat.loc[node, node] = 0

    def exit_cond(cond_set_size):
        if cond_set_size > len(nodes) - 2:
            return True

        if max_sepset_size > 0 and cond_set_size > max_sepset_size:
            return True

        # if there are no nodes with more parents than the cond_set_size, we can stop
        for node in nodes:
            if mat[node].sum() > cond_set_size:
                return False

        return True

    cond_set_size = 0
    while not exit_cond(cond_set_size):
        # print(cond_set_size)
        args = []
        for node_i, node_j in combinations(nodes, 2):
            if mat.loc[node_i, node_j] == 0:
                continue

            for edge in existing:
                if node_i in edge and node_j in edge:
                    break

            pot_parents_i = set(mat.index[mat.loc[node_i] == 1]) - {node_j}
            pot_parents_j = set(mat.index[mat.loc[node_j] == 1]) - {node_i}

            if intersection_or_union == "intersection":
                pot_parents_ij = pot_parents_i.intersection(pot_parents_j)
                cond_set_ij = combinations(pot_parents_ij, cond_set_size)
                cond_sets = unique_element_iterator(cond_set_ij)

            else:
                cond_sets_i = combinations(pot_parents_i, cond_set_size)
                cond_sets_j = combinations(pot_parents_j, cond_set_size)

                cond_sets = unique_element_iterator(chain(cond_sets_i, cond_sets_j))

            for cond_set in cond_sets:
                arg = {}
                arg["node_i"] = node_i
                arg["node_j"] = node_j
                arg["cond_set"] = cond_set
                args.append(arg)

        results = execute_parallel(ci_test, args)

        for result in results:
            if verbose:
                print(
                    f"Testing edge {node_i} -- {node_j} based on the conditioning set: {cond_set} // { pot_parents_i } // { pot_parents_j }"
                )

            if result["independent"]:
                node_i = result["node_i"]
                node_j = result["node_j"]
                cond_set = result["cond_set"]

                mat.loc[node_i, node_j] = 0
                mat.loc[node_j, node_i] = 0

                prev_cond_set = sepsets.get((node_i, node_j), [])
                sepsets[(node_i, node_j)] = list(
                    set(prev_cond_set).union(set(cond_set))
                )

        cond_set_size += 1

    return mat, sepsets
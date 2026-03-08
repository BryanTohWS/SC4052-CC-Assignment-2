import time

import networkx as nx
import numpy as np


def build_transition_matrix(G, node_index):
    """
    Build the transition matrix A from a NetworkX directed graph.

    A[i][j] = 1/num_outlinks(j) if page j has a link to page i, else 0.

    This represents the probability of moving from page j to page i. Each column sums to 1 (column-stochastic).
    Dangling nodes (pages with no outlinks) are handled by assigning their rank equally to all pages.
    """
    n = len(node_index)

    # Creating matrix of all 0s. This is the starting point before we fill in the transition probabilities
    A = np.zeros((n, n))

    for node in G.nodes():
        j = node_index[node]
        num_outlinks = dict(G.out_degree())[node]

        # Dangling node: if a page has no outlinks, it should distribute its rank equally to all pages
        if num_outlinks == 0:
            A[:, j] = 1.0 / n
        else:
            # Distribute rank equally among outgoing links
            for nb in G.successors(node):
                A[node_index[nb]][j] = 1.0 / num_outlinks

    return A

def iterative_method(A, p=0.15, max_iter=100, tol=1e-6):
    """
    Compute PageRank using the Iterative Method

    Stops when the L1 difference between iterations is below tolerance threshold tol -> meaning ranks have stopped changing significantly.
    """
    n = A.shape[0]  # A.shape[0] gives the number of rows (= number of pages)

    # Initialising all ranks to 1/n (uniform distribution)
    R = np.ones(n) / n

    for i in range(max_iter):
        # Compute new ranks based on the PageRank formula:
        #  R' = (1-p) * A * R + (p/n) * 1
        R_new = (1 - p) * A.dot(R) + (p / n) * np.ones(n)

        # This measures the total change in rank across all pages between iterations
        # When this value drops below tolerance threshold tol, ranks have converged (stopped changing meaningfully)
        diff = np.linalg.norm(R_new - R, 1)
        R = R_new
        if diff < tol:
            print(f"Converged in {i + 1} iterations")
            return R, i + 1

    print(f"Reached max iterations ({max_iter})")
    return R, max_iter


def closed_form(A, p=0.15):
    """
    Compute PageRank analytically using the Closed Form method
    """
    n = A.shape[0]  # A.shape[0] gives the number of rows (= number of pages)

    # Applying the formula: R' = (p/n) * (I - (1-p)*A)^(-1) * 1
    R = (p / n) * np.linalg.inv(np.eye(n) - (1 - p) * A).dot(np.ones(n))

    # R.sum() sums all entries of the vector and dividing normalises scores to sum to 1
    return R / R.sum()


def small_example():
    """
    Part 1: Run both methods on a 3-node graph and compare results.

    Graph structure: A -> B, B -> A, C -> A, C -> B
    """
    print("-" * 60)
    print("PART 1: SMALL 3-NODE EXAMPLE")
    print("-" * 60)

    # Build a small 3-node directed graph using NetworkX
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "A"), ("C", "A"), ("C", "B")])

    # Map each node to a matrix index (A=0, B=1, C=2)
    node_index = {node: i for i, node in enumerate(sorted(G.nodes()))}

    # Build transition matrix from the graph
    A = build_transition_matrix(G, node_index)

    print("\nTransition Matrix A:")
    print(A)

    print(f"\nEffect of p (Closed Form):\n\n{'p':<6} {'R(A)':<12} {'R(B)':<12} {'R(C)':<12}")
    print("-" * 42)
    for p in [0.10, 0.50, 1.00]:
        R = closed_form(A, p=p)
        print(f"{p:<6} {R[0]:<12.3f} {R[1]:<12.3f} {R[2]:<12.3f}")

    print("\nComparison at p = 0.30:")
    R_iter, _ = iterative_method(A, p=0.30)
    R_cf = closed_form(A, p=0.30)

    print(f"\n{'Page':<8} {'Iterative Method':<22} {'Closed Form':<22} {'Difference':<12}")
    print("-" * 64)
    for i, page in enumerate(["A", "B", "C"]):
        # Calculates the difference between the iterative and closed form results for each page
        diff = abs(R_iter[i] - R_cf[i])
        if diff == 0:
            diff_str = "0"
        else:
            diff_str = f"10^{int(np.floor(np.log10(diff)))}"
        print(f"{page:<8} {R_iter[i]:<22.5f} {R_cf[i]:<22.5f} {diff_str}")


def full_dataset(filepath, p=0.15):
    """
    Part 2: Run both methods on web-Google_10k.txt and compare
    """
    print("\n" + "-" * 60)
    print("PART 2: USING web-Google_10k.txt DATASET")
    print("-" * 60)

    # Load the web graph from the dataset file using NetworkX
    # Each line in the file is an edge: "FromNodeId  ToNodeId"
    G = nx.DiGraph()
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))

    # Map each node ID to a matrix index
    node_index = {node: i for i, node in enumerate(sorted(G.nodes()))}
    print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build the transition matrix from the loaded graph
    A = build_transition_matrix(G, node_index)

    # Run iterative method and record time
    start = time.time()
    R_iter, _ = iterative_method(A, p=p)
    iter_time = time.time() - start

    # Run closed form and record time
    start = time.time()
    R_cf = closed_form(A, p=p)
    cf_time = time.time() - start

    # Finds the largest difference in PageRank scores between the two methods across all pages
    max_diff = np.max(np.abs(R_iter - R_cf))
    max_exp = int(np.floor(np.log10(max_diff))) if max_diff > 0 else 0

    print(f"\n{'Method':<25} {'Top PageRank Score':<22} {'Time (s)':<10}")
    print("-" * 57)

    # np.max() finds the highest PageRank score across all pages
    print(f"{'Iterative Method':<25} {np.max(R_iter):<22.5f} {iter_time:<10.2f}")
    print(f"{'Closed Form':<25} {np.max(R_cf):<22.5f} {cf_time:<10.2f}")
    print(f"Max difference: 10^{max_exp}")

    return R_iter, A, node_index

if __name__ == "__main__":
    DATASET = "web-Google_10k.txt"
    small_example()
    R, A, node_index = full_dataset(DATASET, p=0.15)
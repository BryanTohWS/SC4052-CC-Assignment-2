import numpy as np

# Small directed web graph represented as a dictionary.
# Each key is a URL, and its value is a list of URLs it links to (outlinks).
web_graph = {
    "openai.com": ["huggingface.co", "deepmind.com", "wikipedia.org"],
    "deepmind.com": ["openai.com", "nature.com", "wikipedia.org"],
    "huggingface.co": ["openai.com", "github.com", "wikipedia.org"],
    "wikipedia.org": ["nature.com", "github.com"],
    "nature.com": ["openai.com", "deepmind.com"],
    "github.com": ["huggingface.co", "openai.com"],
    "techcrunch.com": ["openai.com", "deepmind.com", "huggingface.co"],
    "blogspam.com": ["openai.com", "deepmind.com", "huggingface.co",
                    "wikipedia.org", "nature.com", "github.com",
                    "techcrunch.com"],
    "linkfarm.net": ["openai.com", "deepmind.com", "huggingface.co",
                    "wikipedia.org", "nature.com", "github.com",
                    "techcrunch.com", "blogspam.com"],
    "reddit.com/r/ml": ["openai.com", "huggingface.co", "github.com"],
}

# Precomputed PageRank scores (provided as input to the program)
precomputed_pagerank = {
    "openai.com": 0.220,
    "deepmind.com": 0.180,
    "huggingface.co": 0.160,
    "wikipedia.org": 0.140,
    "nature.com": 0.100,
    "github.com": 0.080,
    "techcrunch.com": 0.055,
    "blogspam.com": 0.035,
    "linkfarm.net": 0.020,
    "reddit.com/r/ml": 0.010,
}

# AI seed pages - known high-quality AI research pages used to bias teleportation
ai_seeds = ["openai.com", "deepmind.com", "huggingface.co"]

def build_transition_matrix(graph, url_index):
    """
    Build the transition matrix A from the web graph dictionary.

    A[i][j] = 1/num_outlinks(j) if page j links to page i, else 0.

    Dangling nodes (no outlinks) spread rank equally to all pages to prevent rank from being lost at dead ends.
    """
    n = len(url_index)
    A = np.zeros((n, n))

    for url, outlinks in graph.items():
        j = url_index[url]
        num_outlinks = len(outlinks)

        if num_outlinks == 0:
            # Dangling node - spread rank equally to all pages
            A[:, j] = 1.0 / n
        else:
            # Distribute rank equally among outgoing links
            for link in outlinks:
                if link in url_index:
                    A[url_index[link]][j] = 1.0 / num_outlinks

    return A

def standard_pagerank(A, p=0.15, max_iter=100, tol=1e-6):
    """
    Standard PageRank with uniform teleportation E(u) = 1/n.
    Used as the baseline for comparison against the heuristic.
    """
    n = A.shape[0]
    R = np.ones(n) / n

    for i in range(max_iter):
        # Apply PageRank formula: follow links with prob (1-p), teleport uniformly to any page with prob p/n
        R_new = (1 - p) * A.dot(R) + (p / n) * np.ones(n)
        if np.linalg.norm(R_new - R, 1) < tol:
            print(f"Standard PageRank converged in {i + 1} iterations")
            return R_new
        R = R_new

    return R

def personalised_pagerank(A, seed_indices, p=0.15, max_iter=100, tol=1e-6):
    """
    PageRank with teleportation biased towards AI seed pages.
    Teleportation vector E(u) = 1/|S| if u in S (seed set), else 0.
    """
    n = A.shape[0]
    R = np.ones(n) / n

    # Build biased teleportation vector - only teleport to seed pages
    E = np.zeros(n)
    for index in seed_indices:
        E[index] = 1.0 / len(seed_indices)

    for i in range(max_iter):
        # Apply PageRank formula: follow links with prob (1-p), teleport only to seed pages with prob p
        R_new = (1 - p) * A.dot(R) + p * E
        if np.linalg.norm(R_new - R, 1) < tol:
            print(f"Personalised PageRank converged in {i + 1} iterations")
            return R_new
        R = R_new

    return R

def authority_score(graph, url_index):
    """
    Authority score using graph structure (inlinks vs outlinks).
    AuthorityScore(u) = inlinks(u) / (inlinks(u) + outlinks(u) + 1)
    """
    n = len(url_index)

    # Count inlinks and outlinks from the graph dictionary
    inlinks = {}
    for url in graph:
        inlinks[url] = 0

    outlinks = {}
    for url, links in graph.items():
        outlinks[url] = len(links)

    for url, links in graph.items():
        for link in links:
            if link in inlinks:
                inlinks[link] += 1

    # Compute authority score for each page
    auth = np.zeros(n)
    for url in graph:
        i = url_index[url]
        auth[i] = inlinks[url] / (inlinks[url] + outlinks[url] + 1)

    return auth

def crawl_priority(graph, pagerank_scores, seeds, k=5, p=0.15):
    """
    Takes a the web graph (dictionary of URLs and outlinks) and precomputed PageRank scores, then returns the top k URLs to crawl.

    This compares between the baseline approach of using precomputed PageRank scores to rank pages, 
    and a heuristic approach that combines Personalised PageRank (biased towards AI seed pages) with an authority score based on the graph structure.
    """
    print("-" * 60)
    print("AI WEB CRAWLER FOR TRAINING - PRIORITY RANKING OF PAGES TO CRAWL")
    print("-" * 60)

    # Map each URL to a matrix index
    url_index = {}
    for i, url in enumerate(sorted(graph.keys())):
        url_index[url] = i

    index_url = {}
    for url, i in url_index.items():
        index_url[i] = url
    n = len(url_index)

    total_links = 0
    for v in graph.values():
        total_links += len(v)
    print(f"\nWeb graph: {n} pages, {total_links} links")
    print(f"AI seed pages: {seeds}")

    # Build transition matrix from the web graph dictionary
    A = build_transition_matrix(graph, url_index)

    # Baseline: rank pages by precomputed PageRank scores
    baseline_list = []
    for url in sorted(graph.keys()):
        baseline_list.append(pagerank_scores[url])
    baseline = np.array(baseline_list)
    top_baseline = np.argsort(baseline)[::-1][:k]

    print(f"\nTop {k} URLs - Baseline (Precomputed PageRank):")
    print(f"{'Rank':<6} {'URL':<25} {'PageRank Score':<15}")
    print("-" * 46)
    for rank, index in enumerate(top_baseline, 1):
        print(f"{rank:<6} {index_url[index]:<25} {baseline[index]:<15.3f}")

    # Heuristic: PPR * AuthorityScore
    seed_indices = []
    for s in seeds:
        if s in url_index:
            seed_indices.append(url_index[s])

    # Run Personalised PageRank biased towards AI seed pages
    ppr = personalised_pagerank(A, seed_indices, p=p)

    # Compute authority score for each page
    auth = authority_score(graph, url_index)

    # Combined heuristic score
    heuristic = ppr * auth
    top_heuristic = np.argsort(heuristic)[::-1][:k]

    print(f"\nAll URLs - PPR, Authority Score and Overall Score:")
    print(f"{'URL':<25} {'PPR':<10} {'Authority Score':<18} {'Overall Score':<13}")
    print("-" * 66)
    for index in np.argsort(heuristic)[::-1]:
        print(f"{index_url[index]:<25} {ppr[index]:<10.3f} {auth[index]:<18.3f} {heuristic[index]:<13.4f}")

    print(f"\nTop {k} URLs - Heuristic (PPR x Authority Score):")
    print(f"{'Rank':<6} {'URL':<25} {'PPR':<10} {'Authority Score':<18} {'Overall Score':<13}")
    print("-" * 72)
    for rank, index in enumerate(top_heuristic, 1):
        print(f"{rank:<6} {index_url[index]:<25} {ppr[index]:<10.3f} {auth[index]:<18.3f} {heuristic[index]:<13.4f}")

    # Overlap analysis
    overlap = len(set(top_baseline) & set(top_heuristic))
    print(f"\nOverlap between baseline and heuristic top-{k}: {overlap}/{k} pages")
    print(f"Pages uniquely identified by heuristic: {k - overlap}")

if __name__ == "__main__":
    crawl_priority(
        graph=web_graph,
        pagerank_scores=precomputed_pagerank,
        seeds=ai_seeds,
        k=5,
        p=0.15
    )

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import networkx as nx
import networkx.algorithms.community


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags required by the assignment."""
    p = argparse.ArgumentParser(
        prog="graph_analysis.py",
        description="Analyze and visualize .gml graphs (metrics, communities, robustness, verification).",
    )

    p.add_argument("graph_file", help="Input graph (.gml).")

    p.add_argument("--components", type=int, default=None,
                   help="Partition graph into n communities using Girvan–Newman.")

    p.add_argument("--plot", choices=["C", "N", "P"], default=None,
                   help="Plot mode: C=clustering coeff, N=neighborhood overlap, P=plot attributes.")

    p.add_argument("--verify_homophily", action="store_true",
                   help="Run homophily t-test using a node attribute (default: 'color').")

    p.add_argument("--verify_balanced_graph", action="store_true",
                   help="Verify structural balance for signed graphs (edge attr default: 'sign').")

    p.add_argument("--output", default=None,
                   help="Write final graph (with computed attributes) to output .gml.")

    p.add_argument("--simulate_failures", type=int, default=None,
                   help="Randomly remove k edges and analyze impact.")

    p.add_argument("--robustness_check", type=int, default=None,
                   help="Run repeated edge-failure simulations for robustness statistics (k edges per trial).")

    p.add_argument("--temporal_simulation", default=None,
                   help="CSV file with edge updates: source,target,timestamp,action(add/remove).")

    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--verbose", action="store_true", help="Print extra output.")
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on obvious issues."""
    if not args.graph_file.lower().endswith(".gml"):
        raise ValueError("graph_file must be a .gml file")

    if args.components is not None and args.components <= 0:
        raise ValueError("--components must be a positive integer")

    if args.simulate_failures is not None and args.simulate_failures < 0:
        raise ValueError("--simulate_failures must be >= 0")

    if args.robustness_check is not None and args.robustness_check < 0:
        raise ValueError("--robustness_check must be >= 0")

    if args.output is not None and not args.output.lower().endswith(".gml"):
        raise ValueError("--output must end with .gml")

# -----------------------------
# I/O
# -----------------------------
def load_graph(path: str) -> nx.Graph:
    """Load a .gml file into a NetworkX graph and validate basics."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        G = nx.read_gml(path)
    except Exception as e:
        raise ValueError(f"Failed to read .gml file: {e}") from e

    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty (0 nodes).")

    # Many algorithms assume an undirected graph; keep it simple:
    # If your dataset is directed and you must preserve direction, remove this.
    if G.is_directed():
        G = G.to_undirected()

    return G


def export_graph(G: nx.Graph, out_path: str) -> None:
    """Write the graph (including computed attributes) to .gml."""
    nx.write_gml(G, out_path)

# -----------------------------
# Metrics
# -----------------------------
def compute_clustering(G: nx.Graph) -> Dict[Any, float]:
    """Node -> clustering coefficient."""
    return nx.clustering(G)


def compute_neighborhood_overlap(G: nx.Graph) -> Dict[Tuple[Any, Any], float]:
    """
    Edge (u,v) -> neighborhood overlap:
      |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    """
    overlap: Dict[Tuple[Any, Any], float] = {}
    for u, v in G.edges():
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        union = Nu | Nv
        inter = Nu & Nv
        overlap[(u, v)] = (len(inter) / len(union)) if union else 0.0
    return overlap


def compute_metrics(G: nx.Graph) -> None:
    """Compute required metrics and attach as attributes."""
    cc = compute_clustering(G)
    nx.set_node_attributes(G, cc, "clustering")

    no = compute_neighborhood_overlap(G)
    for (u, v), val in no.items():
        if G.has_edge(u, v):
            G.edges[u, v]["neighborhood_overlap"] = float(val)

# -----------------------------
# Community detection (Girvan–Newman)
# -----------------------------
def partition_communities(G: nx.Graph, n: int) -> List[set]:
    """
    Partition the graph into n communities using Girvan–Newman.
    Attach node attribute 'community' (0..n-1).
    """
    if n == 1:
        communities = [set(G.nodes())]
    else:
        iterator = networkx.algorithms.community.girvan_newman(G)
        communities = []
        for com in iterator:
            communities = list(com)
            if len(communities) >= n:
                break
    for com, nodes in enumerate(communities):
        for n in nodes:
            G.nodes[n]["community"] = com
    return communities


# -----------------------------
# Failure simulation + robustness
# -----------------------------
def avg_shortest_path_length_safe(G: nx.Graph) -> Optional[float]:
    """
    Average shortest path length:
    - If connected: compute on whole graph
    - If disconnected: compute on the largest connected component
    """
    if G.number_of_nodes() == 0:
        return None
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        if subgraph.number_of_nodes() > 1:
            return nx.average_shortest_path_length(subgraph)
        return 0.0

def simulate_failures(G: nx.Graph, k: int, seed: int) -> Dict[str, object]:
    """Remove k random edges on a copy of G and report impact."""
    random.seed(seed)

    H = G.copy()

    baseline_components = nx.number_connected_components(H)
    baseline_path_length = avg_shortest_path_length_safe(H)
    baseline_betweenness = nx.edge_betweenness_centrality(H)

    edges = list(H.edges())
    k_actual = min(k, len(edges))
    removed_edges = random.sample(edges, k_actual)
    H.remove_edges_from(removed_edges)

    new_components = nx.number_connected_components(H)
    new_path_length = avg_shortest_path_length_safe(H)
    new_betweenness = nx.edge_betweenness_centrality(H)

    if baseline_path_length is not None and new_path_length is not None:
        path_length_change = new_path_length - baseline_path_length
    else:
        path_length_change = None

    betweenness_diffs = []
    for node in H.nodes():
        old_val = baseline_betweenness.get(node, 0)
        new_val = new_betweenness.get(node, 0)
        betweenness_diffs.append(abs(old_val - new_val))
    avg_betweenness_impact = sum(betweenness_diffs) / len(betweenness_diffs) if betweenness_diffs else 0.0

    return {
        "edges_removed": k_actual,
        "components_before": baseline_components,
        "components_after": new_components,
        "disconnected_components_created": new_components - baseline_components,
        "path_length_change": path_length_change,
        "avg_betweenness_impact": avg_betweenness_impact

    }


def robustness_check(G: nx.Graph, k: int, trials: int = 20, seed: int = 42) -> Dict[str, object]:
    """Repeat edge-failure simulation across trials and aggregate results."""
    random.seed(seed)

    trial_components_count = []
    trial_max_sizes = []
    trial_min_size = []

    for t in range(trials):
        H = G.copy()
        edges = list(H.edges())
        k_actual = min(k, len(edges))
        removed_edges = random.sample(edges, k_actual)
        H.remove_edges_from(removed_edges)

        components = list(nx.connected_components(H))
        trial_components_count.append(len(components))
        if components:
            sizes = [len(c) for c in components]
            trial_max_sizes.append(max(sizes))
            trial_min_size.append(min(sizes))
        else:
            trial_max_sizes.append(0)
            trial_min_size.append(0)

    avg_components = sum(trial_components_count) / trials
    avg_max_size = sum(trial_max_sizes) / trials
    avg_min_size = sum(trial_min_size) / trials

    original_components = nx.number_connected_components(G)
    clusters_persisted = all(count == original_components for count in trial_components_count)
    return {
        "trials": trials,
        "k_edges_removed": k,
        "avg_connected_components": avg_components,
        "avg_max_component_size": avg_max_size,
        "avg_min_component_size": avg_min_size,
        "original_clusters_persisted": clusters_persisted
    }
# Verification: Homophily + Structural balance
# -----------------------------
def verify_homophily(G: nx.Graph, attr: str = "color", samples: int = 2000, seed: int = 42) -> Dict[str, object]:
    """
    Homophily via t-test:
    - similarity = 1 if node attrs match, else 0
    - compare connected-pair similarities vs random-pair similarities
    """
    import scipy.stats as stats
    random.seed(seed)
    edge_similarities = []
    for u, v in G.edges():
        u_attr = G.nodes[u].get(attr)
        v_attr = G.nodes[v].get(attr)
        if u_attr is None or v_attr is None:
            continue
        sim = 1 if u_attr == v_attr else 0
        edge_similarities.append(sim)
    random_similarities = []
    nodes = list(G.nodes())
    for _ in range(samples):
        u, v = random.sample(nodes, 2)
        u_attr = G.nodes[u].get(attr)
        v_attr = G.nodes[v].get(attr)
        if u_attr is None or v_attr is None:
            continue
        sim = 1 if u_attr == v_attr else 0
        random_similarities.append(sim)
    t_stat, p_val = stats.ttest_ind(edge_similarities, random_similarities, equal_var=False)
    return {
        "attribute_tested": attr,
        "actual_edge_similarity_avg": sum(edge_similarities) / len(edge_similarities) if edge_similarities else 0.0,
        "random_pair_similarity_avg": sum(random_similarities) / len(random_similarities) if random_similarities else 0.0,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "homophily_detected": bool(p_val < 0.05 and t_stat > 0)
    }

def verify_balanced_graph(G: nx.Graph, sign_attr: str = "sign") -> Dict[str, object]:
    """
    Structural balance check for signed graphs (BFS-based).
    Expected sign: +1/-1 (or strings like 'positive'/'negative').
    Rule:
      + edge => same group
      - edge => different group
    """
    group_assignment = {node: None for node in G.nodes()}
    is_balanced = True
    violating_edges = []
    for start_node in G.nodes():
        if group_assignment[start_node] is not None:
            continue
        group_assignment[start_node] = 1
        queue = [start_node]

        while queue:
            u = queue.pop(0)
            for v in G.neighbors(u):
                # Get the sign of the edge (defaulting to positive if missing)
                edge_data = G.get_edge_data(u, v)
                sign_val = edge_data.get(sign_attr, "+")
                if sign_val in ["+", "positive", 1]:
                    target_group = group_assignment[u]
                else:
                    target_group = -group_assignment[u]
                if group_assignment[v] is None:
                    group_assignment[v] = target_group
                    queue.append(v)
                elif group_assignment[v] != target_group:
                    is_balanced = False
                    violating_edges.append((u, v))
    return {
        "is_balanced": is_balanced,
        "num_violating_edges": len(violating_edges),
        "violating_edges": violating_edges[:10]  # Limit output for readability
    }
    # -----------------------------
# Temporal simulation
# -----------------------------
def temporal_simulation(G: nx.Graph, csv_path: str) -> Dict[str, object]:
    """
    Apply time-ordered edge updates from CSV.
    CSV columns: source,target,timestamp,action(add/remove)
    Updates graph in-place.
    """
    if not os.path.exists(csv_path):
        return {"ok": False, "reason": f"CSV file not found: {csv_path}"}

    events: List[Tuple[str, str, str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"source", "target", "timestamp", "action"}
        if not required.issubset(set(reader.fieldnames or [])):
            return {"ok": False, "reason": f"CSV must have columns: {sorted(required)}"}

        for row in reader:
            ts = int(row["timestamp"])
            action = row["action"].strip().lower()
            u = row["source"]
            v = row["target"]
            events.append((ts, action, u, v))

    events.sort(key=lambda x: x[0])

    added = 0
    removed = 0
    for _, action, u, v in events:
        if action == "add":
            G.add_edge(u, v)
            added += 1
        elif action == "remove":
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                removed += 1

    return {"ok": True, "events": len(events), "edges_added": added, "edges_removed": removed}


# -----------------------------
# Plotting
# -----------------------------
def plot_graph(G: nx.Graph, mode: str) -> None:
    """
    Plot required modes:
      C: node size=clustering, node color=degree
      N: edge width=neighborhood overlap, edge color=sum of endpoint degrees
      P: plot attributes (community if present, else degree)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for --plot (pip install -r requirements.txt).") from e

    pos = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())

    if mode == "C":
        clustering = nx.get_node_attributes(G, "clustering") or compute_clustering(G)
        node_sizes = [300 + 2000 * float(clustering.get(n, 0.0)) for n in G.nodes()]
        node_colors = [degrees.get(n, 0) for n in G.nodes()]

        nx.draw_networkx(G, pos=pos, with_labels=False, node_size=node_sizes, node_color=node_colors, width=0.8)
        plt.title("C: size=clustering, color=degree")
        plt.show()
        return

    if mode == "N":
        widths = []
        edge_colors = []
        for u, v in G.edges():
            ov = float(G.edges[u, v].get("neighborhood_overlap", 0.0))
            widths.append(0.5 + 6.0 * ov)
            edge_colors.append(degrees.get(u, 0) + degrees.get(v, 0))

        nx.draw_networkx_nodes(G, pos=pos, node_size=250)
        nx.draw_networkx_edges(G, pos=pos, width=widths, edge_color=edge_colors)
        plt.title("N: width=neighborhood overlap, color=sum endpoint degrees")
        plt.show()
        return

    if mode == "P":
        comm = nx.get_node_attributes(G, "community")
        node_colors = [comm.get(n, degrees.get(n, 0)) for n in G.nodes()]

        nx.draw_networkx(G, pos=pos, with_labels=False, node_size=250, node_color=node_colors, width=0.8)
        plt.title("P: attributes (community if present, else degree)")
        plt.show()
        return

    raise ValueError("Invalid plot mode (expected C, N, or P).")

# -----------------------------
# Main driver
# -----------------------------
def main(argv: List[str]) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        validate_args(args)

        random.seed(args.seed)

        G = load_graph(args.graph_file)
        if args.verbose:
            print(f"[INFO] Loaded graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

        # Always compute metrics so plot/export has the attributes.
        compute_metrics(G)

        if args.components is not None:
            comms = partition_communities(G, args.components)
            if args.verbose:
                print(f"[INFO] Communities computed: {len(comms)}")

        if args.simulate_failures is not None:
            rep = simulate_failures(G, args.simulate_failures, seed=args.seed)
            print("[SIMULATE_FAILURES]", rep)

        if args.robustness_check is not None:
            rep = robustness_check(G, args.robustness_check, trials=20, seed=args.seed)
            print("[ROBUSTNESS_CHECK]", rep)

        if args.verify_homophily:
            rep = verify_homophily(G, attr="color", seed=args.seed)
            print("[VERIFY_HOMOPHILY]", rep)

        if args.verify_balanced_graph:
            rep = verify_balanced_graph(G, sign_attr="sign")
            print("[VERIFY_BALANCED_GRAPH]", rep)

        if args.temporal_simulation:
            rep = temporal_simulation(G, args.temporal_simulation)
            print("[TEMPORAL_SIMULATION]", rep)

        if args.plot:
            plot_graph(G, args.plot)

        if args.output:
            export_graph(G, args.output)
            if args.verbose:
                print(f"[INFO] Exported graph: {args.output}")

        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
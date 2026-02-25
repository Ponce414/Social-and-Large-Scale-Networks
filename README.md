# Social and Large-Scale Network Analysis

Command-line Python tool for analyzing and visualizing `.gml` graph data.

This project computes structural metrics, performs community detection, simulates edge failures, verifies network properties, and supports exporting updated graph structures.

## Setup

Install required libraries:

```bash
pip install networkx matplotlib numpy scipy
```

## Usage

General syntax:

```bash
python graph_analysis.py graph_file.gml [OPTIONS]
```

## Example Commands

Plot clustering coefficients:

```bash
python graph_analysis.py data/karate.gml --plot C
```

Partition into communities:

```bash
python graph_analysis.py data/karate.gml --components 3
```

Simulate edge failures:

```bash
python graph_analysis.py data/karate.gml --simulate_failures 5
```

Run robustness analysis:

```bash
python graph_analysis.py data/karate.gml --robustness_check 5
```

Verify homophily:

```bash
python graph_analysis.py data/homophily.gml --verify_homophily
```

Verify structural balance:

```bash
python graph_analysis.py data/balanced_graph.gml --verify_balanced_graph
```

Run temporal simulation:

```bash
python graph_analysis.py data/karate.gml --temporal_simulation data/edge.csv
```

Export updated graph:

```bash
python graph_analysis.py data/karate.gml --output output.gml
```

## Implemented Features

* Clustering coefficient (node attribute)
* Neighborhood overlap (edge attribute)
* Girvan–Newman community detection
* Random edge failure simulation
* Robustness analysis (multiple trials)
* Homophily testing (t-test)
* Structural balance verification (BFS-based)
* Temporal graph updates from CSV
* Export to `.gml`

## Included Test Data

The `data/` folder contains example graphs for testing:

* `karate.gml`
* `balanced_graph.gml`
* `imbalanced_graph.gml`
* `homophily.gml`
* `edge.csv`
//! Fleet reorganization suggestions.
//!
//! Suggests adding/removing edges to optimize the fleet graph:
//! - Maximize spectral gap (improve resilience)
//! - Minimize diameter (reduce latency)
//! - Balance load (equalize degree distribution)

use crate::fleet_graph::{CommEdge, FleetGraph};
use crate::laplacian::spectrum;

/// A suggested fleet reorganization.
#[derive(Debug, Clone)]
pub struct Reorganization {
    /// Edges to add.
    pub add_edges: Vec<(usize, usize)>,
    /// Edges to remove.
    pub remove_edges: Vec<(usize, usize)>,
    /// Expected improvement in spectral gap.
    pub expected_improvement: f64,
}

impl Reorganization {
    pub fn new(
        add_edges: Vec<(usize, usize)>,
        remove_edges: Vec<(usize, usize)>,
        expected_improvement: f64,
    ) -> Self {
        Self {
            add_edges,
            remove_edges,
            expected_improvement,
        }
    }
}

/// Compute graph diameter (longest shortest path) using BFS.
pub fn diameter(graph: &FleetGraph) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 0.0;
    }
    let adj = graph.undirected_adjacency();
    let mut max_dist = 0.0_f64;

    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] > 0.0 && dist[v] == f64::INFINITY {
                    dist[v] = dist[u] + 1.0;
                    max_dist = max_dist.max(dist[v]);
                    queue.push_back(v);
                }
            }
        }
    }
    max_dist
}

/// Compute degree balance: ratio of min degree to max degree (1.0 = perfect balance).
pub fn degree_balance(graph: &FleetGraph) -> f64 {
    let n = graph.node_count();
    if n == 0 {
        return 1.0;
    }
    let adj = graph.undirected_adjacency();
    let mut degrees: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        degrees.push((0..n).map(|j| adj[i][j]).sum());
    }
    let max_deg = degrees.iter().cloned().fold(0.0_f64, f64::max);
    let min_deg = degrees.iter().cloned().fold(f64::INFINITY, f64::min);
    if max_deg == 0.0 {
        1.0
    } else {
        min_deg / max_deg
    }
}

/// Compute average shortest path length.
pub fn avg_shortest_path(graph: &FleetGraph) -> f64 {
    let n = graph.node_count();
    if n <= 1 {
        return 0.0;
    }
    let adj = graph.undirected_adjacency();
    let mut total = 0.0;
    let mut count = 0;

    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        dist[s] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] > 0.0 && dist[v] == f64::INFINITY {
                    dist[v] = dist[u] + 1.0;
                    total += dist[v];
                    count += 1;
                    queue.push_back(v);
                }
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Suggest reorganization to maximize spectral gap.
/// Greedy approach: try adding each missing edge and pick the one that improves
/// the spectral gap the most.
pub fn suggest_for_spectral_gap(graph: &FleetGraph, max_suggestions: usize) -> Reorganization {
    let n = graph.node_count();
    if n < 2 {
        return Reorganization::new(vec![], vec![], 0.0);
    }

    let adj = graph.undirected_adjacency();
    let baseline = spectrum(graph);
    let baseline_gap = baseline.spectral_gap;

    let mut candidates = Vec::new();

    // Find missing edges
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] == 0.0 {
                // Simulate adding this edge
                let mut modified = graph.clone();
                modified.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                modified.add_edge(CommEdge::new(j, i, 1.0, 1.0));
                let new_spec = spectrum(&modified);
                let improvement = new_spec.spectral_gap - baseline_gap;
                if improvement > 0.0 {
                    candidates.push(((i, j), improvement));
                }
            }
        }
    }

    // Sort by improvement, descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_suggestions);

    let total_improvement: f64 = candidates.iter().map(|(_, imp)| *imp).sum();
    let add_edges: Vec<(usize, usize)> = candidates.into_iter().map(|(e, _)| e).collect();

    Reorganization::new(add_edges, vec![], total_improvement)
}

/// Suggest reorganization to minimize diameter.
/// Add edges between far-apart nodes.
pub fn suggest_for_diameter(graph: &FleetGraph, max_suggestions: usize) -> Reorganization {
    let n = graph.node_count();
    if n < 2 {
        return Reorganization::new(vec![], vec![], 0.0);
    }

    let adj = graph.undirected_adjacency();
    let baseline_diameter = diameter(graph);

    // BFS from all nodes to find far pairs
    let mut all_dists = vec![vec![f64::INFINITY; n]; n];
    for s in 0..n {
        all_dists[s][s] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] > 0.0 && all_dists[s][v] == f64::INFINITY {
                    all_dists[s][v] = all_dists[s][u] + 1.0;
                    queue.push_back(v);
                }
            }
        }
    }

    // Find far-apart pairs that are not connected
    let mut candidates = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] == 0.0 && all_dists[i][j] < f64::INFINITY {
                candidates.push(((i, j), all_dists[i][j]));
            }
        }
    }

    // Sort by distance (add edges between farthest pairs first)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_suggestions);

    let improvement = baseline_diameter;
    let add_edges: Vec<(usize, usize)> = candidates.into_iter().map(|(e, _)| e).collect();

    Reorganization::new(add_edges, vec![], improvement)
}

/// Suggest reorganization to balance load.
/// Add edges to low-degree nodes, remove edges from high-degree nodes.
pub fn suggest_for_balance(graph: &FleetGraph, max_suggestions: usize) -> Reorganization {
    let n = graph.node_count();
    if n < 2 {
        return Reorganization::new(vec![], vec![], 0.0);
    }

    let adj = graph.undirected_adjacency();
    let mut degrees: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let deg: f64 = (0..n).map(|j| adj[i][j]).sum();
            (i, deg)
        })
        .collect();

    degrees.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut add_edges = Vec::new();
    let mut remove_edges = Vec::new();

    // Connect low-degree nodes to each other
    let low_count = (n / 3).max(1);
    for i in 0..low_count.min(degrees.len()) {
        for j in (i + 1)..low_count.min(degrees.len()) {
            let (a, _) = degrees[i];
            let (b, _) = degrees[j];
            if adj[a][b] == 0.0 {
                add_edges.push((a, b));
                if add_edges.len() >= max_suggestions {
                    break;
                }
            }
        }
        if add_edges.len() >= max_suggestions {
            break;
        }
    }

    // Find redundant edges from high-degree nodes
    let high_start = n.saturating_sub(n / 3);
    for i in high_start..degrees.len() {
        let (a, deg_a) = degrees[i];
        if deg_a > (n - 1) as f64 / 2.0 {
            for j in 0..n {
                if a != j && adj[a][j] > 0.0 {
                    let deg_j: f64 = (0..n).map(|k| adj[j][k]).sum();
                    if deg_j > (n - 1) as f64 / 2.0 {
                        remove_edges.push((a.min(j), a.max(j)));
                    }
                }
            }
        }
    }
    remove_edges.truncate(max_suggestions);

    let baseline = degree_balance(graph);
    let expected_improvement = 1.0 - baseline;

    Reorganization::new(add_edges, remove_edges, expected_improvement)
}

/// Comprehensive reorganization suggestion combining all objectives.
pub fn suggest_reorganization(graph: &FleetGraph, max_suggestions: usize) -> Reorganization {
    let gap_reorg = suggest_for_spectral_gap(graph, max_suggestions);
    let diam_reorg = suggest_for_diameter(graph, max_suggestions);
    let balance_reorg = suggest_for_balance(graph, max_suggestions);

    // Merge, deduplicate, and sort by expected improvement
    let mut all_adds = Vec::new();
    all_adds.extend(gap_reorg.add_edges);
    all_adds.extend(diam_reorg.add_edges);
    all_adds.extend(balance_reorg.add_edges);

    all_adds.sort();
    all_adds.dedup();
    all_adds.truncate(max_suggestions);

    let mut all_removes = Vec::new();
    all_removes.extend(balance_reorg.remove_edges);
    all_removes.sort();
    all_removes.dedup();
    all_removes.truncate(max_suggestions);

    let improvement = gap_reorg.expected_improvement
        + diam_reorg.expected_improvement
        + balance_reorg.expected_improvement;

    Reorganization::new(all_adds, all_removes, improvement)
}

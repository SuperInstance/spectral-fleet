//! Bottleneck detection in fleet communication graphs.
//!
//! Identifies agents that are critical communication hubs using:
//! - Betweenness centrality (shortest-path based)
//! - Spectral centrality (eigenvector-based importance)
//! - Bridge detection (edges whose removal disconnects the graph)
//!
//! Suggests replication of bottleneck agents or bypass edges.

use crate::fleet_graph::FleetGraph;

/// A detected bottleneck agent.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Index of the bottleneck agent.
    pub agent: usize,
    /// Betweenness centrality score [0, 1].
    pub betweenness: f64,
    /// Spectral centrality score.
    pub spectral_centrality: f64,
    /// Whether this agent is a bridge (its removal would disconnect the graph).
    pub is_bridge: bool,
}

impl Bottleneck {
    pub fn new(agent: usize, betweenness: f64, spectral_centrality: f64, is_bridge: bool) -> Self {
        Self {
            agent,
            betweenness,
            spectral_centrality,
            is_bridge,
        }
    }
}

/// A suggested bypass edge to reduce bottleneck impact.
#[derive(Debug, Clone)]
pub struct BypassSuggestion {
    pub from: usize,
    pub to: usize,
    pub reason: String,
    pub estimated_improvement: f64,
}

/// Compute shortest paths from source using BFS on unweighted undirected graph.
/// Returns (distances, predecessor map).
fn bfs_shortest_paths(graph: &FleetGraph, source: usize) -> (Vec<Option<usize>>, Vec<f64>) {
    let n = graph.node_count();
    let adj = graph.undirected_adjacency();
    let mut dist = vec![f64::INFINITY; n];
    let mut pred = vec![None::<usize>; n];
    dist[source] = 0.0;
    pred[source] = Some(source);

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if adj[u][v] > 0.0 && dist[v] == f64::INFINITY {
                dist[v] = dist[u] + 1.0;
                pred[v] = Some(u);
                queue.push_back(v);
            }
        }
    }
    (pred, dist)
}

/// Count shortest paths through each node using BFS from source.
/// Returns (number_of_shortest_paths_to_each_node, dependency of each node).
fn brandes_bfs(
    graph: &FleetGraph,
    source: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = graph.node_count();
    let adj = graph.undirected_adjacency();

    let mut sigma = vec![0.0_f64; n]; // number of shortest paths
    let mut dist = vec![-1.0_f64; n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut stack = Vec::new();

    sigma[source] = 1.0;
    dist[source] = 0.0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        for w in 0..n {
            if adj[v][w] > 0.0 {
                // w is a neighbor of v
                if dist[w] < 0.0 {
                    // w found for the first time
                    dist[w] = dist[v] + 1.0;
                    queue.push_back(w);
                }
                if (dist[w] - dist[v] - 1.0).abs() < 1e-10 {
                    // Shortest path to w via v
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }
    }

    // Back-propagation
    let mut delta = vec![0.0_f64; n];
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
    }

    (sigma, delta)
}

/// Compute betweenness centrality for all agents.
/// Uses Brandes' algorithm for undirected, unweighted graphs.
pub fn betweenness_centrality(graph: &FleetGraph) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    let mut cb = vec![0.0; n];

    for s in 0..n {
        let (_, delta) = brandes_bfs(graph, s);
        for w in 0..n {
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Normalize for undirected graph: divide by 2 and by (n-1)(n-2)
    let norm = if n > 2 {
        2.0 / ((n - 1) as f64 * (n - 2) as f64)
    } else {
        1.0
    };
    for x in &mut cb {
        *x *= norm;
    }

    cb
}

/// Compute spectral centrality using the eigenvector of the largest eigenvalue
/// of the adjacency matrix.
pub fn spectral_centrality(graph: &FleetGraph) -> Vec<f64> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }
    let adj = graph.undirected_adjacency();

    // Power iteration for largest eigenvalue
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..1000 {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += adj[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for x in new_v.iter_mut() {
            *x /= norm;
        }
        v = new_v;
    }

    // Make all positive (Perron-Frobenius: dominant eigenvector has all positive entries for connected graph)
    let min_val = v.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_val < 0.0 {
        for x in &mut v {
            *x = x.abs();
        }
    }

    // Normalize to sum to 1
    let sum: f64 = v.iter().sum();
    if sum > 1e-15 {
        for x in &mut v {
            *x /= sum;
        }
    }

    v
}

/// Detect all bottleneck agents in the fleet.
pub fn detect_bottlenecks(graph: &FleetGraph, threshold: f64) -> Vec<Bottleneck> {
    let n = graph.node_count();
    let bc = betweenness_centrality(graph);
    let sc = spectral_centrality(graph);
    let bridges = find_bridge_agents(graph);

    let mut bottlenecks = Vec::new();
    for i in 0..n {
        let is_bridge = bridges.contains(&i);
        if bc[i] >= threshold || is_bridge {
            bottlenecks.push(Bottleneck::new(i, bc[i], sc.get(i).copied().unwrap_or(0.0), is_bridge));
        }
    }

    bottlenecks.sort_by(|a, b| b.betweenness.partial_cmp(&a.betweenness).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks
}

/// Find agents whose removal would disconnect the graph (articulation points).
pub fn find_bridge_agents(graph: &FleetGraph) -> Vec<usize> {
    let n = graph.node_count();
    if n <= 2 {
        return vec![];
    }
    let adj = graph.undirected_adjacency();

    let mut articulation_points = std::collections::HashSet::new();
    let mut discovery_time = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut time = 0u32;

    fn dfs(
        u: usize,
        parent: Option<usize>,
        adj: &[Vec<f64>],
        visited: &mut [bool],
        disc: &mut [u32],
        low: &mut [u32],
        ap: &mut std::collections::HashSet<usize>,
        time: &mut u32,
    ) {
        visited[u] = true;
        *time += 1;
        disc[u] = *time;
        low[u] = *time;
        let mut children = 0;

        for v in 0..adj.len() {
            if adj[u][v] > 0.0 {
                if !visited[v] {
                    children += 1;
                    dfs(v, Some(u), adj, visited, disc, low, ap, time);
                    low[u] = low[u].min(low[v]);

                    // u is an AP if:
                    if parent.is_none() && children > 1 {
                        ap.insert(u);
                    }
                    if parent.is_some() && low[v] >= disc[u] {
                        ap.insert(u);
                    }
                } else if let Some(p) = parent {
                    if v != p {
                        low[u] = low[u].min(disc[v]);
                    }
                }
            }
        }
    }

    // Run DFS from each unvisited node
    for i in 0..n {
        if !visited[i] {
            dfs(i, None, &adj, &mut visited, &mut discovery_time, &mut low, &mut articulation_points, &mut time);
        }
    }

    let mut result: Vec<usize> = articulation_points.into_iter().collect();
    result.sort();
    result
}

/// Find bridge edges whose removal would disconnect the graph.
pub fn find_bridge_edges(graph: &FleetGraph) -> Vec<(usize, usize)> {
    let n = graph.node_count();
    let adj = graph.undirected_adjacency();

    let mut bridges = Vec::new();
    let mut discovery_time = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut time = 0u32;

    fn dfs(
        u: usize,
        parent: Option<usize>,
        adj: &[Vec<f64>],
        visited: &mut [bool],
        disc: &mut [u32],
        low: &mut [u32],
        bridges: &mut Vec<(usize, usize)>,
        time: &mut u32,
    ) {
        visited[u] = true;
        *time += 1;
        disc[u] = *time;
        low[u] = *time;

        for v in 0..adj.len() {
            if adj[u][v] > 0.0 {
                if !visited[v] {
                    dfs(v, Some(u), adj, visited, disc, low, bridges, time);
                    low[u] = low[u].min(low[v]);

                    if low[v] > disc[u] {
                        bridges.push((u.min(v), u.max(v)));
                    }
                } else if let Some(p) = parent {
                    if v != p {
                        low[u] = low[u].min(disc[v]);
                    }
                }
            }
        }
    }

    for i in 0..n {
        if !visited[i] {
            dfs(i, None, &adj, &mut visited, &mut discovery_time, &mut low, &mut bridges, &mut time);
        }
    }

    bridges.sort();
    bridges.dedup();
    bridges
}

/// Suggest bypass edges to reduce bottleneck impact.
pub fn suggest_bypasses(graph: &FleetGraph) -> Vec<BypassSuggestion> {
    let bottlenecks = detect_bottlenecks(graph, 0.1);
    let adj = graph.undirected_adjacency();
    let mut suggestions = Vec::new();

    for bn in &bottlenecks {
        let neighbors = graph.neighbors(bn.agent);
        // Suggest connecting pairs of the bottleneck's neighbors directly
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let a = neighbors[i];
                let b = neighbors[j];
                if adj[a][b] == 0.0 {
                    suggestions.push(BypassSuggestion {
                        from: a,
                        to: b,
                        reason: format!(
                            "Bypass for bottleneck agent '{}' (betweenness={:.3})",
                            graph.agents[bn.agent].id, bn.betweenness
                        ),
                        estimated_improvement: bn.betweenness * 0.5,
                    });
                }
            }
        }
    }

    suggestions
}

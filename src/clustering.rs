//! Spectral clustering of fleet agents.
//!
//! Agents are grouped by communication patterns using spectral methods:
//! 1. Compute the k smallest eigenvectors of the graph Laplacian
//! 2. Embed each agent in R^k using these eigenvectors
//! 3. Apply k-means clustering in the embedded space
//!
//! The optimal number of clusters k is determined by the eigengap heuristic:
//! choose k that maximizes the gap between consecutive eigenvalues.

use crate::fleet_graph::FleetGraph;
use crate::laplacian::{combinatorial_laplacian, eigen_decompose};

/// A cluster of agents.
#[derive(Debug, Clone)]
pub struct FleetCluster {
    /// Agent indices in this cluster.
    pub agents: Vec<usize>,
    /// Cohesion: average pairwise weight within the cluster.
    pub cohesion: f64,
    /// Separator: agents on the boundary of this cluster (connected to other clusters).
    pub separator: Vec<usize>,
}

impl FleetCluster {
    pub fn new(agents: Vec<usize>, cohesion: f64, separator: Vec<usize>) -> Self {
        Self {
            agents,
            cohesion,
            separator,
        }
    }
}

/// Determine the optimal number of clusters using the eigengap heuristic.
/// Looks for the largest gap between consecutive eigenvalues.
pub fn optimal_k(eigenvalues: &[f64], max_k: usize) -> usize {
    if eigenvalues.len() < 2 {
        return 1;
    }
    let max_k = max_k.min(eigenvalues.len() / 2).max(2);
    let mut best_k = 1;
    let mut best_gap = 0.0;
    for k in 1..max_k {
        let gap = eigenvalues[k] - eigenvalues[k - 1];
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k.max(1)
}

/// Simple k-means clustering in R^k.
/// Returns cluster assignments: cluster_id for each point.
fn k_means(data: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = data.len();
    let dim = if n > 0 { data[0].len() } else { 0 };
    if n == 0 || k == 0 || dim == 0 {
        return vec![];
    }
    if k == 1 {
        return vec![0; n];
    }
    if k >= n {
        return (0..n).collect();
    }

    // Initialize centroids: spread evenly across sorted data
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    for c in 0..k {
        let idx = (c as f64 * (n - 1) as f64 / (k - 1) as f64).round() as usize;
        centroids.push(data[idx.min(n - 1)].clone());
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for c in 0..k {
                let dist: f64 = data[i]
                    .iter()
                    .zip(centroids[c].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                changed = true;
                assignments[i] = best_c;
            }
        }
        if !changed {
            break;
        }

        // Update centroids
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += data[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f64;
                }
            }
        }
    }

    assignments
}

/// Perform spectral clustering with a given number of clusters.
pub fn spectral_clustering(graph: &FleetGraph, k: usize) -> Vec<FleetCluster> {
    let n = graph.node_count();
    if n == 0 || k == 0 {
        return vec![];
    }

    let lap = combinatorial_laplacian(graph);
    let (_eigenvalues, eigenvectors) = eigen_decompose(&lap, 1000, 1e-10);

    // Use the k smallest eigenvectors for embedding
    let num_vecs = k.min(eigenvectors.len()).max(1);
    let mut embedding: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut point = Vec::with_capacity(num_vecs);
        for j in 0..num_vecs {
            point.push(eigenvectors[j][i]);
        }
        embedding.push(point);
    }

    // Normalize rows
    for point in &mut embedding {
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in point.iter_mut() {
                *x /= norm;
            }
        }
    }

    let assignments = k_means(&embedding, k, 100);

    // Build clusters
    let adj = graph.undirected_adjacency();
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c].push(i);
    }

    clusters
        .into_iter()
        .enumerate()
        .map(|(_cluster_id, agents)| {
            // Compute cohesion: average pairwise weight
            let cohesion = if agents.len() > 1 {
                let mut total = 0.0;
                let mut count = 0;
                for &a in &agents {
                    for &b in &agents {
                        if a != b {
                            total += adj[a][b];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    total / count as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Find separator: agents connected to agents outside this cluster
            let agent_set: std::collections::HashSet<usize> =
                agents.iter().copied().collect();
            let separator: Vec<usize> = agents
                .iter()
                .filter(|&&a| {
                    (0..n).any(|b| !agent_set.contains(&b) && adj[a][b] > 0.0)
                })
                .copied()
                .collect();

            FleetCluster::new(agents, cohesion, separator)
        })
        .filter(|c| !c.agents.is_empty())
        .collect()
}

/// Perform spectral clustering with automatic k selection via eigengap heuristic.
pub fn spectral_clustering_auto(graph: &FleetGraph, max_k: usize) -> Vec<FleetCluster> {
    let lap = combinatorial_laplacian(graph);
    let (eigenvalues, _) = eigen_decompose(&lap, 1000, 1e-10);
    let k = optimal_k(&eigenvalues, max_k);
    spectral_clustering(graph, k)
}

/// Compute cluster assignments as a vector of cluster IDs.
pub fn cluster_assignments(graph: &FleetGraph, k: usize) -> Vec<usize> {
    let clusters = spectral_clustering(graph, k);
    let n = graph.node_count();
    let mut assignments = vec![0usize; n];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &agent in &cluster.agents {
            assignments[agent] = cluster_id;
        }
    }
    assignments
}

//! # spectral-fleet — Spectral Graph Theory for Fleet Analysis
//!
//! Uses eigenvalues and eigenvectors of graph matrices to analyze fleet
//! connectivity, cluster agents, and detect structural patterns.

use std::collections::HashMap;

// ─── Graph ───────────────────────────────────────────────────────────────────

/// A weighted undirected graph.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n: usize,
    adj: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self { n, adj: vec![Vec::new(); n] }
    }

    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) {
        self.adj[i].push((j, w));
        self.adj[j].push((i, w));
    }

    /// Adjacency matrix (n×n).
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for &(j, w) in &self.adj[i] {
                m[i][j] = w;
            }
        }
        m
    }

    /// Degree matrix (diagonal).
    pub fn degree_matrix(&self) -> Vec<Vec<f64>> {
        let mut d = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            let degree: f64 = self.adj[i].iter().map(|(_, w)| w).sum();
            d[i][i] = degree;
        }
        d
    }

    /// Laplacian matrix: L = D - A.
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let a = self.adjacency_matrix();
        let d = self.degree_matrix();
        let mut l = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                l[i][j] = d[i][j] - a[i][j];
            }
        }
        l
    }

    /// Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}.
    pub fn normalized_laplacian(&self) -> Vec<Vec<f64>> {
        let d = self.degree_matrix();
        let l = self.laplacian();
        let mut ln = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                let di = if d[i][i] > 0.0 { 1.0 / d[i][i].sqrt() } else { 0.0 };
                let dj = if d[j][j] > 0.0 { 1.0 / d[j][j].sqrt() } else { 0.0 };
                ln[i][j] = di * l[i][j] * dj;
            }
        }
        ln
    }

    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|v| v.len()).sum::<usize>() / 2
    }

    /// Check if connected via BFS.
    pub fn is_connected(&self) -> bool {
        if self.n == 0 { return true; }
        let mut visited = vec![false; self.n];
        let mut queue = vec![0];
        visited[0] = true;
        while !queue.is_empty() {
            let v = queue.remove(0);
            for &(u, _) in &self.adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    queue.push(u);
                }
            }
        }
        visited.iter().all(|&v| v)
    }
}

// ─── Power Iteration ─────────────────────────────────────────────────────────

/// Power iteration for finding the largest eigenvalue/eigenvector.
pub fn power_iteration(matrix: &[Vec<f64>], iterations: usize) -> (f64, Vec<f64>) {
    let n = matrix.len();
    let mut v = vec![1.0; n];
    let norm = (n as f64).sqrt();
    for x in &mut v { *x /= norm; }

    for _ in 0..iterations {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += matrix[i][j] * v[j];
            }
        }
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut new_v { *x /= norm; }
        }
        v = new_v;
    }

    // Rayleigh quotient
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix[i][j] * v[j];
        }
    }
    let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();

    (eigenvalue, v)
}

// ─── Spectral Clustering ────────────────────────────────────────────────────

/// Cluster agents using the Fiedler vector (eigenvector of second-smallest Laplacian eigenvalue).
pub fn spectral_cluster(graph: &Graph, k: usize) -> Vec<Vec<usize>> {
    let lap = graph.laplacian();

    // For k=2, use sign of Fiedler vector
    // Shift Laplacian to find smallest eigenvalue → use -L + shift
    let shift = graph.n as f64 * 4.0;
    let mut shifted = vec![vec![0.0; graph.n]; graph.n];
    for i in 0..graph.n {
        for j in 0..graph.n {
            shifted[i][j] = -lap[i][j] + if i == j { shift } else { 0.0 };
        }
    }

    let (_, fiedler) = power_iteration(&shifted, 100);

    if k <= 2 {
        let mut c1 = Vec::new();
        let mut c2 = Vec::new();
        for (i, &v) in fiedler.iter().enumerate() {
            if v >= 0.0 { c1.push(i); } else { c2.push(i); }
        }
        vec![c1, c2]
    } else {
        // Recursive: split into 2, then recursively split larger cluster
        let mut clusters = spectral_cluster(graph, 2);
        while clusters.len() < k {
            let largest_idx = clusters.iter().enumerate().max_by_key(|(_, c)| c.len()).map(|(i, _)| i).unwrap();
            let largest = clusters.remove(largest_idx);
            // Just split in half by Fiedler value ordering
            let mut sorted = largest;
            sorted.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());
            let mid = sorted.len() / 2;
            clusters.push(sorted[..mid].to_vec());
            clusters.push(sorted[mid..].to_vec());
        }
        clusters
    }
}

// ─── Connectivity Metrics ───────────────────────────────────────────────────

/// Algebraic connectivity = second smallest eigenvalue of Laplacian.
pub fn algebraic_connectivity(graph: &Graph) -> f64 {
    // Compute the Fiedler value (2nd smallest Laplacian eigenvalue)
    // Step 1: Find the largest eigenvector of shift - L (which is the constant vector)
    // Step 2: Deflate it out, then find the next largest
    let lap = graph.laplacian();
    let trace: f64 = (0..graph.n).map(|i| lap[i][i]).sum();
    let shift = trace + 1.0;
    let mut shifted = vec![vec![0.0; graph.n]; graph.n];
    for i in 0..graph.n {
        for j in 0..graph.n {
            shifted[i][j] = -lap[i][j] + if i == j { shift } else { 0.0 };
        }
    }
    
    // First eigenvector (constant) via power iteration
    let (e1, v1) = power_iteration(&shifted, 200);
    
    // Deflate
    let n = graph.n;
    let mut deflated = shifted.clone();
    for i in 0..n {
        for j in 0..n {
            deflated[i][j] -= e1 * v1[i] * v1[j];
        }
    }
    
    // Second eigenvector gives the Fiedler vector
    let (e2, _) = power_iteration(&deflated, 500);
    shift - e2
}

/// Spectral gap = difference between largest and second-largest eigenvalues.
pub fn spectral_gap(adj_matrix: &[Vec<f64>]) -> f64 {
    let (e1, _) = power_iteration(adj_matrix, 100);
    // Deflate
    let n = adj_matrix.len();
    let mut deflated = adj_matrix.to_vec();
    let (_, v1) = power_iteration(adj_matrix, 100);
    for i in 0..n {
        for j in 0..n {
            deflated[i][j] -= e1 * v1[i] * v1[j];
        }
    }
    let (e2, _) = power_iteration(&deflated, 100);
    (e1 - e2).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let g = Graph::new(5);
        assert_eq!(g.n, 5);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_graph_edges() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 3.0);
        let a = g.adjacency_matrix();
        assert!((a[0][1] - 3.0).abs() < 0.001);
        assert!((a[1][0] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_degree_matrix() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        let d = g.degree_matrix();
        assert!((d[0][0] - 1.0).abs() < 0.001);
        assert!((d[1][1] - 3.0).abs() < 0.001);
        assert!((d[2][2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_laplacian() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let l = g.laplacian();
        assert!((l[0][0] - 1.0).abs() < 0.001);
        assert!((l[1][1] - 2.0).abs() < 0.001);
        assert!((l[0][1] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_is_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert!(g.is_connected());
    }

    #[test]
    fn test_not_connected() {
        let g = Graph::new(3);
        assert!(!g.is_connected());
    }

    #[test]
    fn test_power_iteration() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (eigenvalue, v) = power_iteration(&m, 100);
        assert!(eigenvalue > 3.0); // Largest eigenvalue of [[2,1],[1,3]] ≈ 3.618
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_spectral_cluster_2() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let clusters = spectral_cluster(&g, 2);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_algebraic_connectivity_connected() {
        // Complete graph K₄ has algebraic connectivity = 4
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let ac = algebraic_connectivity(&g);
        assert!(ac > 1.0); // K₄ should have high algebraic connectivity
    }

    #[test]
    fn test_algebraic_connectivity_disconnected() {
        let g = Graph::new(3); // No edges
        let ac = algebraic_connectivity(&g);
        // Disconnected graph has algebraic connectivity = 0
        assert!(ac.abs() < 2.0);
    }

    #[test]
    fn test_normalized_laplacian() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 1.0);
        let ln = g.normalized_laplacian();
        // For 2-node graph: [[1, -1], [-1, 1]]
        assert!((ln[0][0] - 1.0).abs() < 0.01);
        assert!((ln[0][1] - (-1.0)).abs() < 0.01);
    }
}

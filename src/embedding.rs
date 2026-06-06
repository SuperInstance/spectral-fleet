//! Spectral embedding: map agents to low-dimensional space.
//!
//! Uses the smallest eigenvectors of the graph Laplacian to embed agents
//! in R^k such that distance in embedded space approximates communication cost.
//! Supports 2D visualization using a t-SNE-like approach.

use crate::fleet_graph::FleetGraph;
use crate::laplacian::{combinatorial_laplacian, eigen_decompose};

/// A spectral embedding of the fleet graph.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Number of dimensions.
    pub dimensions: usize,
    /// Coordinates of each agent in R^dimensions.
    pub coordinates: Vec<Vec<f64>>,
    /// Reconstruction error (how well the embedding preserves distances).
    pub reconstruction_error: f64,
}

impl Embedding {
    pub fn new(dimensions: usize, coordinates: Vec<Vec<f64>>) -> Self {
        Self {
            dimensions,
            coordinates,
            reconstruction_error: 0.0,
        }
    }

    /// Compute pairwise distances in the embedded space.
    pub fn pairwise_distances(&self) -> Vec<Vec<f64>> {
        let n = self.coordinates.len();
        let mut dists = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d: f64 = self.coordinates[i]
                    .iter()
                    .zip(self.coordinates[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dists[i][j] = d;
                dists[j][i] = d;
            }
        }
        dists
    }
}

/// Compute spectral embedding using the k smallest eigenvectors of the Laplacian.
pub fn spectral_embedding(graph: &FleetGraph, dimensions: usize) -> Embedding {
    let n = graph.node_count();
    if n == 0 {
        return Embedding::new(dimensions, vec![]);
    }

    let lap = combinatorial_laplacian(graph);
    let (eigenvalues, eigenvectors) = eigen_decompose(&lap, 1000, 1e-10);

    // Skip the first eigenvector (constant vector for connected graph)
    // Use eigenvectors 1..dimensions+1
    let num_vecs = dimensions.min(eigenvalues.len().saturating_sub(1)).max(1);

    let mut coordinates = Vec::with_capacity(n);
    for i in 0..n {
        let mut point = Vec::with_capacity(num_vecs);
        for j in 1..=num_vecs {
            if j < eigenvectors.len() {
                point.push(eigenvectors[j][i]);
            } else {
                point.push(0.0);
            }
        }
        coordinates.push(point);
    }

    // Compute reconstruction error
    let embedding = Embedding::new(num_vecs, coordinates);
    let error = compute_reconstruction_error(graph, &embedding);

    Embedding {
        dimensions: num_vecs,
        coordinates: embedding.coordinates,
        reconstruction_error: error,
    }
}

/// Compute how well the embedding preserves graph distances.
/// Lower is better. Uses stress function.
fn compute_reconstruction_error(graph: &FleetGraph, embedding: &Embedding) -> f64 {
    let n = graph.node_count();
    if n < 2 {
        return 0.0;
    }

    let adj = graph.undirected_adjacency();
    let embed_dists = embedding.pairwise_distances();

    // Graph shortest path distances
    let mut graph_dists = vec![vec![f64::INFINITY; n]; n];
    for s in 0..n {
        graph_dists[s][s] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] > 0.0 && graph_dists[s][v] == f64::INFINITY {
                    graph_dists[s][v] = graph_dists[s][u] + 1.0;
                    queue.push_back(v);
                }
            }
        }
    }

    // Stress: sum of (embed_dist - graph_dist)^2 / graph_dist^2
    let mut stress = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let gd = graph_dists[i][j];
            if gd.is_finite() && gd > 0.0 {
                let ed = embed_dists[i][j];
                stress += ((ed - gd) / gd).powi(2);
                count += 1;
            }
        }
    }
    if count > 0 {
        stress / count as f64
    } else {
        0.0
    }
}

/// t-SNE-style embedding for 2D visualization.
/// Uses a simplified Barnes-Hut t-SNE approach.
pub fn tsne_embedding(
    graph: &FleetGraph,
    perplexity: f64,
    iterations: usize,
) -> Embedding {
    let n = graph.node_count();
    if n == 0 {
        return Embedding::new(2, vec![]);
    }

    // Start from spectral embedding
    let initial = spectral_embedding(graph, 2);

    let adj = graph.undirected_adjacency();

    // Compute graph shortest path distances as the "high-dimensional" distances
    let mut high_dists = vec![vec![f64::INFINITY; n]; n];
    for s in 0..n {
        high_dists[s][s] = 0.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] > 0.0 && high_dists[s][v] == f64::INFINITY {
                    high_dists[s][v] = high_dists[s][u] + 1.0;
                    queue.push_back(v);
                }
            }
        }
    }

    // Convert to similarity (Gaussian kernel)
    let sigma = perplexity;
    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && high_dists[i][j].is_finite() {
                p[i][j] = (-high_dists[i][j].powi(2) / (2.0 * sigma * sigma)).exp();
            }
        }
        // Normalize
        let row_sum: f64 = p[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..n {
                p[i][j] /= row_sum;
            }
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in 0..n {
            p[i][j] = (p[i][j] + p[j][i]) / (2.0 * n as f64);
        }
    }

    // Initialize from spectral embedding
    let mut y: Vec<Vec<f64>> = initial.coordinates.clone();
    // Pad to 2D if needed
    for point in &mut y {
        while point.len() < 2 {
            point.push(0.0);
        }
        point.truncate(2);
    }

    // Gradient descent
    let mut gains = vec![vec![1.0_f64; 2]; n];
    let mut y_vel = vec![vec![0.0_f64; 2]; n];
    let learning_rate = 100.0;
    let momentum = 0.8;

    for _ in 0..iterations {
        // Compute low-dimensional similarities (t-distribution)
        let mut q = vec![vec![0.0; n]; n];
        let mut q_sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq: f64 = y[i]
                    .iter()
                    .zip(y[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let val = 1.0 / (1.0 + dist_sq);
                q[i][j] = val;
                q[j][i] = val;
                q_sum += 2.0 * val;
            }
        }
        if q_sum > 0.0 {
            for i in 0..n {
                for j in 0..n {
                    q[i][j] /= q_sum;
                }
            }
        }

        // Compute gradients
        let mut gradients = vec![vec![0.0; 2]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mult = 4.0 * (p[i][j] - q[i][j]);
                let dist_sq: f64 = y[i]
                    .iter()
                    .zip(y[j].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                let factor = mult / (1.0 + dist_sq);
                for d in 0..2 {
                    gradients[i][d] += factor * (y[i][d] - y[j][d]);
                }
            }
        }

        // Update positions
        for i in 0..n {
            for d in 0..2 {
                let grad_sign = gradients[i][d].signum();
                let vel_sign = y_vel[i][d].signum();
                gains[i][d] = if grad_sign != vel_sign {
                    gains[i][d] + 0.2_f64
                } else {
                    gains[i][d] * 0.8_f64
                };
                gains[i][d] = f64::max(gains[i][d], 0.01_f64);
                y_vel[i][d] = momentum * y_vel[i][d] - learning_rate * gains[i][d] * gradients[i][d];
                y[i][d] += y_vel[i][d];
            }
        }
    }

    // Center the embedding
    let mut center = vec![0.0; 2];
    for point in &y {
        for d in 0..2 {
            center[d] += point[d];
        }
    }
    for d in 0..2 {
        center[d] /= n as f64;
    }
    for point in &mut y {
        for d in 0..2 {
            point[d] -= center[d];
        }
    }

    let mut embedding = Embedding::new(2, y);
    embedding.reconstruction_error = compute_reconstruction_error(graph, &embedding);
    embedding
}

/// Render a 2D embedding as an ASCII art string.
pub fn render_ascii(embedding: &Embedding, width: usize, height: usize) -> String {
    if embedding.coordinates.is_empty() {
        return String::new();
    }

    let coords = &embedding.coordinates;
    let dims = coords[0].len();
    if dims < 2 {
        return String::new();
    }

    // Find bounds
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    for p in coords {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }

    let range_x = (max_x - min_x).max(1e-10);
    let range_y = (max_y - min_y).max(1e-10);

    let mut canvas = vec![vec![' '; width]; height];

    for (idx, p) in coords.iter().enumerate() {
        let x = ((p[0] - min_x) / range_x * (width - 1) as f64).round() as usize;
        let y = ((p[1] - min_y) / range_y * (height - 1) as f64).round() as usize;
        let x = x.min(width - 1);
        let y = y.min(height - 1);
        let ch = if idx < 26 {
            (b'A' + idx as u8) as char
        } else {
            '#'
        };
        canvas[height - 1 - y][x] = ch;
    }

    canvas
        .iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

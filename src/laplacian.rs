//! Graph Laplacian computation and eigenvalue decomposition.
//!
//! The Laplacian L = D - A encodes graph connectivity in its spectrum:
//! - Number of zero eigenvalues = number of connected components
//! - Fiedler value (2nd smallest eigenvalue) = algebraic connectivity
//! - Spectral gap = gap between largest and 2nd-largest eigenvalues
//!
//! We implement eigenvalue decomposition using the Jacobi eigenvalue algorithm
//! for symmetric matrices, which is well-suited for Laplacians.

use crate::fleet_graph::FleetGraph;

/// The spectral decomposition of a graph Laplacian.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Eigenvalues sorted in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors (column-major: eigenvectors[i] is the i-th eigenvector).
    pub eigenvectors: Vec<Vec<f64>>,
    /// The Fiedler value (2nd smallest eigenvalue). Zero for disconnected graphs.
    pub fiedler_value: f64,
    /// Spectral gap: difference between largest and 2nd-largest eigenvalue.
    pub spectral_gap: f64,
    /// Algebraic connectivity = Fiedler value.
    pub algebraic_connectivity: f64,
}

/// Compute the combinatorial Laplacian: L = D - A (undirected).
pub fn combinatorial_laplacian(graph: &FleetGraph) -> Vec<Vec<f64>> {
    let n = graph.node_count();
    let adj = graph.undirected_adjacency();
    let mut lap = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            if i != j {
                degree += adj[i][j];
                lap[i][j] = -adj[i][j];
            }
        }
        lap[i][i] = degree;
    }
    lap
}

/// Compute the normalized Laplacian: L_sys = I - D^{-1/2} A D^{-1/2}.
pub fn normalized_laplacian(graph: &FleetGraph) -> Option<Vec<Vec<f64>>> {
    let n = graph.node_count();
    let adj = graph.undirected_adjacency();
    let mut lap = vec![vec![0.0; n]; n];

    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        let degree: f64 = (0..n).map(|j| adj[i][j]).sum();
        d_inv_sqrt[i] = if degree > 0.0 { 1.0 / degree.sqrt() } else { 0.0 };
    }

    for i in 0..n {
        for j in 0..n {
            if i == j {
                lap[i][j] = 1.0;
            } else {
                lap[i][j] = -d_inv_sqrt[i] * adj[i][j] * d_inv_sqrt[j];
            }
        }
    }
    Some(lap)
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue ascending.
/// Eigenvectors are stored as columns: eigenvectors[i][j] = j-th component of i-th eigenvector.
pub(crate) fn eigen_decompose(mat: &[Vec<f64>], max_iter: usize, _tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Copy matrix into a flat array for faster access
    let mut a: Vec<f64> = mat.iter().flat_map(|row| row.iter().copied()).collect();

    // Initialize eigenvector matrix as identity
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation: A' = G^T A G
        // Update rows and columns p and q
        let mut new_a = a.clone();

        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }

        new_a[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;

        a = new_a;

        // Update eigenvectors: V' = V G
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues (diagonal of a)
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Extract eigenvectors (columns of v)
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| v[i * n + j]).collect())
        .collect();

    // Sort by eigenvalue ascending
    let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.into_iter().zip(eigenvectors.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let (sorted_vals, sorted_vecs): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    (sorted_vals, sorted_vecs)
}

/// Compute the spectral decomposition of a fleet graph.
pub fn spectrum(graph: &FleetGraph) -> Spectrum {
    let lap = combinatorial_laplacian(graph);
    let n = graph.node_count();

    if n == 0 {
        return Spectrum {
            eigenvalues: vec![],
            eigenvectors: vec![],
            fiedler_value: 0.0,
            spectral_gap: 0.0,
            algebraic_connectivity: 0.0,
        };
    }

    // For Jacobi, max_iter is number of sweeps
    let (eigenvalues, eigenvectors) = eigen_decompose(&lap, 100, 1e-10);

    // Fiedler value = 2nd smallest eigenvalue
    let fiedler_value = if eigenvalues.len() > 1 {
        eigenvalues[1]
    } else {
        0.0
    };

    // Spectral gap = largest - 2nd largest
    let spectral_gap = if eigenvalues.len() > 1 {
        let n_vals = eigenvalues.len();
        eigenvalues[n_vals - 1] - eigenvalues[n_vals - 2]
    } else {
        0.0
    };

    Spectrum {
        eigenvalues,
        eigenvectors,
        fiedler_value,
        spectral_gap,
        algebraic_connectivity: fiedler_value,
    }
}

/// Get the Fiedler vector (eigenvector corresponding to the 2nd smallest eigenvalue).
pub fn fiedler_vector(graph: &FleetGraph) -> Option<Vec<f64>> {
    let spec = spectrum(graph);
    if spec.eigenvectors.len() > 1 {
        Some(spec.eigenvectors[1].clone())
    } else {
        None
    }
}

/// Count connected components from the spectrum (number of zero eigenvalues).
pub fn count_components_from_spectrum(spec: &Spectrum) -> usize {
    spec.eigenvalues
        .iter()
        .filter(|&&e| e.abs() < 1e-6)
        .count()
}

/// Compute the adjacency matrix eigenvalues.
pub fn adjacency_spectrum(graph: &FleetGraph) -> (Vec<f64>, Vec<Vec<f64>>) {
    let adj = graph.undirected_adjacency();
    eigen_decompose(&adj, 100, 1e-10)
}

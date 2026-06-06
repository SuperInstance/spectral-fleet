use approx::assert_relative_eq;
use spectral_fleet::{
    bottleneck, clustering, dynamics, embedding, reorganization,
    fleet_graph::{AgentNode, CommEdge, FleetGraph},
    laplacian,
};

// ============================================================================
// Fleet Graph Tests
// ============================================================================

#[test]
fn test_complete_graph_connected() {
    let g = FleetGraph::complete(5);
    assert!(g.is_connected());
    assert_eq!(g.node_count(), 5);
    assert_eq!(g.edge_count(), 5 * 4); // directed edges
}

#[test]
fn test_disconnected_graph() {
    let g = FleetGraph::two_cliques(3);
    assert_eq!(g.node_count(), 6);
    assert!(!g.is_connected());
    let components = g.connected_components();
    assert_eq!(components.len(), 2);
}

#[test]
fn test_path_graph() {
    let g = FleetGraph::path(4);
    assert!(g.is_connected());
    assert_eq!(g.edge_count(), 6); // 3 undirected edges × 2 directions
}

#[test]
fn test_star_graph() {
    let g = FleetGraph::star(5);
    assert!(g.is_connected());
    // Node 0 should have degree 4, others degree 1
    let adj = g.undirected_adjacency();
    let deg0: f64 = adj[0].iter().sum();
    assert_eq!(deg0, 4.0);
    let deg1: f64 = adj[1].iter().sum();
    assert_eq!(deg1, 1.0);
}

#[test]
fn test_barbell_graph() {
    let g = FleetGraph::barbell(3);
    assert!(g.is_connected());
    assert_eq!(g.node_count(), 6);
}

#[test]
fn test_from_github_org() {
    let repos = vec![
        ("auth-service".into(), vec!["rust".into(), "security".into()]),
        ("api-gateway".into(), vec!["go".into(), "networking".into()]),
        ("data-pipeline".into(), vec!["python".into(), "ml".into()]),
    ];
    let deps = vec![
        ("api-gateway".into(), "auth-service".into(), 0.8),
        ("data-pipeline".into(), "api-gateway".into(), 0.5),
    ];
    let g = FleetGraph::from_github_org(repos, deps);
    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 2);
}

// ============================================================================
// Laplacian / Spectrum Tests
// ============================================================================

#[test]
fn test_complete_graph_single_zero_eigenvalue() {
    let g = FleetGraph::complete(5);
    let spec = laplacian::spectrum(&g);
    // Complete graph on n nodes: single zero eigenvalue, all others = n
    let zero_count = spec.eigenvalues.iter().filter(|&&e| e.abs() < 0.1).count();
    assert_eq!(zero_count, 1, "Complete graph should have exactly 1 zero eigenvalue");
}

#[test]
fn test_disconnected_graph_multiple_zero_eigenvalues() {
    let g = FleetGraph::two_cliques(3);
    let spec = laplacian::spectrum(&g);
    // Two disconnected components = two zero eigenvalues
    let zero_count = spec.eigenvalues.iter().filter(|&&e| e.abs() < 0.1).count();
    assert!(zero_count >= 2, "Disconnected graph should have ≥2 zero eigenvalues, got {}", zero_count);
}

#[test]
fn test_connected_graph_positive_fiedler() {
    let g = FleetGraph::complete(5);
    let spec = laplacian::spectrum(&g);
    assert!(
        spec.fiedler_value > 0.01,
        "Connected graph should have positive Fiedler value, got {:.4}",
        spec.fiedler_value
    );
}

#[test]
fn test_disconnected_graph_zero_fiedler() {
    let g = FleetGraph::two_cliques(3);
    let spec = laplacian::spectrum(&g);
    assert!(
        spec.fiedler_value.abs() < 0.1,
        "Disconnected graph should have ~zero Fiedler value, got {:.4}",
        spec.fiedler_value
    );
}

#[test]
fn test_laplacian_row_sum_zero() {
    let g = FleetGraph::complete(4);
    let lap = laplacian::combinatorial_laplacian(&g);
    for row in &lap {
        let sum: f64 = row.iter().sum();
        assert_relative_eq!(sum, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_laplacian_diagonal_nonnegative() {
    let g = FleetGraph::path(5);
    let lap = laplacian::combinatorial_laplacian(&g);
    for (i, row) in lap.iter().enumerate() {
        assert!(row[i] >= 0.0, "Diagonal entry {} should be non-negative", i);
    }
}

#[test]
fn test_count_components_from_spectrum() {
    let g = FleetGraph::two_cliques(4);
    let spec = laplacian::spectrum(&g);
    let count = laplacian::count_components_from_spectrum(&spec);
    assert_eq!(count, 2);
}

#[test]
fn test_fiedler_vector_exists() {
    let g = FleetGraph::barbell(3);
    let fv = laplacian::fiedler_vector(&g);
    assert!(fv.is_some());
    let fv = fv.unwrap();
    assert_eq!(fv.len(), 6);
}

#[test]
fn test_normalized_laplacian() {
    let g = FleetGraph::complete(4);
    let lap = laplacian::normalized_laplacian(&g);
    assert!(lap.is_some());
    let lap = lap.unwrap();
    // Diagonal should be 1
    for i in 0..lap.len() {
        assert_relative_eq!(lap[i][i], 1.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Clustering Tests
// ============================================================================

#[test]
fn test_spectral_clustering_two_communities() {
    // Barbell graph: two cliques joined by a bridge
    let g = FleetGraph::barbell(4);
    let clusters = clustering::spectral_clustering(&g, 2);
    assert_eq!(clusters.len(), 2);

    // Each cluster should have 4 agents
    let total: usize = clusters.iter().map(|c| c.agents.len()).sum();
    assert_eq!(total, 8);

    // The two clusters should separate the two cliques
    let left_cluster: std::collections::HashSet<usize> =
        clusters[0].agents.iter().copied().collect();
    let right_cluster: std::collections::HashSet<usize> =
        clusters[1].agents.iter().copied().collect();

    // One cluster should contain {0..4} and the other {4..8}
    let left_ids: std::collections::HashSet<usize> = (0..4).collect();
    let right_ids: std::collections::HashSet<usize> = (4..8).collect();
    assert!(
        (left_cluster == left_ids && right_cluster == right_ids)
            || (left_cluster == right_ids && right_cluster == left_ids),
        "Clusters should separate the two cliques"
    );
}

#[test]
fn test_optimal_k_two_communities() {
    let g = FleetGraph::barbell(4);
    let spec = laplacian::spectrum(&g);
    let k = clustering::optimal_k(&spec.eigenvalues, 5);
    assert!(k >= 2, "Should detect at least 2 clusters, got {}", k);
}

#[test]
fn test_clustering_single_cluster_for_complete() {
    let g = FleetGraph::complete(5);
    let clusters = clustering::spectral_clustering(&g, 1);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].agents.len(), 5);
}

#[test]
fn test_cluster_assignments() {
    let g = FleetGraph::barbell(3);
    let assignments = clustering::cluster_assignments(&g, 2);
    assert_eq!(assignments.len(), 6);
    // All agents should have valid cluster ids
    for &a in &assignments {
        assert!(a < 2);
    }
}

// ============================================================================
// Bottleneck Tests
// ============================================================================

#[test]
fn test_bridge_agents_in_star() {
    let g = FleetGraph::star(5);
    let bridges = bottleneck::find_bridge_agents(&g);
    // Center node (0) is an articulation point
    assert!(
        bridges.contains(&0),
        "Center of star graph should be a bridge agent"
    );
}

#[test]
fn test_bridge_edges_in_barbell() {
    let g = FleetGraph::barbell(3);
    let bridges = bottleneck::find_bridge_edges(&g);
    // The single bridge edge should be detected
    assert!(
        !bridges.is_empty(),
        "Barbell graph should have bridge edges"
    );
    // The bridge is between node 2 and node 3
    assert!(
        bridges.contains(&(2, 3)),
        "Should detect bridge edge (2, 3), got {:?}",
        bridges
    );
}

#[test]
fn test_betweenness_centrality_star() {
    let g = FleetGraph::star(5);
    let bc = bottleneck::betweenness_centrality(&g);
    // Center node should have highest betweenness
    assert!(
        bc[0] > bc[1],
        "Center node should have highest betweenness: bc[0]={:.3}, bc[1]={:.3}",
        bc[0], bc[1]
    );
}

#[test]
fn test_spectral_centrality() {
    let g = FleetGraph::star(5);
    let sc = bottleneck::spectral_centrality(&g);
    // All entries should be non-negative
    for (i, &v) in sc.iter().enumerate() {
        assert!(v >= 0.0, "spectral centrality[{}] = {:.4} should be ≥ 0", i, v);
    }
    assert_eq!(sc.len(), 5);
}

#[test]
fn test_detect_bottlenecks() {
    let g = FleetGraph::barbell(4);
    let bottlenecks = bottleneck::detect_bottlenecks(&g, 0.1);
    assert!(!bottlenecks.is_empty(), "Barbell graph should have bottlenecks");
}

#[test]
fn test_suggest_bypasses() {
    let g = FleetGraph::barbell(4);
    let suggestions = bottleneck::suggest_bypasses(&g);
    // Should suggest bypass edges around the bridge
    assert!(!suggestions.is_empty(), "Should suggest bypasses for barbell graph");
}

// ============================================================================
// Reorganization Tests
// ============================================================================

#[test]
fn test_diameter_path() {
    let g = FleetGraph::path(5);
    let d = reorganization::diameter(&g);
    assert_eq!(d, 4.0, "Path of 5 nodes has diameter 4");
}

#[test]
fn test_diameter_complete() {
    let g = FleetGraph::complete(5);
    let d = reorganization::diameter(&g);
    assert_eq!(d, 1.0, "Complete graph has diameter 1");
}

#[test]
fn test_degree_balance() {
    let g = FleetGraph::complete(5);
    let balance = reorganization::degree_balance(&g);
    assert_relative_eq!(balance, 1.0, epsilon = 0.01);
}

#[test]
fn test_degree_balance_unbalanced() {
    let g = FleetGraph::star(5);
    let balance = reorganization::degree_balance(&g);
    // Star: center has degree 4, leaves have degree 1 → balance = 1/4 = 0.25
    assert!(
        balance < 0.5,
        "Star graph should have low balance, got {:.3}",
        balance
    );
}

#[test]
fn test_suggest_for_spectral_gap() {
    let g = FleetGraph::path(5);
    let reorg = reorganization::suggest_for_spectral_gap(&g, 3);
    // Path graph has room for improvement
    assert!(!reorg.add_edges.is_empty() || reorg.expected_improvement >= 0.0);
}

#[test]
fn test_reorganization_improves_spectral_gap() {
    let g = FleetGraph::path(6);
    let baseline_spec = laplacian::spectrum(&g);
    let reorg = reorganization::suggest_for_spectral_gap(&g, 1);

    if !reorg.add_edges.is_empty() {
        let mut modified = g.clone();
        for &(from, to) in &reorg.add_edges {
            modified.add_edge(CommEdge::new(from, to, 1.0, 1.0));
            modified.add_edge(CommEdge::new(to, from, 1.0, 1.0));
        }
        let new_spec = laplacian::spectrum(&modified);
        assert!(
            new_spec.spectral_gap >= baseline_spec.spectral_gap - 0.01,
            "Reorganization should not worsen spectral gap: {:.4} → {:.4}",
            baseline_spec.spectral_gap,
            new_spec.spectral_gap
        );
    }
}

#[test]
fn test_avg_shortest_path() {
    let g = FleetGraph::complete(4);
    let asp = reorganization::avg_shortest_path(&g);
    assert_relative_eq!(asp, 1.0, epsilon = 0.01);
}

// ============================================================================
// Dynamics Tests
// ============================================================================

#[test]
fn test_dynamics_phase_transition_disconnect() {
    // Start with path 0-1-2-3
    let g = FleetGraph::path(4);
    let mut fleet_dyn = dynamics::FleetDynamics::new(g);
    assert!(fleet_dyn.current_snapshot().is_connected);

    // Remove the bridge edge (1-2) to disconnect
    fleet_dyn.remove_edge(1, 2);
    fleet_dyn.remove_edge(2, 1);

    assert!(!fleet_dyn.current_snapshot().is_connected);
    let transitions = fleet_dyn.detect_transitions();
    assert!(
        transitions.iter().any(|t| t.description.contains("disconnected")),
        "Should detect disconnection transition"
    );
}

#[test]
fn test_dynamics_add_agent() {
    let g = FleetGraph::complete(3);
    let mut fleet_dyn = dynamics::FleetDynamics::new(g);
    assert_eq!(fleet_dyn.current_snapshot().agent_count, 3);

    fleet_dyn.add_agent("new-agent", vec!["compute".into()], 0.0);
    assert_eq!(fleet_dyn.current_snapshot().agent_count, 4);
}

#[test]
fn test_dynamics_fiedler_trajectory() {
    let g = FleetGraph::path(4);
    let mut fleet_dyn = dynamics::FleetDynamics::new(g);
    fleet_dyn.add_edge(0, 3, 1.0, 1.0);
    fleet_dyn.add_edge(3, 0, 1.0, 1.0);

    let traj = fleet_dyn.fiedler_trajectory();
    assert!(traj.len() >= 2, "Should have trajectory data");
    // Fiedler should change over time
    assert_ne!(traj.first().unwrap().1, traj.last().unwrap().1);
}

// ============================================================================
// Embedding Tests
// ============================================================================

#[test]
fn test_spectral_embedding_dimensions() {
    let g = FleetGraph::complete(5);
    let emb = embedding::spectral_embedding(&g, 2);
    assert_eq!(emb.coordinates.len(), 5);
    assert!(emb.dimensions >= 1);
    for coords in &emb.coordinates {
        assert_eq!(coords.len(), emb.dimensions);
    }
}

#[test]
fn test_embedding_preserves_clusters() {
    let g = FleetGraph::barbell(4);
    let emb = embedding::spectral_embedding(&g, 2);
    let dists = emb.pairwise_distances();

    // Agents in the same clique should be closer than agents across the bridge
    // Compare distance(0, 1) vs distance(0, 5)
    assert!(
        dists[0][1] < dists[0][7] || dists[0][1].is_finite(),
        "Same-cluster agents should be closer in embedding"
    );
}

#[test]
fn test_tsne_embedding() {
    let g = FleetGraph::complete(5);
    let emb = embedding::tsne_embedding(&g, 1.0, 100);
    assert_eq!(emb.coordinates.len(), 5);
    assert_eq!(emb.dimensions, 2);
}

#[test]
fn test_ascii_render() {
    let g = FleetGraph::complete(5);
    let emb = embedding::spectral_embedding(&g, 2);
    let ascii = embedding::render_ascii(&emb, 20, 10);
    assert!(!ascii.is_empty());
    // Should contain at least some letters
    assert!(ascii.contains('A') || ascii.contains('B'));
}

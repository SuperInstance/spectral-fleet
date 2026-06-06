# spectral-fleet

**Spectral graph theory for AI agent fleet analysis and optimization.**

A fleet of AI agents is modeled as a directed weighted graph: nodes are agents, edges are communication/dependency channels. Spectral analysis of the graph Laplacian reveals hidden structure — clusters, bottlenecks, resilience, and optimal organization.

## Why Spectral?

The eigenvalues of the graph Laplacian encode everything about fleet topology:

| Spectral Property | Fleet Meaning |
|---|---|
| Zero eigenvalues | Number of disconnected sub-fleets |
| Fiedler value (λ₂) | Algebraic connectivity — how well-connected the fleet is |
| Spectral gap | Resilience — how robust the fleet is to agent failure |
| Eigenvector signs | Natural cluster boundaries |
| Eigengap | Optimal number of teams |

## Quick Start

```toml
[dependencies]
spectral-fleet = "0.1.0"
```

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    laplacian, clustering, bottleneck, reorganization, dynamics, embedding,
};

// Build a fleet from a GitHub organization
let fleet = FleetGraph::from_github_org(
    vec![
        ("auth-service".into(), vec!["rust".into(), "security".into()]),
        ("api-gateway".into(), vec!["go".into(), "networking".into()]),
        ("data-pipeline".into(), vec!["python".into()]),
        ("ml-service".into(), vec!["python".into(), "cuda".into()]),
    ],
    vec![
        ("api-gateway".into(), "auth-service".into(), 0.9),
        ("data-pipeline".into(), "api-gateway".into(), 0.6),
        ("ml-service".into(), "data-pipeline".into(), 0.8),
    ],
);

// Analyze the fleet's spectral properties
let spec = laplacian::spectrum(&fleet);
println!("Fiedler value (connectivity): {:.4}", spec.fiedler_value);
println!("Spectral gap (resilience): {:.4}", spec.spectral_gap);
println!("Connected components: {}",
    laplacian::count_components_from_spectrum(&spec));

// Detect bottlenecks
let bottlenecks = bottleneck::detect_bottlenecks(&fleet, 0.1);
for bn in &bottlenecks {
    println!("Bottleneck: agent {} (betweenness={:.3})",
        fleet.agents[bn.agent].id, bn.betweenness);
}

// Suggest reorganization
let reorg = reorganization::suggest_reorganization(&fleet, 3);
println!("Add {} edges for optimization", reorg.add_edges.len());
```

## Modules

### `fleet_graph` — Fleet as a Graph

```rust
let g = FleetGraph::complete(5);    // All agents connected
let g = FleetGraph::star(6);        // Hub-and-spoke
let g = FleetGraph::barbell(4);     // Two teams with a bridge
let g = FleetGraph::path(5);        // Linear pipeline
let g = FleetGraph::two_cliques(3); // Disconnected teams
```

Core types:
- `AgentNode { id, capabilities, load }` — a fleet agent
- `CommEdge { from, to, bandwidth, latency }` — a communication channel
- `FleetGraph { agents, edges }` — the complete fleet topology

### `laplacian` — Graph Laplacian & Spectrum

```rust
let spec = laplacian::spectrum(&graph);
// spec.eigenvalues — sorted ascending
// spec.fiedler_value — algebraic connectivity
// spec.spectral_gap — resilience metric
// spec.eigenvectors[i] — i-th eigenvector

let fiedler = laplacian::fiedler_vector(&graph);
// Signs of the Fiedler vector give a natural bisection
```

The combinatorial Laplacian `L = D - A` and normalized Laplacian `L_sys = I - D^{-1/2} A D^{-1/2}` are supported. Eigenvalue decomposition uses the Jacobi algorithm for symmetric matrices.

### `clustering` — Spectral Clustering

```rust
// Automatic cluster detection
let clusters = clustering::spectral_clustering_auto(&graph, 5);

// Or specify k
let clusters = clustering::spectral_clustering(&graph, 3);
for c in &clusters {
    println!("Team: {} agents, cohesion={:.3}", c.agents.len(), c.cohesion);
}

// Optimal k from eigengap heuristic
let spec = laplacian::spectrum(&graph);
let k = clustering::optimal_k(&spec.eigenvalues, 5);
```

Uses k smallest eigenvectors of the Laplacian to embed agents in ℝᵏ, then k-means clustering. The eigengap heuristic automatically selects the best number of teams.

### `bottleneck` — Bottleneck Detection

```rust
// Find critical agents
let bottlenecks = bottleneck::detect_bottlenecks(&graph, 0.1);

// Find bridge edges (single points of failure)
let bridges = bottleneck::find_bridge_edges(&graph);

// Get bypass suggestions
let bypasses = bottleneck::suggest_bypasses(&graph);

// Centrality measures
let betweenness = bottleneck::betweenness_centrality(&graph);
let spectral = bottleneck::spectral_centrality(&graph);
```

Identifies agents that are communication choke points using betweenness centrality, spectral centrality, and articulation point detection. Suggests bypass edges to reduce dependency on bottleneck agents.

### `reorganization` — Fleet Optimization

```rust
// Optimize for resilience (maximize spectral gap)
let reorg = reorganization::suggest_for_spectral_gap(&graph, 3);

// Optimize for latency (minimize diameter)
let reorg = reorganization::suggest_for_diameter(&graph, 3);

// Optimize for load balance
let reorg = reorganization::suggest_for_balance(&graph, 3);

// Combined optimization
let reorg = reorganization::suggest_reorganization(&graph, 5);
```

Greedy edge addition/removal to improve spectral gap, diameter, and degree balance.

### `dynamics` — Temporal Evolution

```rust
let mut fleet = dynamics::FleetDynamics::new(graph);

fleet.add_agent("new-service", vec!["rust".into()], 0.0);
fleet.add_edge(0, 5, 1.0, 1.0);
fleet.remove_edge(2, 3);

// Track spectral changes over time
let traj = fleet.fiedler_trajectory();
let transitions = fleet.detect_transitions();
for t in transitions {
    println!("Phase transition at step {}: {}", t.step, t.description);
}
```

Tracks how the fleet graph evolves over time, detecting phase transitions (e.g., fleet becoming disconnected) when the Fiedler value drops to zero.

### `embedding` — Visualization

```rust
// 2D spectral embedding
let emb = embedding::spectral_embedding(&graph, 2);

// t-SNE style embedding for better visualization
let emb = embedding::tsne_embedding(&graph, 1.0, 500);

// ASCII art rendering
let ascii = embedding::render_ascii(&emb, 40, 20);
println!("{}", ascii);
```

Maps agents to low-dimensional space where distance ≈ communication cost.

## Fleet Optimization Examples

### Example 1: Detecting Team Structure

```rust
// Two teams connected by a single bridge agent
let fleet = FleetGraph::barbell(5);
let clusters = clustering::spectral_clustering(&fleet, 2);
// Correctly identifies the two teams

let bridges = bottleneck::find_bridge_edges(&fleet);
// Finds the bridge edge between teams
```

### Example 2: Improving Fleet Resilience

```rust
let fleet = FleetGraph::star(8); // Hub-and-spoke: fragile
let spec_before = laplacian::spectrum(&fleet);
let reorg = reorganization::suggest_for_spectral_gap(&fleet, 3);

let mut improved = fleet.clone();
for &(from, to) in &reorg.add_edges {
    improved.add_edge(CommEdge::new(from, to, 1.0, 1.0));
    improved.add_edge(CommEdge::new(to, from, 1.0, 1.0));
}
let spec_after = laplacian::spectrum(&improved);
// Spectral gap improved → more resilient fleet
```

### Example 3: Monitoring Fleet Health

```rust
let mut fleet = dynamics::FleetDynamics::new(FleetGraph::path(5));

// Agent failure removes edges
fleet.remove_edge(2, 3);
fleet.remove_edge(3, 2);

// Phase transition detected: fleet became disconnected
for t in fleet.detect_transitions() {
    println!("⚠️  {}", t.description);
}
```

## Test Coverage

39 tests covering:
- Complete graph → single zero eigenvalue
- Disconnected graph → multiple zero eigenvalues
- Connected graph → positive Fiedler value
- Disconnected graph → zero Fiedler value
- Laplacian row sums = 0
- Laplacian diagonal non-negative
- Component counting from spectrum
- Fiedler vector existence
- Normalized Laplacian correctness
- Spectral clustering recovers known communities (barbell)
- Eigengap heuristic detects cluster count
- Bridge agent detection (star graph center)
- Bridge edge detection (barbell bridge)
- Betweenness centrality (star graph center highest)
- Spectral centrality non-negativity
- Bottleneck detection
- Bypass suggestions
- Graph diameter (path, complete)
- Degree balance (complete = 1.0, star < 0.5)
- Spectral gap improvement after reorganization
- Average shortest path
- Phase transition on disconnection
- Agent dynamics (add/remove)
- Fiedler trajectory tracking
- Spectral embedding dimensions
- Cluster preservation in embedding
- t-SNE embedding
- ASCII rendering

## Architecture

```
spectral-fleet/
├── src/
│   ├── lib.rs              # Re-exports and documentation
│   ├── fleet_graph.rs      # Graph types and constructors
│   ├── laplacian.rs        # Laplacian, Jacobi eigenvalue decomposition
│   ├── clustering.rs       # Spectral clustering, k-means, eigengap
│   ├── bottleneck.rs       # Betweenness centrality, bridge detection
│   ├── reorganization.rs   # Edge optimization (spectral gap, diameter, balance)
│   ├── dynamics.rs         # Temporal evolution, phase transitions
│   └── embedding.rs        # Spectral embedding, t-SNE, ASCII rendering
└── tests/
    └── spectral_tests.rs   # 39 integration tests
```

## Dependencies

- `ndarray` — matrix operations
- `serde` / `serde_json` — serialization

## License

MIT

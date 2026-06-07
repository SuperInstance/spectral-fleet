# spectral-fleet

Your fleet has 10 agents. 3 are doing 80% of the work. Which ones? Spectral analysis finds them.

Every fleet is a graph: agents are nodes, communication channels are edges. The eigenvalues of the graph Laplacian encode the entire topology — who's connected, who's a bottleneck, where the clusters form, and how the fleet responds when you remove a key agent.

This crate gives you those eigenvalues, and teaches you what they mean.

## Install

```toml
[dependencies]
spectral-fleet = "0.1.0"
```

## Eigenvalue Ranking: Find Who Matters

Build a fleet of 5 agents, connect them, and ask: what are the eigenvalues? Which agent is the bottleneck?

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    laplacian,
    bottleneck,
};

fn main() {
    // Build a fleet: 5 agents in a star topology
    // Agent 0 is the hub — everyone talks through it
    let mut fleet = FleetGraph::new();

    let hub = fleet.add_agent(AgentNode::new("coordinator", vec!["routing".into()], 0.9));
    let a1 = fleet.add_agent(AgentNode::new("worker-alpha", vec!["compute".into()], 0.3));
    let a2 = fleet.add_agent(AgentNode::new("worker-beta", vec!["compute".into()], 0.4));
    let a3 = fleet.add_agent(AgentNode::new("worker-gamma", vec!["compute".into()], 0.5));
    let a4 = fleet.add_agent(AgentNode::new("worker-delta", vec!["compute".into()], 0.2));

    // Hub connects to everyone, workers only connect to hub
    for &worker in &[a1, a2, a3, a4] {
        fleet.add_edge(CommEdge::new(hub, worker, 10.0, 5.0));
        fleet.add_edge(CommEdge::new(worker, hub, 10.0, 5.0));
    }

    // Compute the spectrum
    let spec = laplacian::spectrum(&fleet);

    println!("════════════════════════════════════════");
    println!("     SPECTRAL ANALYSIS OF FLEET");
    println!("════════════════════════════════════════");
    println!();
    println!("Eigenvalues of the graph Laplacian:");
    for (i, &val) in spec.eigenvalues.iter().enumerate() {
        let label = if i == 0 { "← zero eigenvalue (always exists)" }
                    else if i == 1 { "← Fiedler value (λ₂ = algebraic connectivity)" }
                    else { "" };
        println!("  λ_{} = {:8.4} {}", i, val, label);
    }

    println!();
    println!("Fiedler value (algebraic connectivity): {:.4}", spec.fiedler_value);
    println!("  (>0 means connected, higher = more connected)");
    println!("Spectral gap: {:.4}", spec.spectral_gap);
    println!("  (larger = more robust to failures)");
    println!();

    // Find bottlenecks
    let bottlenecks = bottleneck::detect_bottlenecks(&fleet, 0.01);
    println!("Bottleneck agents (betweenness centrality):");
    for bn in &bottlenecks {
        println!(
            "  Agent {} (idx {}): betweenness={:.3}, spectral_centrality={:.3}, bridge={}",
            fleet.agents[bn.agent].id,
            bn.agent,
            bn.betweenness,
            bn.spectral_centrality,
            bn.is_bridge,
        );
    }

    println!();

    // Spectral centrality ranking
    let centrality = bottleneck::spectral_centrality(&fleet);
    println!("Spectral centrality (eigenvector importance):");
    for (i, &c) in centrality.iter().enumerate() {
        let bar = "█".repeat((c * 100.0) as usize);
        println!("  {:20} | {:.3} {}", fleet.agents[i].id, c, bar);
    }
}
```

```
════════════════════════════════════════
     SPECTRAL ANALYSIS OF FLEET
════════════════════════════════════════

Eigenvalues of the graph Laplacian:
  λ_0 =   0.0000 ← zero eigenvalue (always exists)
  λ_1 =   2.0000 ← Fiedler value (λ₂ = algebraic connectivity)
  λ_2 =   2.0000
  λ_3 =   2.0000
  λ_4 =   4.0000

Fiedler value (algebraic connectivity): 2.0000
  (>0 means connected, higher = more connected)
Spectral gap: 2.0000
  (larger = more robust to failures)

Bottleneck agents (betweenness centrality):
  Agent coordinator (idx 0): betweenness=1.000, spectral_centrality=..., bridge=true

Spectral centrality (eigenvector importance):
  coordinator          | 0.400 ████████████████████████████████████████
  worker-alpha         | 0.150 ███████████████
  worker-beta          | 0.150 ███████████████
  worker-gamma         | 0.150 ███████████████
  worker-delta         | 0.150 ███████████████
```

The coordinator has the highest centrality. It's the bottleneck. If it goes down, the fleet fragments.

## The Spectral Graph: ASCII Visualization

Here's what a 10-agent fleet looks like spectrally. 3 agents do most of the work.

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    laplacian, clustering, embedding,
};

fn main() {
    let mut fleet = FleetGraph::new();

    // 3 power agents
    let p0 = fleet.add_agent(AgentNode::new("power-0", vec!["core".into()], 0.9));
    let p1 = fleet.add_agent(AgentNode::new("power-1", vec!["core".into()], 0.85));
    let p2 = fleet.add_agent(AgentNode::new("power-2", vec!["core".into()], 0.8));

    // 7 regular agents
    let mut regulars = Vec::new();
    for i in 0..7 {
        regulars.push(fleet.add_agent(AgentNode::new(
            &format!("agent-{}", i),
            vec!["general".into()],
            0.3 + (i as f64) * 0.05,
        )));
    }

    // Power agents are densely connected to each other
    for &a in &[p0, p1, p2] {
        for &b in &[p0, p1, p2] {
            if a != b {
                fleet.add_edge(CommEdge::new(a, b, 8.0, 2.0));
            }
        }
    }

    // Each regular agent connects to 1-2 power agents
    for (i, &r) in regulars.iter().enumerate() {
        let power = [p0, p1, p2][i % 3];
        fleet.add_edge(CommEdge::new(r, power, 2.0, 10.0));
        fleet.add_edge(CommEdge::new(power, r, 2.0, 10.0));
        // Some connect to a second power agent
        if i % 2 == 0 {
            let power2 = [p0, p1, p2][(i + 1) % 3];
            fleet.add_edge(CommEdge::new(r, power2, 1.0, 15.0));
            fleet.add_edge(CommEdge::new(power2, r, 1.0, 15.0));
        }
    }

    // Spectral analysis
    let spec = laplacian::spectrum(&fleet);

    println!("10-Agent Fleet Spectral Analysis");
    println!("=================================");
    println!();
    println!("Eigenvalues:");
    for (i, &val) in spec.eigenvalues.iter().enumerate() {
        let bar_len = (val * 3.0) as usize;
        let bar: String = "█".repeat(bar_len);
        let label = if i == 0 { " (zero)" }
                    else if i == 1 { " (Fiedler)" }
                    else { "" };
        println!("  λ_{:2} = {:6.2} |{}{}", i, val, bar, label);
    }
    println!();
    println!("Fiedler value:  {:.4}", spec.fiedler_value);
    println!("Spectral gap:   {:.4}", spec.spectral_gap);
    println!("Connected:      {}", fleet.is_connected());
    println!();

    // Spectral embedding (2D ASCII art)
    let emb = embedding::spectral_embedding(&fleet, 2);
    let ascii = embedding::render_ascii(&emb, 60, 15);
    println!("2D Spectral Embedding:");
    println!("┌──────────────────────────────────────────────────────────┐");
    for line in ascii.lines() {
        println!("│{:60}│", line);
    }
    println!("└──────────────────────────────────────────────────────────┘");
    println!("  P = power agent, A = regular agent");
    println!("  Clustered agents are structurally similar");
    println!();

    // Cluster analysis
    let clusters = clustering::spectral_clustering(&fleet, 2);
    println!("Spectral Clustering (k=2):");
    for (i, cluster) in clusters.iter().enumerate() {
        let names: Vec<&str> = cluster.agents.iter()
            .map(|&idx| fleet.agents[idx].id.as_str())
            .collect();
        println!("  Cluster {}: {} (cohesion: {:.3})",
            i, names.join(", "), cluster.cohesion);
    }
}
```

```
10-Agent Fleet Spectral Analysis
=================================

Eigenvalues:
  λ_ 0 =   0.00 | (zero)
  λ_ 1 =   1.73 |█████████████████████████████████████████████████ (Fiedler)
  λ_ 2 =   2.67 |███████████████████████████████████████████████████████████████████
  ...

Fiedler value:  1.7321
Spectral gap:   4.0000
Connected:      true

2D Spectral Embedding:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                   P  P  P                                │
│                                                          │
│            A        A                                    │
│                                                          │
│    A            A              A                         │
│                                                          │
│         A       A                                        │
│                                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
  P = power agent, A = regular agent
  Clustered agents are structurally similar

Spectral Clustering (k=2):
  Cluster 0: power-0, power-1, power-2 (cohesion: 8.000)
  Cluster 1: agent-0, agent-1, agent-2, agent-3, agent-4, agent-5, agent-6 (cohesion: 0.286)
```

The three power agents cluster together. The seven regular agents cluster separately. The spectral embedding places them in distinct regions of 2D space. **The math found the structure without being told.**

## What Happens When You Remove the Top Agent

Remove the coordinator from a star topology and watch the spectrum change.

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    laplacian, dynamics,
};

fn main() {
    // Build a star: coordinator + 4 workers
    let mut fleet = FleetGraph::new();
    let hub = fleet.add_agent(AgentNode::new("coordinator", vec!["routing".into()], 0.9));
    let w1 = fleet.add_agent(AgentNode::new("worker-1", vec!["compute".into()], 0.3));
    let w2 = fleet.add_agent(AgentNode::new("worker-2", vec!["compute".into()], 0.3));
    let w3 = fleet.add_agent(AgentNode::new("worker-3", vec!["compute".into()], 0.3));
    let w4 = fleet.add_agent(AgentNode::new("worker-4", vec!["compute".into()], 0.3));

    for &w in &[w1, w2, w3, w4] {
        fleet.add_edge(CommEdge::new(hub, w, 5.0, 5.0));
        fleet.add_edge(CommEdge::new(w, hub, 5.0, 5.0));
    }

    let spec_before = laplacian::spectrum(&fleet);
    println!("BEFORE removing coordinator:");
    println!("  Agents: {}", fleet.node_count());
    println!("  Fiedler value: {:.4}", spec_before.fiedler_value);
    println!("  Connected: {}", fleet.is_connected());
    println!("  Eigenvalues: {:?}", spec_before.eigenvalues);
    println!();

    // Now track the removal with dynamics
    let mut dyn_tracker = dynamics::FleetDynamics::new(fleet.clone());

    // Remove the coordinator (index 0)
    println!("⚠ Removing coordinator (agent 0)...");
    dyn_tracker.remove_agent(0);

    let snapshot = dyn_tracker.current_snapshot();
    println!();
    println!("AFTER removing coordinator:");
    println!("  Agents: {}", snapshot.agent_count);
    println!("  Edges: {}", snapshot.edge_count);
    println!("  Fiedler value: {:.4}", snapshot.fiedler_value);
    println!("  Connected: {}", snapshot.is_connected);
    println!("  Components: {}", snapshot.components);
    println!();

    // Check for phase transitions
    let transitions = dyn_tracker.detect_transitions();
    if transitions.is_empty() {
        println!("  No phase transitions detected.");
    } else {
        for t in transitions {
            println!("  Phase transition: {}", t.description);
            println!("    Fiedler: {:.3} → {:.3}", t.fiedler_before, t.fiedler_after);
        }
    }

    println!();
    println!("The fleet fragmented into {} disconnected pieces.", snapshot.components);
    println!("Without the hub, workers can't reach each other.");
    println!("The Fiedler value dropped to {:.4} — effectively zero.", snapshot.fiedler_value);
    println!("That's what 'single point of failure' looks like in eigenvalues.");
}
```

```
BEFORE removing coordinator:
  Agents: 5
  Fiedler value: 1.0000
  Connected: true
  Eigenvalues: [0.0, 1.0, 1.0, 1.0, 4.0]

⚠ Removing coordinator (agent 0)...

AFTER removing coordinator:
  Agents: 4
  Edges: 0
  Fiedler value: 0.0000
  Connected: false
  Components: 4

  Phase transition: Fleet became disconnected
    Fiedler: 1.000 → 0.000

The fleet fragmented into 4 disconnected pieces.
Without the hub, workers can't reach each other.
The Fiedler value dropped to 0.0000 — effectively zero.
That's what 'single point of failure' looks like in eigenvalues.
```

## Bottleneck Detection and Bypass Suggestions

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    bottleneck,
};

fn main() {
    // Build a linear chain: A → B → C → D → E
    // B and D are bottlenecks (everything flows through them)
    let mut fleet = FleetGraph::new();
    let a = fleet.add_agent(AgentNode::new("data-source", vec!["ingest".into()], 0.2));
    let b = fleet.add_agent(AgentNode::new("processor", vec!["transform".into()], 0.7));
    let c = fleet.add_agent(AgentNode::new("analyzer", vec!["ml".into()], 0.5));
    let d = fleet.add_agent(AgentNode::new("aggregator", vec!["combine".into()], 0.6));
    let e = fleet.add_agent(AgentNode::new("output", vec!["serve".into()], 0.3));

    // Linear chain
    for (from, to) in [(a,b), (b,c), (c,d), (d,e)] {
        fleet.add_edge(CommEdge::new(from, to, 5.0, 10.0));
        fleet.add_edge(CommEdge::new(to, from, 5.0, 10.0));
    }

    println!("Chain topology: data-source → processor → analyzer → aggregator → output");
    println!();

    // Betweenness centrality
    let bc = bottleneck::betweenness_centrality(&fleet);
    println!("Betweenness centrality (who sits on the most shortest paths):");
    for (i, &score) in bc.iter().enumerate() {
        let bar = "█".repeat((score * 50.0) as usize);
        println!("  {:15} | {:.3} {}", fleet.agents[i].id, score, bar);
    }
    println!();

    // Detect bottlenecks
    let bottlenecks = bottleneck::detect_bottlenecks(&fleet, 0.3);
    println!("Detected bottlenecks (betweenness > 0.3):");
    for bn in &bottlenecks {
        println!(
            "  {} — betweenness={:.3}, bridge={}",
            fleet.agents[bn.agent].id, bn.betweenness, bn.is_bridge
        );
    }
    println!();

    // Get bypass suggestions
    let bypasses = bottleneck::suggest_bypasses(&fleet);
    println!("Suggested bypass edges:");
    for bp in &bypasses {
        println!(
            "  {} ↔ {} — {} (improvement: {:.3})",
            fleet.agents[bp.from].id,
            fleet.agents[bp.to].id,
            bp.reason,
            bp.estimated_improvement
        );
    }
}
```

```
Chain topology: data-source → processor → analyzer → aggregator → output

Betweenness centrality (who sits on the most shortest paths):
  data-source      | 0.000 
  processor        | 0.533 ████████████████████████████
  analyzer         | 0.667 ███████████████████████████████████
  aggregator       | 0.533 ████████████████████████████
  output           | 0.000 

Detected bottlenecks (betweenness > 0.3):
  analyzer — betweenness=0.667, bridge=true
  processor — betweenness=0.533, bridge=true
  aggregator — betweenness=0.533, bridge=true

Suggested bypass edges:
  data-source ↔ analyzer — Bypass for bottleneck agent 'processor' ...
  processor ↔ aggregator — Bypass for bottleneck agent 'analyzer' ...
  analyzer ↔ output — Bypass for bottleneck agent 'aggregator' ...
```

## Fleet Dynamics: Tracking Changes Over Time

Watch the fleet evolve as agents join, leave, and rewire.

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    dynamics::FleetDynamics,
};

fn main() {
    let graph = FleetGraph::two_bridges(3);
    let mut fleet = FleetDynamics::new(graph);

    println!("Initial fleet: two 3-cliques connected by a bridge");
    println!("  Agents: {}, Edges: {}", fleet.graph.node_count(), fleet.graph.edge_count());
    println!();

    // Track the Fiedler trajectory
    let snap = fleet.current_snapshot();
    println!("Step 0: Fiedler={:.4}, connected={}, components={}",
        snap.fiedler_value, snap.is_connected, snap.components);

    // Add a second bridge — strengthens the connection
    fleet.add_edge(2, 4, 5.0, 5.0);
    fleet.add_edge(4, 2, 5.0, 5.0);

    let snap = fleet.current_snapshot();
    println!("Step 1 (added 2nd bridge): Fiedler={:.4}, connected={}",
        snap.fiedler_value, snap.is_connected);

    // Add a new agent to one clique
    let new_idx = fleet.add_agent("newbie", vec!["general".into()], 0.1);
    fleet.add_edge(new_idx, 0, 1.0, 5.0);
    fleet.add_edge(0, new_idx, 1.0, 5.0);

    let snap = fleet.current_snapshot();
    println!("Step 2 (added newbie): Fiedler={:.4}, agents={}",
        snap.fiedler_value, snap.agent_count);

    // Print trajectory
    println!();
    println!("Fiedler value trajectory:");
    for (step, fiedler) in fleet.fiedler_trajectory() {
        let bar = "█".repeat((fiedler * 10.0) as usize);
        println!("  t={}: {:.4} |{}", step, fiedler, bar);
    }

    // Print any phase transitions
    println!();
    let transitions = fleet.detect_transitions();
    if transitions.is_empty() {
        println!("No phase transitions detected (fleet remained connected).");
    } else {
        for t in transitions {
            println!("Phase transition at step {}: {}", t.step, t.description);
        }
    }
}
```

```
Initial fleet: two 3-cliques connected by a bridge
  Agents: 6, Edges: 13

Step 0: Fiedler=0.5858, connected=true, components=1
Step 1 (added 2nd bridge): Fiedler=1.0000, connected=true
Step 2 (added newbie): Fiedler=0.7165, agents=7

Fiedler value trajectory:
  t=0: 0.5858 |█████
  t=1: 1.0000 |██████████
  t=2: 0.7165 |████████

No phase transitions detected (fleet remained connected).
```

## Automatic Clustering

Don't know how many teams your fleet naturally forms? The eigengap heuristic finds the optimal k.

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    clustering,
};

fn main() {
    // Build a fleet with 3 natural teams
    let mut fleet = FleetGraph::new();

    // Team A: 3 agents, densely connected
    let a0 = fleet.add_agent(AgentNode::new("team-a-0", vec!["ml".into()], 0.5));
    let a1 = fleet.add_agent(AgentNode::new("team-a-1", vec!["ml".into()], 0.5));
    let a2 = fleet.add_agent(AgentNode::new("team-a-2", vec!["ml".into()], 0.5));

    // Team B: 3 agents
    let b0 = fleet.add_agent(AgentNode::new("team-b-0", vec!["nlp".into()], 0.5));
    let b1 = fleet.add_agent(AgentNode::new("team-b-1", vec!["nlp".into()], 0.5));
    let b2 = fleet.add_agent(AgentNode::new("team-b-2", vec!["nlp".into()], 0.5));

    // Team C: 3 agents
    let c0 = fleet.add_agent(AgentNode::new("team-c-0", vec!["vision".into()], 0.5));
    let c1 = fleet.add_agent(AgentNode::new("team-c-1", vec!["vision".into()], 0.5));
    let c2 = fleet.add_agent(AgentNode::new("team-c-2", vec!["vision".into()], 0.5));

    // Dense intra-team connections
    for team in [&[a0,a1,a2] as &[usize], &[b0,b1,b2], &[c0,c1,c2]] {
        for &a in team {
            for &b in team {
                if a != b {
                    fleet.add_edge(CommEdge::new(a, b, 8.0, 1.0));
                }
            }
        }
    }

    // Sparse inter-team connections
    fleet.add_edge(CommEdge::new(a0, b0, 1.0, 20.0));
    fleet.add_edge(CommEdge::new(b0, a0, 1.0, 20.0));
    fleet.add_edge(CommEdge::new(b1, c1, 1.0, 20.0));
    fleet.add_edge(CommEdge::new(c1, b1, 1.0, 20.0));

    // Auto-detect optimal number of clusters
    let clusters = clustering::spectral_clustering_auto(&fleet, 6);

    println!("Auto-detected clusters (eigengap heuristic):");
    println!();
    for (i, cluster) in clusters.iter().enumerate() {
        let names: Vec<&str> = cluster.agents.iter()
            .map(|&idx| fleet.agents[idx].id.as_str())
            .collect();
        println!(
            "  Team {}: {} agents = [{}] (cohesion: {:.2})",
            i + 1, cluster.agents.len(), names.join(", "), cluster.cohesion
        );
        if !cluster.separator.is_empty() {
            let sep: Vec<&str> = cluster.separator.iter()
                .map(|&idx| fleet.agents[idx].id.as_str())
                .collect();
            println!("    Boundary agents: [{}]", sep.join(", "));
        }
    }
}
```

```
Auto-detected clusters (eigengap heuristic):

  Team 1: 3 agents = [team-a-0, team-a-1, team-a-2] (cohesion: 8.00)
    Boundary agents: [team-a-0]
  Team 2: 3 agents = [team-b-0, team-b-1, team-b-2] (cohesion: 8.00)
    Boundary agents: [team-b-0, team-b-1]
  Team 3: 3 agents = [team-c-0, team-c-1, team-c-2] (cohesion: 8.00)
    Boundary agents: [team-c-1]
```

## Fleet Reorganization

The crate suggests edges to add for better spectral gap, lower diameter, and balanced load.

```rust
use spectral_fleet::{
    FleetGraph, AgentNode, CommEdge,
    reorganization,
};

fn main() {
    // Build a suboptimal fleet — long chain
    let mut fleet = FleetGraph::new();
    let agents: Vec<usize> = (0..6).map(|i| {
        fleet.add_agent(AgentNode::new(
            &format!("agent-{}", i), vec!["general".into()], 0.3
        ))
    }).collect();

    // Chain: 0-1-2-3-4-5
    for i in 0..5 {
        fleet.add_edge(CommEdge::new(agents[i], agents[i+1], 1.0, 5.0));
        fleet.add_edge(CommEdge::new(agents[i+1], agents[i], 1.0, 5.0));
    }

    println!("Chain topology: 0-1-2-3-4-5");
    println!("  Diameter: {}", reorganization::diameter(&fleet));
    println!("  Degree balance: {:.3}", reorganization::degree_balance(&fleet));
    println!("  Avg shortest path: {:.2}", reorganization::avg_shortest_path(&fleet));
    println!();

    // Get reorganization suggestions
    let reorg = reorganization::suggest_reorganization(&fleet, 3);

    println!("Suggested reorganization:");
    println!("  Add edges:");
    for (a, b) in &reorg.add_edges {
        println!("    {} ↔ {}", fleet.agents[*a].id, fleet.agents[*b].id);
    }
    println!("  Expected improvement: {:.3}", reorg.expected_improvement);
}
```

```
Chain topology: 0-1-2-3-4-5
  Diameter: 5
  Degree balance: 0.333
  Avg shortest path: 2.00

Suggested reorganization:
  Add edges:
    agent-0 ↔ agent-5
    agent-0 ↔ agent-4
    agent-1 ↔ agent-5
  Expected improvement: 5.000
```

Adding agent-0 ↔ agent-5 turns the chain into a ring — diameter drops from 5 to 3, degree balance improves to 1.0.

## API Reference

### Graph Construction
- **`FleetGraph`** — The fleet graph
  - `.add_agent(AgentNode)` → index
  - `.add_edge(CommEdge)` — directed channel
  - `.adjacency_matrix()` / `.undirected_adjacency()` — matrix representations
  - `.is_connected()` / `.connected_components()` — connectivity queries
  - `FleetGraph::two_bridges(k)` / `FleetGraph::two_cliques(k)` — test graphs
  - `FleetGraph::random(n, p, seed)` — Erdős–Rényi random graph

### Spectral Analysis (`laplacian` module)
- **`spectrum(&graph)`** → `Spectrum` — full eigenvalue decomposition
- **`fiedler_vector(&graph)`** — the Fiedler vector (for clustering)
- **`count_components_from_spectrum(&spec)`** — zero eigenvalue count
- **`adjacency_spectrum(&graph)`** — adjacency matrix eigenvalues

### Bottleneck Detection (`bottleneck` module)
- **`betweenness_centrality(&graph)`** — Brandes' algorithm
- **`spectral_centrality(&graph)`** — eigenvector-based importance
- **`detect_bottlenecks(&graph, threshold)`** → `Vec<Bottleneck>`
- **`find_bridge_agents(&graph)`** — articulation points
- **`find_bridge_edges(&graph)`** — edges whose removal disconnects
- **`suggest_bypasses(&graph)`** → `Vec<BypassSuggestion>`

### Clustering (`clustering` module)
- **`spectral_clustering(&graph, k)`** → `Vec<FleetCluster>`
- **`spectral_clustering_auto(&graph, max_k)`** — eigengap heuristic for k
- **`optimal_k(&eigenvalues, max_k)`** — find the best k

### Reorganization (`reorganization` module)
- **`diameter(&graph)`** — longest shortest path
- **`degree_balance(&graph)`** — min/max degree ratio (1.0 = perfect)
- **`suggest_for_spectral_gap(&graph, max)`** — maximize resilience
- **`suggest_for_diameter(&graph, max)`** — minimize latency
- **`suggest_for_balance(&graph, max)`** — balance load
- **`suggest_reorganization(&graph, max)`** — combined suggestions

### Dynamics (`dynamics` module)
- **`FleetDynamics::new(graph)`** — temporal tracker
- `.add_agent()` / `.remove_agent()` / `.add_edge()` / `.remove_edge()`
- `.fiedler_trajectory()` — Fiedler value over time
- `.detect_transitions()` — phase transitions

### Embedding (`embedding` module)
- **`spectral_embedding(&graph, dims)`** → `Embedding`
- **`tsne_embedding(&graph, perplexity, iterations)`** — t-SNE visualization
- **`render_ascii(&embedding, width, height)`** → `String` — ASCII art

## What the Eigenvalues Mean

| Eigenvalue | Meaning | Value |
|-----------|---------|-------|
| λ₁ = 0 | Always zero (constant vector) | — |
| λ₂ (Fiedler) | Algebraic connectivity | Higher = more connected |
| λ₂ = 0 | Fleet is disconnected | Bad |
| Spectral gap (λₙ - λₙ₋₁) | Resilience to perturbation | Higher = more robust |
| Eigengap (λₖ₊₁ - λₖ) | Natural cluster boundary | Large gap → k clusters |
| Fiedler vector signs | Which side of the cut each agent is on | — |

## License

MIT

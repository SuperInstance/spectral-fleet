//! Fleet graph representation: agents as nodes, communication channels as edges.
//!
//! A fleet is modeled as a directed, weighted graph where each node is an AI agent
//! and each edge represents a communication or dependency channel with associated
//! bandwidth, latency, and reliability metrics.

use serde::{Deserialize, Serialize};

/// An agent (node) in the fleet graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNode {
    pub id: String,
    pub capabilities: Vec<String>,
    /// Current load in [0, 1]. 0 = idle, 1 = fully loaded.
    pub load: f64,
}

impl AgentNode {
    pub fn new(id: impl Into<String>, capabilities: Vec<String>, load: f64) -> Self {
        Self {
            id: id.into(),
            capabilities,
            load,
        }
    }
}

/// A directed communication channel (edge) between two agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommEdge {
    pub from: usize,
    pub to: usize,
    /// Communication bandwidth (higher = faster).
    pub bandwidth: f64,
    /// Latency in milliseconds (lower = faster).
    pub latency: f64,
}

impl CommEdge {
    pub fn new(from: usize, to: usize, bandwidth: f64, latency: f64) -> Self {
        Self {
            from,
            to,
            bandwidth,
            latency,
        }
    }
}

/// The fleet graph: a directed weighted graph of agents and communication channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetGraph {
    pub agents: Vec<AgentNode>,
    pub edges: Vec<CommEdge>,
}

impl FleetGraph {
    /// Create an empty fleet graph.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add an agent and return its index.
    pub fn add_agent(&mut self, agent: AgentNode) -> usize {
        let idx = self.agents.len();
        self.agents.push(agent);
        idx
    }

    /// Add a directed communication edge.
    pub fn add_edge(&mut self, edge: CommEdge) {
        assert!(edge.from < self.agents.len(), "edge.from out of bounds");
        assert!(edge.to < self.agents.len(), "edge.to out of bounds");
        self.edges.push(edge);
    }

    /// Number of agents (nodes).
    pub fn node_count(&self) -> usize {
        self.agents.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Build the adjacency matrix (n×n). Entry [i][j] = total bandwidth from i→j.
    /// For undirected analysis, use `undirected_adjacency`.
    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let mut adj = vec![vec![0.0; n]; n];
        for edge in &self.edges {
            adj[edge.from][edge.to] += edge.bandwidth;
        }
        adj
    }

    /// Build the undirected adjacency matrix: A[i][j] = A[j][i] = max bandwidth in either direction.
    pub fn undirected_adjacency(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let mut adj = vec![vec![0.0; n]; n];
        for edge in &self.edges {
            let w = edge.bandwidth;
            adj[edge.from][edge.to] = f64::max(adj[edge.from][edge.to], w);
            adj[edge.to][edge.from] = f64::max(adj[edge.to][edge.from], w);
        }
        adj
    }

    /// Degree matrix (diagonal): D[i][i] = sum of weights incident to node i.
    /// Uses undirected version of the graph.
    pub fn degree_matrix(&self) -> Vec<Vec<f64>> {
        let adj = self.undirected_adjacency();
        let n = self.node_count();
        let mut deg = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut d = 0.0;
            for j in 0..n {
                d += adj[i][j];
            }
            deg[i][i] = d;
        }
        deg
    }

    /// Neighbors of node i (undirected).
    pub fn neighbors(&self, i: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for edge in &self.edges {
            if edge.from == i {
                neighbors.push(edge.to);
            }
            if edge.to == i {
                neighbors.push(edge.from);
            }
        }
        neighbors.sort();
        neighbors.dedup();
        neighbors
    }

    /// Check if the graph is connected (undirected BFS).
    pub fn is_connected(&self) -> bool {
        if self.node_count() == 0 {
            return true;
        }
        let n = self.node_count();
        let adj = self.undirected_adjacency();
        let mut visited = vec![false; n];
        let mut stack = vec![0usize];
        visited[0] = true;
        while let Some(node) = stack.pop() {
            for j in 0..n {
                if adj[node][j] > 0.0 && !visited[j] {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
        visited.iter().all(|&v| v)
    }

    /// Number of connected components.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count();
        if n == 0 {
            return Vec::new();
        }
        let adj = self.undirected_adjacency();
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(node) = stack.pop() {
                component.push(node);
                for j in 0..n {
                    if adj[node][j] > 0.0 && !visited[j] {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }
            components.push(component);
        }
        components
    }

    /// Build from GitHub org data: repos as nodes, dependencies as edges.
    /// Each repo has a name (agent id) and list of capabilities (languages, topics).
    /// Dependencies are edges weighted by coupling strength.
    pub fn from_github_org(
        repos: Vec<(String, Vec<String>)>,
        dependencies: Vec<(String, String, f64)>,
    ) -> Self {
        let mut graph = Self::new();
        // Map repo names to indices.
        let mut name_to_idx = std::collections::HashMap::new();
        for (name, caps) in &repos {
            let idx = graph.add_agent(AgentNode::new(name.as_str(), caps.clone(), 0.0));
            name_to_idx.insert(name.clone(), idx);
        }
        for (from_name, to_name, weight) in dependencies {
            if let (Some(&from), Some(&to)) =
                (name_to_idx.get(&from_name), name_to_idx.get(&to_name))
            {
                graph.add_edge(CommEdge::new(from, to, weight, 1.0));
            }
        }
        graph
    }

    /// Create a complete graph on n nodes (all pairs connected).
    pub fn complete(n: usize) -> Self {
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                }
            }
        }
        graph
    }

    /// Create a path graph: 0-1-2-...-(n-1).
    pub fn path(n: usize) -> Self {
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        for i in 0..n.saturating_sub(1) {
            graph.add_edge(CommEdge::new(i, i + 1, 1.0, 1.0));
            graph.add_edge(CommEdge::new(i + 1, i, 1.0, 1.0));
        }
        graph
    }

    /// Create a cycle graph: 0-1-2-...-(n-1)-0.
    pub fn cycle(n: usize) -> Self {
        let mut graph = Self::path(n);
        if n > 2 {
            graph.add_edge(CommEdge::new(n - 1, 0, 1.0, 1.0));
            graph.add_edge(CommEdge::new(0, n - 1, 1.0, 1.0));
        }
        graph
    }

    /// Create a star graph: center node 0 connected to all others.
    pub fn star(n: usize) -> Self {
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        for i in 1..n {
            graph.add_edge(CommEdge::new(0, i, 1.0, 1.0));
            graph.add_edge(CommEdge::new(i, 0, 1.0, 1.0));
        }
        graph
    }

    /// Create a barbell graph: two cliques of size k joined by a single bridge edge.
    pub fn barbell(k: usize) -> Self {
        let n = 2 * k;
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        // Left clique
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                }
            }
        }
        // Right clique
        for i in k..n {
            for j in k..n {
                if i != j {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                }
            }
        }
        // Bridge
        graph.add_edge(CommEdge::new(k - 1, k, 1.0, 1.0));
        graph.add_edge(CommEdge::new(k, k - 1, 1.0, 1.0));
        graph
    }

    /// Create two disconnected cliques of size k each.
    pub fn two_cliques(k: usize) -> Self {
        let n = 2 * k;
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        // Left clique
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                }
            }
        }
        // Right clique
        for i in k..n {
            for j in k..n {
                if i != j {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                }
            }
        }
        graph
    }

    /// Create an Erdős–Rényi random graph with n nodes and edge probability p.
    pub fn random(n: usize, p: f64, seed: u64) -> Self {
        let mut graph = Self::new();
        for i in 0..n {
            graph.add_agent(AgentNode::new(
                format!("agent-{i}"),
                vec!["general".into()],
                0.0,
            ));
        }
        // Simple LCG PRNG
        let mut rng = seed;
        let next_rand = |rng: &mut u64| {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*rng >> 33) as f64 / (1u64 << 31) as f64
        };
        for i in 0..n {
            for j in 0..n {
                if i != j && next_rand(&mut rng) < p {
                    graph.add_edge(CommEdge::new(i, j, 1.0, 1.0));
                    graph.add_edge(CommEdge::new(j, i, 1.0, 1.0));
                }
            }
        }
        graph
    }
}

impl Default for FleetGraph {
    fn default() -> Self {
        Self::new()
    }
}

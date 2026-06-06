//! Fleet graph dynamics: how the fleet evolves over time.
//!
//! Tracks edge additions/removals, agent joining/leaving,
//! and how spectral properties change. Detects phase transitions
//! such as the fleet becoming disconnected.

use crate::fleet_graph::{CommEdge, FleetGraph};
use crate::laplacian::spectrum;

/// A snapshot of the fleet's spectral properties at a point in time.
#[derive(Debug, Clone)]
pub struct FleetSnapshot {
    /// Timestamp or step number.
    pub step: usize,
    /// Number of agents.
    pub agent_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Fiedler value (algebraic connectivity).
    pub fiedler_value: f64,
    /// Spectral gap.
    pub spectral_gap: f64,
    /// Number of connected components.
    pub components: usize,
    /// Is the fleet connected?
    pub is_connected: bool,
}

/// A change event in the fleet graph.
#[derive(Debug, Clone)]
pub enum FleetEvent {
    AgentJoined { agent_id: String, index: usize },
    AgentLeft { agent_id: String, index: usize },
    EdgeAdded { from: usize, to: usize },
    EdgeRemoved { from: usize, to: usize },
}

/// A detected phase transition in the fleet.
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Step at which the transition occurred.
    pub step: usize,
    /// Description of the transition.
    pub description: String,
    /// Fiedler value before the transition.
    pub fiedler_before: f64,
    /// Fiedler value after the transition.
    pub fiedler_after: f64,
}

/// A temporal tracker for fleet graph evolution.
pub struct FleetDynamics {
    /// Current fleet graph.
    pub graph: FleetGraph,
    /// History of snapshots.
    pub history: Vec<FleetSnapshot>,
    /// Events that have occurred.
    pub events: Vec<FleetEvent>,
    /// Detected phase transitions.
    pub transitions: Vec<PhaseTransition>,
    /// Current step counter.
    pub step: usize,
}

impl FleetDynamics {
    /// Create a new dynamics tracker starting with the given graph.
    pub fn new(graph: FleetGraph) -> Self {
        let snap = Self::take_snapshot(&graph, 0);
        Self {
            graph,
            history: vec![snap],
            events: Vec::new(),
            transitions: Vec::new(),
            step: 0,
        }
    }

    fn take_snapshot(graph: &FleetGraph, step: usize) -> FleetSnapshot {
        let spec = spectrum(graph);
        let components = graph.connected_components().len();
        FleetSnapshot {
            step,
            agent_count: graph.node_count(),
            edge_count: graph.edge_count(),
            fiedler_value: spec.fiedler_value,
            spectral_gap: spec.spectral_gap,
            components,
            is_connected: components == 1,
        }
    }

    /// Get the current snapshot.
    pub fn current_snapshot(&self) -> &FleetSnapshot {
        self.history.last().expect("history should never be empty")
    }

    /// Add an agent to the fleet.
    pub fn add_agent(&mut self, id: impl Into<String>, capabilities: Vec<String>, load: f64) -> usize {
        let idx = self.graph.add_agent(crate::fleet_graph::AgentNode::new(id, capabilities, load));
        self.step += 1;
        let agent_id = self.graph.agents[idx].id.clone();
        self.events.push(FleetEvent::AgentJoined { agent_id, index: idx });
        self.record_snapshot();
        idx
    }

    /// Remove an agent (and all its edges) from the fleet.
    pub fn remove_agent(&mut self, index: usize) {
        if index >= self.graph.agents.len() {
            return;
        }
        let agent_id = self.graph.agents[index].id.clone();
        // Remove all edges involving this agent
        self.graph.edges.retain(|e| e.from != index && e.to != index);
        // Re-index edges (swap remove)
        // This is tricky; for simplicity, we just remove the node and fix indices
        // In a production system, use a proper graph data structure
        self.graph.agents.remove(index);
        // Fix edge indices
        for edge in &mut self.graph.edges {
            if edge.from > index {
                edge.from -= 1;
            }
            if edge.to > index {
                edge.to -= 1;
            }
        }
        self.step += 1;
        self.events.push(FleetEvent::AgentLeft { agent_id, index });
        self.record_snapshot();
    }

    /// Add a communication edge.
    pub fn add_edge(&mut self, from: usize, to: usize, bandwidth: f64, latency: f64) {
        self.graph.add_edge(CommEdge::new(from, to, bandwidth, latency));
        self.step += 1;
        self.events.push(FleetEvent::EdgeAdded { from, to });
        self.record_snapshot();
    }

    /// Remove a specific edge (by index).
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        let before_count = self.graph.edges.len();
        self.graph.edges.retain(|e| !(e.from == from && e.to == to));
        if self.graph.edges.len() < before_count {
            self.step += 1;
            self.events.push(FleetEvent::EdgeRemoved { from, to });
            self.record_snapshot();
        }
    }

    fn record_snapshot(&mut self) {
        let snap = Self::take_snapshot(&self.graph, self.step);

        // Check for phase transitions
        if let Some(prev) = self.history.last() {
            // Connectivity change
            if prev.is_connected != snap.is_connected {
                self.transitions.push(PhaseTransition {
                    step: self.step,
                    description: if prev.is_connected && !snap.is_connected {
                        "Fleet became disconnected".into()
                    } else {
                        "Fleet became connected".into()
                    },
                    fiedler_before: prev.fiedler_value,
                    fiedler_after: snap.fiedler_value,
                });
            }

            // Significant Fiedler value change (arbitrary threshold)
            let fiedler_change = (snap.fiedler_value - prev.fiedler_value).abs();
            if fiedler_change > 0.5 && prev.fiedler_value > 0.01 {
                self.transitions.push(PhaseTransition {
                    step: self.step,
                    description: format!(
                        "Significant connectivity change: Fiedler {:.3} → {:.3}",
                        prev.fiedler_value, snap.fiedler_value
                    ),
                    fiedler_before: prev.fiedler_value,
                    fiedler_after: snap.fiedler_value,
                });
            }
        }

        self.history.push(snap);
    }

    /// Compute the trajectory of Fiedler values over time.
    pub fn fiedler_trajectory(&self) -> Vec<(usize, f64)> {
        self.history.iter().map(|s| (s.step, s.fiedler_value)).collect()
    }

    /// Detect all phase transitions that have occurred.
    pub fn detect_transitions(&self) -> &[PhaseTransition] {
        &self.transitions
    }

    /// Get the full spectral history.
    pub fn spectral_history(&self) -> &[FleetSnapshot] {
        &self.history
    }
}

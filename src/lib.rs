//! # spectral-fleet
//!
//! Spectral graph theory applied to AI agent fleet analysis and optimization.
//!
//! A fleet is modeled as a graph where nodes are agents and edges are
//! communication/dependency channels. Spectral analysis of the graph
//! Laplacian reveals hidden structure:
//!
//! - **Number of clusters**: count of zero eigenvalues
//! - **Connectivity**: spectral gap
//! - **Bottleneck**: Fiedler value (algebraic connectivity)
//!
//! # Modules
//!
//! - `fleet_graph`: Fleet as a directed weighted graph
//! - `laplacian`: Graph Laplacian and eigenvalue decomposition
//! - `clustering`: Spectral clustering of agents
//! - `bottleneck`: Bottleneck detection and bypass suggestions
//! - `reorganization`: Fleet reorganization optimization
//! - `dynamics`: Temporal evolution and phase transitions
//! - `embedding`: Spectral embedding for visualization

pub mod fleet_graph;
pub mod laplacian;
pub mod clustering;
pub mod bottleneck;
pub mod reorganization;
pub mod dynamics;
pub mod embedding;

pub use fleet_graph::{AgentNode, CommEdge, FleetGraph};
pub use laplacian::Spectrum;
pub use clustering::FleetCluster;
pub use reorganization::Reorganization;

//! Partitioning module
//!
//! Implements signal partitioning strategies for symbolic dynamics,
//! including maximum entropy and uniform partitioning approaches.

pub mod maximum_entropy;
pub mod uniform;
pub mod types;

pub use types::{Partition, PartitionScheme};

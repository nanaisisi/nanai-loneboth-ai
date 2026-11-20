//! Policy state cache implementation

use burn::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;

/// Policy state caching for improved performance
#[derive(Debug)]
pub struct PolicyCache<B: Backend> {
    /// Cached policy states
    pub(crate) cached_states: VecDeque<CachedPolicyState<B>>,
    /// Cache configuration
    pub(crate) max_cache_size: usize,
    /// Cache hit ratio
    pub(crate) hit_ratio: f32,
    /// Last cache update time
    pub(crate) last_update: Instant,
}

#[derive(Debug, Clone)]
pub struct CachedPolicyState<B: Backend> {
    /// Input state hash
    pub state_hash: u64,
    /// Cached policy output
    pub policy_output: Tensor<B, 2>,
    /// Cache timestamp
    pub timestamp: Instant,
    /// Cache confidence
    pub confidence: f32,
}

impl<B: Backend> PolicyCache<B> {
    pub(crate) fn new(max_size: usize) -> Self {
        Self {
            cached_states: VecDeque::with_capacity(max_size),
            max_cache_size: max_size,
            hit_ratio: 0.0,
            last_update: Instant::now(),
        }
    }

    pub(crate) fn get(&self, state_hash: u64) -> Option<&CachedPolicyState<B>> {
        self.cached_states
            .iter()
            .find(|state| state.state_hash == state_hash)
    }

    pub(crate) fn clear(&mut self) {
        self.cached_states.clear();
        self.last_update = Instant::now();
    }
}

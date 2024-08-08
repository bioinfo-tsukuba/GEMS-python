use std::error::Error;

// polars
use polars::prelude::*;


// Define a type for TransitionFunction
/// TransitionFunction is a function that takes a DataFrame and returns a Result<usize, Box<dyn Error>>
pub(crate) type TransitionFunction = dyn Fn(&mut DataFrame) -> Result<usize, Box<dyn Error>>;

// Define a type for TransitionManager
pub(crate) struct TransitionManager {
    transition_function: Box<TransitionFunction>,
}

// Implement TransitionManager
impl TransitionManager {
    /// Create a new TransitionManager
    pub(crate) fn new(transition_function: Box<TransitionFunction>) -> Self {
        Self {
            transition_function,
        }
    }
    
    /// Determine the next state type
    pub(crate) fn determine_next_state_index(&self, df: &mut DataFrame) -> Result<usize, Box<dyn Error>> {
        (self.transition_function)(df)
    }
}

// Test TransitionManager
#[cfg(test)]
mod tests;
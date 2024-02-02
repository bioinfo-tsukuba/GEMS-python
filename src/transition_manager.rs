use std::error::Error;

// polars
use polars::prelude::*;

use crate::common_param_type::*;


// Define a type for TransitionFunction
/// TransitionFunction is a function that takes a DataFrame and returns a Result<StateIndex, Box<dyn Error>>
pub(crate) type TransitionFunction = dyn Fn(&mut DataFrame) -> Result<StateIndex, Box<dyn Error>>;

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
    pub(crate) fn determine_next_state_index(&self, df: &mut DataFrame) -> Result<StateIndex, Box<dyn Error>> {
        (self.transition_function)(df)
    }
}

// Test TransitionManager
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_manager() {
        // 例として、常に状態ID 1 を返す遷移関数
        let transition_func = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(1) // ここでは単純化のため、常に1を返す
        };
    
        // TransitionManagerのインスタンスを作成
        let manager = TransitionManager::new(Box::new(transition_func));
    
        // データフレームの例（実際にはここに実験データが入る）
        let mut variable_history = match df!("variable" => [1, 2, 3, 4, 5]) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        
        // Check the variable_history
        println!("variable_history:\n{}", variable_history);

        // Determine the next state type
        let next_state_type = match manager.determine_next_state_index(&mut variable_history) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        // Print result, from to
        println!("next_state_type: {}", next_state_type);
    }
}
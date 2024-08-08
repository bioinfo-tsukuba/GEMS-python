use std::{error::Error, os::unix::process, ops::Deref};
// polars
use polars::prelude::*;

use crate::common_param_type::Task;


type TaskFunction = dyn Fn(&DataFrame) -> Result<(String, OptimalTiming, PenaltyType), Box<dyn Error>>;
type ProcessingTimeFunction = dyn Fn(&DataFrame) -> Result<ProcessingTime, Box<dyn Error>>;


pub(crate) struct TaskGenerator {
    processing_time_function: Box<ProcessingTimeFunction>,
    timing_function: Box<TaskFunction>,
}
impl TaskGenerator {
    pub(crate)fn new(experiment_operation: Box<String>, processing_time_function: Box<ProcessingTimeFunction>, timing_func: Box<TaskFunction>)
    -> Self{
        Self {
            processing_time_function,
            timing_function: timing_func,
        }
    }

    pub fn generate_task(&self, variable_history: &mut DataFrame, experiment_name: String, experiment_uuid: String) -> Result<Task, Box<dyn Error>> {
        // Determine the optimal timing and the penalty type
        let (experiment_operation, optimal_timing, penalty_type) = (self.timing_function)(variable_history)?;
        let processing_time = (self.processing_time_function)(variable_history)?;

        // Return the task
        Ok(Task {
            optimal_timing,
            processing_time,
            penalty_type,
            experiment_operation,
            experiment_name,
            experiment_uuid,
            task_id: 0,
        })
    }
}


// test
#[cfg(test)]
mod tests;


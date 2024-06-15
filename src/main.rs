use experiment_manager::test_process_commands_delete;

// common_param_type モジュール
pub(crate) mod common_param_type;

pub(crate) mod transition_manager;

/// This module contains the tasks from all states.
pub(crate) mod task_generator;

/// This module contains the task scheduler.
/// task_scheduler recieves tasks and schedule them.
pub(crate) mod task_scheduler;


pub(crate) mod experiment_manager;

fn main(){
    test_process_commands_delete();
}

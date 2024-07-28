use super::*;
use crate::common_param_type::*;
use crate::transition_manager;
use crate::task_generator;
use crate::task_scheduler;
use polars::prelude::*;

#[test]
fn test_process_commands_delete() {
    let mut manager = OneMachineExperimentManager::new(
        vec![
            Experiment {
                experiment_name: "Experiment 2".into(),
                states: vec![],
                current_state_index: 0,
                shared_variable_history: df![
                    "cmd" => ["PAUSE"],
                    "time" => [100],
                ].unwrap(),
                experiment_uuid: "uuid-2".into(),
            },
            Experiment {
                experiment_name: "Experiment 1".into(),
                states: vec![],
                current_state_index: 0,
                shared_variable_history: df![
                    "cmd" => ["DELETE"],
                    "time" => [100],
                ].unwrap(),
                experiment_uuid: "uuid-1".into(),
            },
        ],
        vec![Task::default(), Task::default()],
        PathBuf::from("/path/to/dir"),
    );

    manager.show_experiment_names_and_state_names();

    manager.process_commands();
}

#[test]
fn test_process_commands_pause() {
    let mut manager = OneMachineExperimentManager::new(
        vec![
            Experiment {
                experiment_name: "Experiment 1".into(),
                states: vec![],
                current_state_index: 0,
                shared_variable_history: df![
                    "cmd" => ["PAUSE"],
                    "time" => [100],
                ].unwrap(),
                experiment_uuid: "uuid-1".into(),
            },
        ],
        vec![Task::default(), Task::default()],
        PathBuf::from("/path/to/dir"),
    );

    manager.process_commands();

    // Check if the command was processed and the shared_variable_history was updated
}

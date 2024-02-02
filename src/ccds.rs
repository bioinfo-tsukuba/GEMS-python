use std::error::Error;

use polars::prelude::*;

use crate::common_param_type::*;
use crate::experiment_manager::*;
use crate::task_generator::TaskGenerator;
use crate::transition_manager;
use crate::transition_manager::TransitionManager;
use crate::CCDS::simulator::*;
use crate::CCDS::manager::*;



// manager
pub(crate) mod manager;
// simulator
pub(crate) mod simulator;


#[cfg(test)]
mod tests{
    use polars::lazy::dsl::col;

    use super::*;

    #[test]
    fn test_CCDS() {
        // Define the transition functions
        let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
        let experiment_operation_state_0 = "Expire".to_string();
        let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::None)) // ここでは単純化のため、常に1を返す
        };
    
        let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(2) // ここでは単純化のため、常に1を返す
        };
        let experiment_operation_state_1 = "PASSAGE".to_string();
        let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((24*60, PenaltyType::Linear(1))) // ここでは単純化のため、常に1を返す
        };

        let transition_func_state_2 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(3) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_3 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(4) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_4 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(5) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_5 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(6) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_6 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(1) // ここでは単純化のため、常に1を返す
        };
    
    
        // Define the task generators
        let task_generator_state_0 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_1 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_2 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_3 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_4 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_5 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
        let task_generator_state_6 = |variable_history: &mut DataFrame| -> Result<Task, Box<dyn Error>> {
            Ok(Task {
                optimal_timing: 0,
                processing_time: 0,
                penalty_type: PenaltyType::None,
                experiment_operation: "Expire".to_string(),
                experiment_name: IPS_EXPERIMENT_NAME.to_string(),
                task_id: 0,
            })
        };
    
    
        let transition_manager_state_0 = TransitionManager::new(Box::new(transition_func_state_0));
    
        let mut states = vec![
            State::new(
                TransitionManager::new(Box::new(transition_func_state_0)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_0),
                    Box::new(processing_time_function_state_0),
                    Box::new(timing_function_state_0),
                ),
                0, 
                IPS_CULTURE_STATE_NAMES[0].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_1)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_1),
                    Box::new(processing_time_function_state_1),
                    Box::new(timing_function_state_1),
                ),
                1, 
                IPS_CULTURE_STATE_NAMES[1].to_string()
            ),
        ];
    
        let shared_variable_history = match df!("variable" => [1, 2, 3, 4, 5], "time" => [1, 2, 3, 4, 5]) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
    
        let mut iPS_culture_experiment = Experiment::new(
            IPS_EXPERIMENT_NAME.to_string(),
            states,
            0,
            shared_variable_history,
        );
    
        iPS_culture_experiment.show_experiment_name_and_state_names();
        let new_task = iPS_culture_experiment.execute_one_step();
        println!("new_task: {:?}", new_task);
    
    
        
    }

    #[test]
    fn test_cell_culture() {
        // Define the transition functions
        let experiment_operation_state_0 = "Expire".to_string();
        let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::None)) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
    

        let experiment_operation_state_1 = "PASSAGE".to_string();
        let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::Linear(1))) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(2) // ここでは単純化のため、常に1を返す
        };


        let experiment_operation_state_2 = "GET_IMAGE".to_string();
        let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(0)
        };
        let timing_function_state_2 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((24*60, PenaltyType::Linear(1)))
        };
        let transition_func_state_2 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            // If the latest density ("time" = max(time)) is less than 0.5, MediumChange, otherwise, Passage

            // sort the variable_history by "time", then get the latest density named "latest_density"
            let q = variable_history
                .clone()
                .lazy()
                .sort("time", 
                SortOptions {
                    descending: true,
                    nulls_last: false,
                    multithreaded: false,
                    maintain_order: false,
                    }
                )
                .select([col("density").cast(DataType::Float32)])
                ;
            let latest_density = q.collect().unwrap();
            let latest_density = latest_density.column("density").unwrap().get(0).unwrap();
            let latest_density = match latest_density{
                AnyValue::Float32(it) => it,
                _ => panic!("unexpected type"),
            };
            
            // If the latest density is less than 0.5, MediumChange, otherwise, Passage
            if latest_density < 0.5 {
                Ok(2)
            } else {
                Ok(1)
            }
        };

    
        let transition_manager_state_0 = TransitionManager::new(Box::new(transition_func_state_0));
    
        let mut states = vec![
            State::new(
                TransitionManager::new(Box::new(transition_func_state_0)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_0),
                    Box::new(processing_time_function_state_0),
                    Box::new(timing_function_state_0),
                ),
                0, 
                CELL_CULTURE_STATE_NAMES[0].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_1)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_1),
                    Box::new(processing_time_function_state_1),
                    Box::new(timing_function_state_1),
                ),
                1, 
                CELL_CULTURE_STATE_NAMES[1].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_2)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_2),
                    Box::new(processing_time_function_state_2),
                    Box::new(timing_function_state_2),
                ),
                1, 
                CELL_CULTURE_STATE_NAMES[2].to_string()
            ),
        ];
    
        let mut shared_variable_history = match df!("density" => [0.1, 0.2, 0.3, 0.4, 0.5], "time" => [1, 2, 3, 4, 5]) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
    
        let mut cell_culture_experiment = Experiment::new(
            CELL_CULTURE_EXPERIMENT_NAME.to_string(),
            states,
            2,
            shared_variable_history,
        );
    
        cell_culture_experiment.show_experiment_name_and_state_names();
        let new_task = cell_culture_experiment.execute_one_step();
        println!("new_task: {:?}", new_task);


        let mut schedule = OneMachineExperimentManager::new(
            vec![cell_culture_experiment],
            vec![new_task],
        );

        let new_result_of_experiment = match df!("density" => [0.6], "time" => [6]) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        let mut schedule_task = schedule.update_state_and_reschedule(0, new_result_of_experiment, 'a');
        println!("schedule.tasks: {:?}", schedule.tasks);

        println!("Test the simulator --------------------------\n\n");
        for step in 0..10 {
            // Get the earliest task
            let earliest_task_index = SimpleTaskSimulator::new(schedule_task.clone()).process_earliest_task();


            for task_index in earliest_task_index {
                let new_result_of_experiment = match df!("density" => [0.6], "time" => [6+step]) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                schedule_task = schedule.update_state_and_reschedule(task_index, new_result_of_experiment, 'a');
            }
            
            println!("schedule.tasks: {:?}", schedule.tasks);
            schedule.experiments[0].show_current_state_name();

        }
    
    
        
    }
}

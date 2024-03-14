use std::error::Error;

use polars::prelude::*;

use crate::common_param_type;
use crate::common_param_type::*;
use crate::experiment_manager::*;
use crate::task_generator::TaskGenerator;
use crate::transition_manager;
use crate::transition_manager::TransitionManager;
use crate::CCDS::simulator::*;
use crate::CCDS::manager::*;


pub(crate) static IPS_EXPERIMENT_NAME:&str = "IPS_CULTURE";

pub(crate) static IPS_CULTURE_STATE_NAMES:[&str; 7] = [
    "EXPIRE",
    "PASSAGE",
    "GET_IMAGE_1",
    "MEDIUM_CHANGE_1",
    "GET_IMAGE_2",
    "MEDIUM_CHANGE_2",
    "PLATE_COATING"
];

pub(crate) static IPS_CULTURE_PROCESSING_TIME:[common_param_type::ProcessingTime; 7] = [
    0,
    120,
    10,
    20,
    10,
    20,
    20,
];

pub(crate) static CELL_CULTURE_EXPERIMENT_NAME:&str = "CELL_CULTURE";

pub(crate) static CELL_CULTURE_STATE_NAMES:[&str; 4] = [
    "EXPIRE",
    "PASSAGE",
    "GET_IMAGE",
    "MEDIUM_CHANGE",
];



// manager
pub(crate) mod manager;
// simulator
pub(crate) mod simulator;

pub(crate) mod logistic_estimator;

#[cfg(test)]
mod tests{
    use std::{f32::consts::E, fs::create_dir, path::Path};

    use argmin::{core::{Executor}, solver::neldermead::NelderMead};
    use polars::lazy::dsl::col;

    use self::logistic_estimator::{calculate_logistic_rev, CellGrowthProblem};

    use super::*;

    #[test]
    fn test_CCDS() {

        // Reset global time
        let global_time = 0;
        overwrtite_global_time_manualy(global_time);

        
        let dir = Path::new("testcase/volatile");
        // Define the transition functions

        // EXPIRE
        let experiment_operation_state_0 = IPS_CULTURE_STATE_NAMES[0].to_string();
        let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[0]) 
        };
        let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::None)) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };

        // PASSAGE
        let experiment_operation_state_1 = IPS_CULTURE_STATE_NAMES[1].to_string();
        let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[1]) 
        };
        let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((4 * 60, PenaltyType::LinearWithRange(0, 100, 0, 1)))
        };
        let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(2) // ここでは単純化のため、常に1を返す
        };

        // GET_IMAGE_1
        let experiment_operation_state_2 = IPS_CULTURE_STATE_NAMES[2].to_string();
        let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[2]) 
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
            if latest_density < 0.05 {
                Ok(2)
            } else {
                Ok(3)
            }
        };

    
        // MEDIUM_CHANGE_1
        let experiment_operation_state_3 = IPS_CULTURE_STATE_NAMES[3].to_string();
        let processing_time_function_state_3 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[3]) 
        };
        let timing_function_state_3 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((48 * 60, PenaltyType::Linear(1)))
        };
        let transition_func_state_3 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(4) // ここでは単純化のため、常に1を返す
        };



        // GET_IMAGE_2
        let experiment_operation_state_4 = IPS_CULTURE_STATE_NAMES[4].to_string();
        let processing_time_function_state_4 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[4])
        };
        let timing_function_state_4 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((24*60, PenaltyType::Linear(1)))
        };
        let transition_func_state_4 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            // If the latest density ("time" = max(time)) is less than 0.5, MediumChange, otherwise, Passage

            // sort the variable_history by "time", then get the latest density named "latest_density"
            let q = variable_history
                .clone()
                .lazy()
                .filter(
                    col("time").is_not_null()
                    .and(col("density").is_not_null())
                )
                ;
            let df = q.collect().unwrap();
            let problem = CellGrowthProblem::new_from_df(df, "time", "density").unwrap();
            let init_param = problem.default_parameter()?;
            // Using Nelder-Mead
            let solver = NelderMead::new(init_param);


            // Create an `Executor` object 
            // If the cell growth is abnormal, return 5
            let res = 
            match Executor::new(problem, solver)
            .configure(|state|
                state
                    .max_iters(1000)
                    .target_cost(0.0)
            )
            .run() {
                Ok(it) => it,
                _ => return Ok(5),
            };

        // DEBUG
        println!("debug point 1");

            let best = res.state();
            let best = argmin::core::State::get_best_param(best).unwrap();
            // cast Vec<f64> to Vec<f32>
            let best = best.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            
            let reach_time = calculate_logistic_rev(0.3, best[0], best[1], best[2]);

            let current_time = variable_history
            .column("time")
            .unwrap()
            .f32()
            .unwrap()
            .max()
            .unwrap();

            // Round the reach_time to the nearest integer
            let reach_time = (reach_time - current_time).round() as i32;
            
            if reach_time < 48*60 {
                Ok(6)
            } else {
                Ok(5)
            }
        };
        
        // MEDIUM_CHANGE_2
        let experiment_operation_state_5 = IPS_CULTURE_STATE_NAMES[3].to_string();
        let processing_time_function_state_5 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[5])
        };
        let timing_function_state_5 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((24 * 60, PenaltyType::Linear(1)))
        };
        let transition_func_state_5 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(4) // ここでは単純化のため、常に1を返す
        };

        // PLATE_COATING
        let experiment_operation_state_6 = IPS_CULTURE_STATE_NAMES[5].to_string();
        let processing_time_function_state_6 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(IPS_CULTURE_PROCESSING_TIME[6])
        };
        let timing_function_state_6 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {

            // If the latest density ("time" = max(time)) is less than 0.5, MediumChange, otherwise, Passage

            // sort the variable_history by "time", then get the latest density named "latest_density"
            let q = variable_history
                .clone()
                .lazy()
                .filter(
                    col("time").is_not_null()
                    .and(col("density").is_not_null())
                )
                ;
            let df = q.collect().unwrap();
            let problem = CellGrowthProblem::new_from_df(df, "time", "density").unwrap();
            let init_param = problem.default_parameter()?;
            // Using Nelder-Mead
            let solver = NelderMead::new(init_param);
            
            // Create an `Executor` object 
            let res = Executor::new(problem, solver)
            .configure(|state|
                state
                    .max_iters(1000)
                    .target_cost(0.0)
            )
            .run()?;
            let best = res.state();
            let best = argmin::core::State::get_best_param(best).unwrap();
            // cast Vec<f64> to Vec<f32>
            let best = best.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            
            let reach_time = calculate_logistic_rev(0.3, best[0], best[1], best[2]);

            let current_time = variable_history
            .column("time")
            .unwrap()
            .f32()
            .unwrap()
            .max()
            .unwrap();

            // Round the reach_time to the nearest integer
            let reach_time = (reach_time - current_time).round() as common_param_type::OptimalTiming;
            
            if reach_time < 48*60 {
                Ok((reach_time - 4*60, PenaltyType::LinearWithRange(0, 200, 0, 200)))
            } else {
                panic!()
            }
        };
        let transition_func_state_6 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(1)
        };
    
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

            State::new(
                TransitionManager::new(Box::new(transition_func_state_2)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_2),
                    Box::new(processing_time_function_state_2),
                    Box::new(timing_function_state_2),
                ),
                2, 
                IPS_CULTURE_STATE_NAMES[2].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_3)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_3),
                    Box::new(processing_time_function_state_3),
                    Box::new(timing_function_state_3),
                ),
                3, 
                IPS_CULTURE_STATE_NAMES[3].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_4)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_4),
                    Box::new(processing_time_function_state_4),
                    Box::new(timing_function_state_4),
                ),
                4, 
                IPS_CULTURE_STATE_NAMES[4].to_string()
            ),

            State::new(
                TransitionManager::new(Box::new(transition_func_state_5)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_5),
                    Box::new(processing_time_function_state_5),
                    Box::new(timing_function_state_5),
                ),
                5, 
                IPS_CULTURE_STATE_NAMES[5].to_string()
            ),
            State::new(
                TransitionManager::new(Box::new(transition_func_state_6)),
                TaskGenerator::new(
                    Box::new(experiment_operation_state_6),
                    Box::new(processing_time_function_state_6),
                    Box::new(timing_function_state_6),
                ),
                6, 
                IPS_CULTURE_STATE_NAMES[6].to_string()
            ),
        ];
    
        let shared_variable_history = DataFrame::empty();

        let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
    
        let mut cell_culture_experiment = Experiment::new(
            IPS_EXPERIMENT_NAME.to_string(),
            states,
            1,
            shared_variable_history,
        );

        cell_culture_experiment.show_experiment_name_and_state_names();
        let new_task = cell_culture_experiment.generate_task_of_the_state();
        println!("new_task: {:?}", new_task);


        let mut schedule = OneMachineExperimentManager::new(
            vec![cell_culture_experiment],
            vec![new_task],
            std::path::PathBuf::new(),
        );

        let mut normal_cell_simulators = vec![
            NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05),
        ];

        let mut maholo_simulator = vec![
            Simulator::new(DataFrame::empty(), NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05))
        ];

        

        println!("schedule.tasks: {:?}", schedule.tasks);


        schedule.show_experiment_names_and_state_names();
        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler(schedule.tasks.clone());
        println!("schedule_task: {:?}", schedule_task);

        println!("Test the simulator --------------------------\n\n");
        for step in 0..10 {
            println!("step: {}=================", step);
            println!("Current state name: {}", schedule.experiments[0].states[schedule.experiments[0].current_state_index].state_name);
            println!("Current history: {}", schedule.experiments[0].shared_variable_history);

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }




            // // Get the earliest task
            let (task_id, new_result_of_experiment, update_type) = SimpleTaskSimulator::new(schedule_task.clone())
            .process_earliest_task(&schedule, &mut maholo_simulator);

            println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
            schedule_task = schedule.update_state_and_reschedule(task_id, new_result_of_experiment, update_type);
            


            // for task_id in earliest_task_ids{
            //     let new_result_of_experiment = match df!("density" => [0.6], "time" => [6+step]) {
            //         Ok(it) => it,
            //         Err(err) => panic!("{}", err),
            //     };
                
            // }
            
            println!("schedule.tasks: {:?}", schedule.tasks);
            schedule.experiments[0].show_current_state_name();

            // create csv of the shared_variable_history as dir/step_{}.csv
            let mut file = std::fs::File::create(&step_dir.join("shared_variable_history.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut schedule.experiments[0].shared_variable_history)
                .unwrap();

            let schedule_path = step_dir.join("schedule.csv");
            match scheduled_task_convert_to_csv(&schedule_path, &schedule_task) {
                Ok(_) => (),
                Err(err) => panic!("{}", err),
            }
            schedule_task = read_scheduled_task(&schedule_path).unwrap();
            schedule.show_experiment_names_and_state_names();
            for normal_cell_simulator in &mut normal_cell_simulators{
                println!("normal_cell_simulator: {:?}", normal_cell_simulator);
            }
        }
        
    }

    #[test]
    fn test_cell_culture() {
        let dir = Path::new("testcase/volatile");

        match create_dir(dir){
            Ok(_) => (),
            Err(err) => println!("{}", err),
        }
        // Define the transition functions
        let experiment_operation_state_0 = "Expire".to_string();
        let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(10) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::None)) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(0) // ここでは単純化のため、常に1を返す
        };
    

        let experiment_operation_state_1 = "PASSAGE".to_string();
        let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(10) // ここでは単純化のため、常に1を返す
        };
        let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::Linear(1))) // ここでは単純化のため、常に1を返す
        };
        let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(2) // ここでは単純化のため、常に1を返す
        };


        let experiment_operation_state_2 = "GET_IMAGE".to_string();
        let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(10) // ここでは単純化のため、常に1を返す
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
            std::path::PathBuf::new(),
        );

        let new_result_of_experiment = match df!("density" => [0.6], "time" => [6]) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        let mut schedule_task = schedule.update_state_and_reschedule(0, new_result_of_experiment, 'a');
        println!("schedule.tasks: {:?}", schedule.tasks);

        println!("Test the simulator --------------------------\n\n");
        for step in 0..10 {

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }


            // Get the earliest task
            // let earliest_task_index = SimpleTaskSimulator::new(
            //     schedule_task.clone())
            //     .process_earliest_task(&step_dir);


            // for task_index in earliest_task_index {
            //     let new_result_of_experiment = match df!("density" => [0.6], "time" => [6+step]) {
            //         Ok(it) => it,
            //         Err(err) => panic!("{}", err),
            //     };
            //     schedule_task = schedule.update_state_and_reschedule(task_index, new_result_of_experiment, 'a');
            // }
            
            println!("schedule.tasks: {:?}", schedule.tasks);
            schedule.experiments[0].show_current_state_name();

            // create csv of the shared_variable_history as dir/step_{}.csv
            let mut file = std::fs::File::create(&step_dir.join("shared_variable_history.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut schedule.experiments[0].shared_variable_history)
                .unwrap();

            let schedule_path = step_dir.join("schedule.csv");
            match scheduled_task_convert_to_csv(&schedule_path, &schedule_task) {
                Ok(_) => (),
                Err(err) => panic!("{}", err),
            }
            schedule_task = read_scheduled_task(&schedule_path).unwrap();

        }
    
    
        
    }
}

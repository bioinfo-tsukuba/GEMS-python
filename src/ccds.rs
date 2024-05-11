use std::error::Error;

use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;
use polars::prelude::*;

use crate::common_param_type;
use crate::common_param_type::*;
use crate::experiment_manager::*;
use crate::task_generator::TaskGenerator;
use crate::transition_manager;
use crate::transition_manager::TransitionManager;
use crate::ccds::logistic_estimator::calculate_logistic_inverse_binary_search;
use crate::ccds::logistic_estimator::calculate_logistic_inverse_function;
use crate::ccds::logistic_estimator::CellGrowthProblem;
use crate::ccds::simulator::*;
use crate::ccds::manager::*;

pub(crate) const IPS_OPERATION_INTERVAL: i64 = 24*60;
pub(crate) const NORMAL_CELL_OPERATION_INTERVAL: i64 = 4*60;


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

/// Generate the states of the cell culture experiment
pub(crate) fn iPS_culture_experiment_states() -> Vec<State>{
    // EXPIRE
    let experiment_operation_state_0 = IPS_CULTURE_STATE_NAMES[0].to_string();
    let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(IPS_CULTURE_PROCESSING_TIME[0]) 
    };
    let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 0, PenaltyType::None))  
    };
    let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(0)  
    };

    // PASSAGE
    let experiment_operation_state_1 = IPS_CULTURE_STATE_NAMES[1].to_string();
    let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(IPS_CULTURE_PROCESSING_TIME[1]) 
    };
    let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 4 * 60, PenaltyType::LinearWithRange{ lower: 0, lower_coefficient: 100, upper: 0, upper_coefficient: 100 }))
    };
    let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(2)  
    };

    // GET_IMAGE_1
    let experiment_operation_state_2 = IPS_CULTURE_STATE_NAMES[2].to_string();
    let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(IPS_CULTURE_PROCESSING_TIME[2]) 
    };
    let timing_function_state_2 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + IPS_OPERATION_INTERVAL/*replace for search:GET_IMAGE_1;ips*/, PenaltyType::Linear { coefficient: 1 }))
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
            .select([col("density")])
            ;
        let latest_density = q.collect().unwrap();
        let latest_density = latest_density.column("density").unwrap().get(0).unwrap();
        let latest_density = match latest_density{
            AnyValue::Float64(it) => it,
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
        Ok((get_current_absolute_time() + 48 * 60, PenaltyType::Linear { coefficient: 1 }))
    };
    let transition_func_state_3 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(4)  
    };



    // GET_IMAGE_2
    let experiment_operation_state_4 = IPS_CULTURE_STATE_NAMES[4].to_string();
    let processing_time_function_state_4 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(IPS_CULTURE_PROCESSING_TIME[4])
    };
    let timing_function_state_4 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 0/*replace for search:GET_IMAGE_2;ips*/, PenaltyType::Linear { coefficient: 1 }))
    };
    let transition_func_state_4 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        // If the latest density ("time" = max(time)) is less than 0.5, MediumChange, otherwise, Passage

        // sort the variable_history by "time", then get the latest density named "latest_density"
        // Convert time (sec) to time (min)
        let q = variable_history
            .clone()
            .lazy()
            .filter(
                col("time").is_not_null()
                .and(col("density").is_not_null())
            )
            ;
        let df = q.collect().unwrap();
        

        // Sort by time with operation == "PASSAGE"
        let latest_passage_time = df
            .clone()
            .lazy()
            .filter(
                col("operation").eq(lit("PASSAGE"))
            )
            .sort("time", 
            SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: false,
                maintain_order: false,
                }
            )
            .select([col("time")]);
        let latest_passage_time = latest_passage_time.collect().unwrap();
        let latest_passage_time = latest_passage_time.column("time")?.iter().nth(0);
        let latest_passage_time = match latest_passage_time{
            Some(it) => it.try_extract::<f64>()?,
            None => 0 as f64,
        };

        let passed_time_from_latest_passage = get_current_absolute_time() as f64 - latest_passage_time;
        println!("passed_time_from_latest_passage: {}", passed_time_from_latest_passage);
        if passed_time_from_latest_passage < 24.0 * 60.0 * 2.0 {
            println!("passed_time_from_latest_passage < 24.0 * 60.0 * 2.0");
            return Ok(4);
        }

        // Filter the all time > latest_passage_time and shift - latest_passage_time
        let df = df
            .clone()
            .lazy()
            .filter(
                col("time").gt_eq(latest_passage_time)
            )
            .with_columns(
                [
                    (col("time") - lit(latest_passage_time)).alias("time"),
                ]
            )
            .select([col("time").cast(DataType::Float64), col("density")]);

        let df = df.collect().unwrap();
        eprintln!("Passage clipped df: {:?}", df);


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

        let best = res.state();
        let best = argmin::core::State::get_best_param(best).unwrap();
        eprintln!("best params MEDIUM CHANGE 2: {:?}", best);
        
        let reach_time = calculate_logistic_inverse_function(0.3, best[0], best[1], best[2]);
        let reach_time = reach_time + latest_passage_time;
        eprintln!("estimated reach_time: {}", reach_time);

        let current_time = get_current_absolute_time() as f64;

        // Round the reach_time to the nearest integer
        let reach_time = (reach_time - current_time).round() as i64;
        
        if reach_time < IPS_OPERATION_INTERVAL*2 {
            Ok(6)
        } else {
            Ok(5)
        }
    };

    // MEDIUM_CHANGE_2
    let experiment_operation_state_5 = IPS_CULTURE_STATE_NAMES[5].to_string();
    let processing_time_function_state_5 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(IPS_CULTURE_PROCESSING_TIME[5])
    };
    let timing_function_state_5 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + IPS_OPERATION_INTERVAL, PenaltyType::Linear { coefficient: 1 }))
    };
    let transition_func_state_5 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(4)  
    };

    // PLATE_COATING
    let experiment_operation_state_6 = IPS_CULTURE_STATE_NAMES[6].to_string();
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

        // Sort by time with operation == "PASSAGE"
        let latest_passage_time = df
            .clone()
            .lazy()
            .filter(
                col("operation").eq(lit("PASSAGE"))
            )
            .sort("time", 
            SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: false,
                maintain_order: false,
                }
            )
            .select([col("time")]);
        let latest_passage_time = latest_passage_time.collect().unwrap();
        let latest_passage_time = latest_passage_time.column("time")?.iter().nth(0);
        let latest_passage_time = match latest_passage_time{
            Some(it) => it.try_extract::<f64>()?,
            None => 0 as f64,
        };

        // Filter the all time > latest_passage_time and shift - latest_passage_time
        let df = df
            .clone()
            .lazy()
            .filter(
                col("time").gt_eq(latest_passage_time)
            )
            .with_columns(
                [
                    (col("time") - lit(latest_passage_time)).alias("time"),
                ]
            )
            .select([col("time").cast(DataType::Float64), col("density")]);

        let df = df.collect().unwrap();
        eprintln!("Passage clipped df: {:?}", df);


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
        eprintln!("best params PLATE COATING: {:?}", best);
        
        let reach_time = calculate_logistic_inverse_function(0.3, best[0], best[1], best[2]);
        let reach_time = reach_time + latest_passage_time;
        eprintln!("estimated reach_time: {}", reach_time);

        let current_time = get_current_absolute_time();

        // Round the reach_time to the nearest integer
        let reach_time = reach_time.round() as common_param_type::OptimalTiming;
        
        if reach_time - current_time < IPS_OPERATION_INTERVAL*2 {
            Ok((reach_time - 4*60, PenaltyType::LinearWithRange{ lower: 0, lower_coefficient: 100, upper: 0, upper_coefficient: 100 }))
        } else {
            panic!()
        }
    };
    let transition_func_state_6 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(1)
    };

     vec![
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
    ]
}

pub(crate) static REAGENT_EXPERIMENT_NAME:&str = "REAGENT";

pub(crate) static REAGENT_STATE_NAMES:[&str; 2] = [
    "EXPIRE",
    "REAGENT_FILL",
];

pub(crate) static REAGENT_PROCESSING_TIME:[common_param_type::ProcessingTime; 2] = [
    0,
    60,
];

/// Generate the states of the reagent experiment
pub(crate) fn reagent_experiment_states() -> Vec<State>{
    // EXPIRE
    let experiment_operation_state_0 = REAGENT_STATE_NAMES[0].to_string();
    let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(REAGENT_PROCESSING_TIME[0]) 
    };
    let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 0, PenaltyType::None))  
    };
    let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(0)  
    };

    // REAGENT_FILL
    let experiment_operation_state_1 = REAGENT_STATE_NAMES[1].to_string();
    let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(REAGENT_PROCESSING_TIME[1]) 
    };
    let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 7*24*60, PenaltyType::CyclicalRestPenaltyWithLinear { start_minute: 0, cycle_minute: 24*60, ranges: vec![(0, 9*60), (18*60, 24*60)], coefficient: 1 }))
    };
    let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(1)  
    };

    vec![
        State::new(
            TransitionManager::new(Box::new(transition_func_state_0)),
            TaskGenerator::new(
                Box::new(experiment_operation_state_0),
                Box::new(processing_time_function_state_0),
                Box::new(timing_function_state_0),
            ),
            0, 
            REAGENT_STATE_NAMES[0].to_string()
        ),

        State::new(
            TransitionManager::new(Box::new(transition_func_state_1)),
            TaskGenerator::new(
                Box::new(experiment_operation_state_1),
                Box::new(processing_time_function_state_1),
                Box::new(timing_function_state_1),
            ),
            1, 
            REAGENT_STATE_NAMES[1].to_string()
        ),
    ]
}



pub(crate) static NORMAL_EXPERIMENT_NAME:&str = "NORMAL_CULTURE";

pub(crate) static NORMAL_CULTURE_STATE_NAMES:[&str; 3] = [
    "EXPIRE",
    "PASSAGE",
    "GET_IMAGE_1",
];

pub(crate) static NORMAL_CULTURE_PROCESSING_TIME:[common_param_type::ProcessingTime; 3] = [
    0,
    120,
    10,
];

/// Generate the states of the normal culture experiment
pub(crate) fn normal_culture_experiment_states() -> Vec<State>{
    // EXPIRE
    let experiment_operation_state_0 = NORMAL_CULTURE_STATE_NAMES[0].to_string();
    let processing_time_function_state_0 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(NORMAL_CULTURE_PROCESSING_TIME[0]) 
    };
    let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + 0, PenaltyType::None))  
    };
    let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(0)  
    };

    // PASSAGE
    let experiment_operation_state_1 = NORMAL_CULTURE_STATE_NAMES[1].to_string();
    let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(NORMAL_CULTURE_PROCESSING_TIME[1]) 
    };
    let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {

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

        // Sort by time with operation == "PASSAGE"
        let latest_passage_time = df
            .clone()
            .lazy()
            .filter(
                col("operation").eq(lit("PASSAGE"))
            )
            .sort("time", 
            SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: false,
                maintain_order: false,
                }
            )
            .select([col("time")]);
        let latest_passage_time = latest_passage_time.collect().unwrap();
        let latest_passage_time = latest_passage_time.column("time")?.iter().nth(0);
        let latest_passage_time = match latest_passage_time{
            Some(it) => it.try_extract::<f64>()?,
            None => 0 as f64,
        };

        // Filter the all time > latest_passage_time and shift - latest_passage_time
        let df = df
            .clone()
            .lazy()
            .filter(
                col("time").gt_eq(latest_passage_time)
            )
            .with_columns(
                [
                    (col("time") - lit(latest_passage_time)).alias("time"),
                ]
            )
            .select([col("time").cast(DataType::Float64), col("density")]);

        let df = df.collect().unwrap();
        eprintln!("Passage clipped df: {:?}", df);


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
        eprintln!("best params PLATE COATING: {:?}", best);
        
        let reach_time = calculate_logistic_inverse_function(0.7, best[0], best[1], best[2]);
        let reach_time = reach_time + latest_passage_time;
        eprintln!("estimated reach_time: {}", reach_time);

        let current_time = get_current_absolute_time();

        // Round the reach_time to the nearest integer
        let reach_time = reach_time.round() as common_param_type::OptimalTiming;
        
        if reach_time - current_time < NORMAL_CELL_OPERATION_INTERVAL*2 {
            Ok((reach_time, PenaltyType::Linear { coefficient: 1 }))
        } else {
            panic!()
        }
    };
    let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
        Ok(2)  
    };

    // GET_IMAGE_1
    let experiment_operation_state_2 = NORMAL_CULTURE_STATE_NAMES[2].to_string();
    let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(NORMAL_CULTURE_PROCESSING_TIME[2]) 
    };
    let timing_function_state_2 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
        Ok((get_current_absolute_time() + NORMAL_CELL_OPERATION_INTERVAL/*replace for search:GET_IMAGE_1;normalcell*/, PenaltyType::Linear { coefficient: 1 }))
    };
    let transition_func_state_2 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
       // If the latest density ("time" = max(time)) is less than 0.5, MediumChange, otherwise, Passage

        // sort the variable_history by "time", then get the latest density named "latest_density"
        // Convert time (sec) to time (min)
        let q = variable_history
            .clone()
            .lazy()
            .filter(
                col("time").is_not_null()
                .and(col("density").is_not_null())
            )
            ;
        let df = q.collect().unwrap();
        

        // Sort by time with operation == "PASSAGE"
        let latest_passage_time = df
            .clone()
            .lazy()
            .filter(
                col("operation").eq(lit("PASSAGE"))
            )
            .sort("time", 
            SortOptions {
                descending: true,
                nulls_last: false,
                multithreaded: false,
                maintain_order: false,
                }
            )
            .select([col("time")]);
        let latest_passage_time = latest_passage_time.collect().unwrap();
        let latest_passage_time = latest_passage_time.column("time")?.iter().nth(0);
        let latest_passage_time = match latest_passage_time{
            Some(it) => it.try_extract::<f64>()?,
            None => 0 as f64,
        };

        let passed_time_from_latest_passage = get_current_absolute_time() as f64 - latest_passage_time;
        println!("passed_time_from_latest_passage: {}", passed_time_from_latest_passage);
        if passed_time_from_latest_passage < 24.0 * 60.0 {
            println!("passed_time_from_latest_passage < 24.0 * 60.0");
            return Ok(2);
        }

        // Filter the all time > latest_passage_time and shift - latest_passage_time
        let df = df
            .clone()
            .lazy()
            .filter(
                col("time").gt_eq(latest_passage_time)
            )
            .with_columns(
                [
                    (col("time") - lit(latest_passage_time)).alias("time"),
                ]
            )
            .select([col("time").cast(DataType::Float64), col("density")]);

        let df = df.collect().unwrap();
        eprintln!("Passage clipped df: {:?}", df);


        let problem = CellGrowthProblem::new_from_df(df, "time", "density").unwrap();
        let init_param = problem.default_parameter()?;
        // Using Nelder-Mead
        let solver = NelderMead::new(init_param);


        // Create an `Executor` object 
        // If the cell growth is abnormal, return 2
        let res = 
        match Executor::new(problem, solver)
        .configure(|state|
            state
                .max_iters(1000)
                .target_cost(0.0)
        )
        .run() {
            Ok(it) => it,
            _ => return Ok(2),
        };

        let best = res.state();
        let best = argmin::core::State::get_best_param(best).unwrap();
        eprintln!("best params MEDIUM CHANGE 2: {:?}", best);
        
        let reach_time = calculate_logistic_inverse_function(0.7, best[0], best[1], best[2]);
        let reach_time = reach_time + latest_passage_time;
        eprintln!("estimated reach_time: {}", reach_time);

        let current_time = get_current_absolute_time() as f64;

        // Round the reach_time to the nearest integer
        let reach_time = (reach_time - current_time).round() as i64;
        
        if reach_time < NORMAL_CELL_OPERATION_INTERVAL*2 {
            Ok(1)
        } else {
            Ok(2)
        }
    };

    vec![
        State::new(
            TransitionManager::new(Box::new(transition_func_state_0)),
            TaskGenerator::new(
                Box::new(experiment_operation_state_0),
                Box::new(processing_time_function_state_0),
                Box::new(timing_function_state_0),
            ),
            0, 
            NORMAL_CULTURE_STATE_NAMES[0].to_string()
        ),

        State::new(
            TransitionManager::new(Box::new(transition_func_state_1)),
            TaskGenerator::new(
                Box::new(experiment_operation_state_1),
                Box::new(processing_time_function_state_1),
                Box::new(timing_function_state_1),
            ),
            1, 
            NORMAL_CULTURE_STATE_NAMES[1].to_string()
        ),

        State::new(
            TransitionManager::new(Box::new(transition_func_state_2)),
            TaskGenerator::new(
                Box::new(experiment_operation_state_2),
                Box::new(processing_time_function_state_2),
                Box::new(timing_function_state_2),
            ),
            2, 
            NORMAL_CULTURE_STATE_NAMES[2].to_string()
        ),
    ]
}

// manager
pub(crate) mod manager;
// simulator
pub(crate) mod simulator;

pub(crate) mod logistic_estimator;

#[cfg(test)]
mod tests{
    use std::{default, f32::consts::E, fmt::format, fs::{create_dir, create_dir_all, remove_dir_all}, path::{Path, PathBuf}, vec};

    use argmin::{core::{Executor}, solver::neldermead::NelderMead};
    use polars::lazy::dsl::col;
    use tests::logistic_estimator::calculate_logistic_inverse_binary_search;

    use self::logistic_estimator::CellGrowthProblem;

    use super::*;

    fn robot_mediator(
        one_machine_experiment: &super::OneMachineExperimentManager, 
        simulators: &mut Vec<Simulator>,
        step_dir: &Path,
    ) -> (TaskId, DataFrame, char) {

        let task_id_path = step_dir.join("task_id.txt");
        let new_result_of_experiment_path = step_dir.join("new_result_of_experiment.csv");
        let update_type_path = step_dir.join("update_type.txt");

        'read_result_loop: loop{
            std::thread::sleep(std::time::Duration::from_secs(1));
            if task_id_path.exists() && new_result_of_experiment_path.exists() && update_type_path.exists(){
                std::thread::sleep(std::time::Duration::from_secs(1));
                break 'read_result_loop;
            }
        }

        let task_id = std::fs::read_to_string(&task_id_path).unwrap().parse::<TaskId>().unwrap();
        let new_result_of_experiment = CsvReader::from_path(&new_result_of_experiment_path).unwrap()
            .has_header(true)
            .finish()
            .unwrap();
        let update_type = std::fs::read_to_string(&update_type_path).unwrap().parse::<char>().unwrap();
        (task_id, new_result_of_experiment, update_type)
    }




    fn run_simulation_SA_with_stop(schedule_task: Vec<ScheduledTask> , schedule: OneMachineExperimentManager, maholo_simulator: Vec<Simulator>, loop_num: usize, dir: &Path){
        let mut schedule_task = schedule_task;
        let mut schedule = schedule;
        let mut maholo_simulator = maholo_simulator;
        // Save the all experiment and the states
        for experiment in &schedule.experiments{
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            let experiment_dir = dir.join(uuid);
            match create_dir(&experiment_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(experiment_dir.join("experiment.csv")).unwrap();
            // create string vec from the state names
            let names: Vec<String> = experiment.states.iter().map(|state| state.state_name.clone()).collect();
            let names = names.join(",");
            std::io::Write::write_all(&mut file, names.as_bytes()).unwrap();
        }
        println!("Test the simulator --------------------------\n\n");
        for step in 0..loop_num {
            println!("step: {}=================", step);

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            // // Get the earliest task
            let (task_id, new_result_of_experiment, update_type) = SimpleTaskSimulator::new(schedule_task.clone())
            .process_earliest_task(&schedule, &mut maholo_simulator);

            println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
            schedule_task = schedule.update_state_and_reschedule(task_id, new_result_of_experiment, update_type, 's');
            
            println!("simulate: schedule.tasks");
            for task in &schedule_task{
                println!("{:?}", task);
            }

            // create csv of the shared_variable_history as dir/step_{}.csv
            for experiment in &schedule.experiments{
                let uuid = experiment.experiment_uuid.clone();
                let uuid = format!("EXPERIMENT_{}", uuid);
                let experiment_dir = step_dir.join(uuid);
                match create_dir_all(&experiment_dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }
                let mut file = std::fs::File::create(experiment_dir.join("shared_variable_history.csv")).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut experiment.shared_variable_history.clone())
                    .unwrap();

                let mut file = std::fs::File::create(experiment_dir.join("current_state.csv")).unwrap();
                let mut current_state = df!(
                    "current_state" => [experiment.get_current_state_name()]
                ).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut current_state)
                    .unwrap();

            }
            let mut file = std::fs::File::create(&step_dir.join("shared_variable_history.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut schedule.experiments[0].shared_variable_history)
                .unwrap();

            let schedule_path = step_dir.join("schedule.csv");
            match scheduled_task_convert_to_csv(&schedule_path, &schedule_task) {
                Ok(_) => (),
                Err(err) => panic!("{}", err),
            }

            let absolute_time = get_current_absolute_time();
            let absolute_time = format!("{}", absolute_time);
            let mut file = std::fs::File::create(&step_dir.join("absolute_time.txt")).unwrap();
            std::io::Write::write_all(&mut file, absolute_time.as_bytes()).unwrap();
            schedule_task = read_scheduled_task(&schedule_path).unwrap();
        }

        // Save the simulation result df
        for experiment_index in 0..schedule.experiments.len(){
            let mut experiment = &mut schedule.experiments[experiment_index];
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            // Make the directory
            let dir = dir.join(uuid);
            match create_dir(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(dir.join("simulation_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut experiment.shared_variable_history)
                .unwrap();
            let mut file = std::fs::File::create(dir.join("simulator_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut maholo_simulator[experiment_index].cell_history().clone())
                .unwrap();    
        }
        
         
    }


    fn run_simulation_SA(schedule_task: Vec<ScheduledTask> , schedule: OneMachineExperimentManager, maholo_simulator: Vec<Simulator>, loop_num: usize, dir: &Path){
        let mut schedule_task = schedule_task;
        let mut schedule = schedule;
        let mut maholo_simulator = maholo_simulator;
        // Save the all experiment and the states
        for experiment in &schedule.experiments{
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            let experiment_dir = dir.join(uuid);
            match create_dir(&experiment_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(experiment_dir.join("experiment.csv")).unwrap();
            // create string vec from the state names
            let names: Vec<String> = experiment.states.iter().map(|state| state.state_name.clone()).collect();
            let names = names.join(",");
            std::io::Write::write_all(&mut file, names.as_bytes()).unwrap();
        }
        println!("Test the simulator --------------------------\n\n");
        for step in 0..loop_num {
            println!("step: {}=================", step);

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            // // Get the earliest task
            let (task_id, new_result_of_experiment, update_type) = SimpleTaskSimulator::new(schedule_task.clone())
            .process_earliest_task(&schedule, &mut maholo_simulator);

            println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
            schedule_task = schedule.update_state_and_reschedule(task_id, new_result_of_experiment, update_type, 's');
            
            println!("simulate: schedule.tasks");
            for task in &schedule_task{
                println!("{:?}", task);
            }

            // create csv of the shared_variable_history as dir/step_{}.csv
            for experiment in &schedule.experiments{
                let uuid = experiment.experiment_uuid.clone();
                let uuid = format!("EXPERIMENT_{}", uuid);
                let experiment_dir = step_dir.join(uuid);
                match create_dir_all(&experiment_dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }
                let mut file = std::fs::File::create(experiment_dir.join("shared_variable_history.csv")).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut experiment.shared_variable_history.clone())
                    .unwrap();

                let mut file = std::fs::File::create(experiment_dir.join("current_state.csv")).unwrap();
                let mut current_state = df!(
                    "current_state" => [experiment.get_current_state_name()]
                ).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut current_state)
                    .unwrap();

            }
            let mut file = std::fs::File::create(&step_dir.join("shared_variable_history.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut schedule.experiments[0].shared_variable_history)
                .unwrap();

            let schedule_path = step_dir.join("schedule.csv");
            match scheduled_task_convert_to_csv(&schedule_path, &schedule_task) {
                Ok(_) => (),
                Err(err) => panic!("{}", err),
            }

            let absolute_time = get_current_absolute_time();
            let absolute_time = format!("{}", absolute_time);
            let mut file = std::fs::File::create(&step_dir.join("absolute_time.txt")).unwrap();
            std::io::Write::write_all(&mut file, absolute_time.as_bytes()).unwrap();
            schedule_task = read_scheduled_task(&schedule_path).unwrap();
        }

        // Save the simulation result df
        for experiment_index in 0..schedule.experiments.len(){
            let mut experiment = &mut schedule.experiments[experiment_index];
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            // Make the directory
            let dir = dir.join(uuid);
            match create_dir(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(dir.join("simulation_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut experiment.shared_variable_history)
                .unwrap();
            let mut file = std::fs::File::create(dir.join("simulator_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut maholo_simulator[experiment_index].cell_history().clone())
                .unwrap();    
        }
        
         
    }


    fn run_simulation_fifo(schedule_task: Vec<ScheduledTask> , schedule: OneMachineExperimentManager, maholo_simulator: Vec<Simulator>, loop_num: usize, dir: &Path){
        let mut schedule_task = schedule_task;
        let mut schedule = schedule;
        let mut maholo_simulator = maholo_simulator;
        // Save the all experiment and the states
        for experiment in &schedule.experiments{
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            let experiment_dir = dir.join(uuid);
            match create_dir(&experiment_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(experiment_dir.join("experiment.csv")).unwrap();
            // create string vec from the state names
            let names: Vec<String> = experiment.states.iter().map(|state| state.state_name.clone()).collect();
            let names = names.join(",");
            std::io::Write::write_all(&mut file, names.as_bytes()).unwrap();
        }
        println!("Test the simulator --------------------------\n\n");
        for step in 0..loop_num {
            println!("step: {}=================", step);

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            // // Get the earliest task
            let (task_id, new_result_of_experiment, update_type) = SimpleTaskSimulator::new(schedule_task.clone())
            .process_earliest_task(&schedule, &mut maholo_simulator);

            println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
            schedule_task = schedule.update_state_and_reschedule(task_id, new_result_of_experiment, update_type, 'f');
            


            // for task_id in earliest_task_ids{
            //     let new_result_of_experiment = match df!("density" => [0.6], "time" => [6+step]) {
            //         Ok(it) => it,
            //         Err(err) => panic!("{}", err),
            //     };
                
            // }
            
            println!("simulate: schedule.tasks");
            for task in &schedule_task{
                println!("{:?}", task);
            }

            // create csv of the shared_variable_history as dir/step_{}.csv
            for experiment in &schedule.experiments{
                let uuid = experiment.experiment_uuid.clone();
                let uuid = format!("EXPERIMENT_{}", uuid);
                let experiment_dir = step_dir.join(uuid);
                match create_dir_all(&experiment_dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }
                let mut file = std::fs::File::create(experiment_dir.join("shared_variable_history.csv")).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut experiment.shared_variable_history.clone())
                    .unwrap();

                let mut file = std::fs::File::create(experiment_dir.join("current_state.csv")).unwrap();
                let mut current_state = df!(
                    "current_state" => [experiment.get_current_state_name()]
                ).unwrap();
                CsvWriter::new(&mut file)
                    .finish(&mut current_state)
                    .unwrap();

            }
            let mut file = std::fs::File::create(&step_dir.join("shared_variable_history.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut schedule.experiments[0].shared_variable_history)
                .unwrap();

            let schedule_path = step_dir.join("schedule.csv");
            match scheduled_task_convert_to_csv(&schedule_path, &schedule_task) {
                Ok(_) => (),
                Err(err) => panic!("{}", err),
            }

            let absolute_time = get_current_absolute_time();
            let absolute_time = format!("{}", absolute_time);
            let mut file = std::fs::File::create(&step_dir.join("absolute_time.txt")).unwrap();
            std::io::Write::write_all(&mut file, absolute_time.as_bytes()).unwrap();
            schedule_task = read_scheduled_task(&schedule_path).unwrap();
        }

        // Save the simulation result df
        for experiment_index in 0..schedule.experiments.len(){
            let mut experiment = &mut schedule.experiments[experiment_index];
            let uuid = experiment.experiment_uuid.clone();
            let uuid = format!("EXPERIMENT_{}", uuid);
            // Make the directory
            let dir = dir.join(uuid);
            match create_dir(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(dir.join("simulation_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut experiment.shared_variable_history)
                .unwrap();
            let mut file = std::fs::File::create(dir.join("simulator_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut maholo_simulator[experiment_index].cell_history().clone())
                .unwrap();    
        }
        
         
    }

    #[test]
    fn test_robot_mediator(){
        let schedule = OneMachineExperimentManager::new(
            Vec::with_capacity(1),
            Vec::with_capacity(1),
            PathBuf::new(),
        );

        let mut maholo_simulator = vec![Simulator::new(DataFrame::empty(), NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); 1];
        let mut schedule_task = vec![];
        let task = ScheduledTask::default();
        schedule_task.push(task);
        let dir = Path::new("/home/cab314/Project/GEMS/testcase");
        let (task_id, new_result_of_experiment, update_type) = robot_mediator(&schedule, &mut maholo_simulator, dir);
        println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
    }


    #[test]
    fn test_small_mix_SA_long_vs_FIFO_with_reagent_change() {
        // /home/cab314/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_mix_SA_vs_FIFO --exact --nocapture  > testcase/aaa.txt
        // Reset global time

        let patterns = ["SA", "FIFO"];
        let sim_num = 1;
        for pattern in patterns.iter(){
            for i in 0..sim_num{
                let global_time = 0;
                overwrtite_global_time_manualy(global_time);


                let ips_num = 1;
                let normal_num = 1;
                let all_num = ips_num + normal_num + 1;

                let dir = Path::new(RESULT_PATH);
                let dir = &dir.join("2024-04-10/sa_vs_fifo_reagent_chenge/small_mix/with_naive_SA").join(pattern).join(format!("sim_{}", i));
                println!("dir: {:?}", dir);
                match create_dir_all(&dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }

                let mut schedule = OneMachineExperimentManager::new(
                    Vec::with_capacity(all_num),
                    Vec::with_capacity(all_num),
                    PathBuf::new(),
                );

                for i in 0..ips_num{

                    // Define the transition functions
                    let states = iPS_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "operation" => ["PASSAGE"],
                        "error" => [false]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );

                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }

                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

                for i in 0..normal_num{

                    // Define the transition functions
                    let states = normal_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "tag" => ["PASSAGE"]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );
        
                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }
        
                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
                maholo_simulator.append(&mut maholo_simulator2);

                let reagent_states = reagent_experiment_states();
                let shared_variable_history = DataFrame::empty();
                let mut reagent_experiment = Experiment::new(
                    REAGENT_EXPERIMENT_NAME.to_string(),
                    reagent_states,
                    1,
                    SharedVariableHistoryInput::DataFrame(shared_variable_history),
                );

                let new_task = reagent_experiment.generate_task_of_the_state();
                schedule.experiments.push(reagent_experiment);
                schedule.tasks.push(new_task);

                let initial_df = match df!(
                    "density" => [0.00], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut reagent_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.00, 0.00, 1.0, 0.00)); 1];
                maholo_simulator.append(&mut reagent_simulator);


                schedule.show_experiment_names_and_state_names();
                // unimplemented!("Implement the FIFO_scheduler");
                println!("tasks: {:?}", schedule.tasks);
                schedule.assign_task_id();
                println!("tasks with assigned task id: {:?}", schedule.tasks);
                match pattern {
                    &"SA" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    &"FIFO" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                        break;
                    },
                    _ => panic!(),
            }
        }





    }
}


#[test]
fn test_mix_SA_long_vs_FIFO_without_reagent_change() {
    // /home/cab314/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_mix_SA_vs_FIFO --exact --nocapture  > testcase/aaa.txt
    // Reset global time

    let patterns = ["SA", "FIFO"];
    let sim_num = 3;
    for pattern in patterns.iter(){
        for i in 0..sim_num{
            let global_time = 0;
            overwrtite_global_time_manualy(global_time);


            let ips_num = 5;
            let normal_num = 5;
            let all_num = ips_num + normal_num + 1;

            let dir = Path::new(RESULT_PATH);
            let dir = &dir.join("2024-04-15/sa_vs_fifo_without_reagent_chenge/mix").join(pattern).join(format!("sim_{}", i));
            println!("dir: {:?}", dir);
            match create_dir_all(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            let mut schedule = OneMachineExperimentManager::new(
                Vec::with_capacity(all_num),
                Vec::with_capacity(all_num),
                PathBuf::new(),
            );

            for i in 0..ips_num{

                // Define the transition functions
                let states = iPS_culture_experiment_states();
                let shared_variable_history = df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "operation" => ["PASSAGE"],
                    "error" => [false]
                ).unwrap();
                let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                let mut cell_culture_experiment = Experiment::new(
                    format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                    states,
                    2,
                    shared_variable_history,
                );

                let new_task = cell_culture_experiment.generate_task_of_the_state();
                schedule.experiments.push(cell_culture_experiment);
                schedule.tasks.push(new_task);
            }

            let initial_df = match df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

            for i in 0..normal_num{

                // Define the transition functions
                let states = normal_culture_experiment_states();
                let shared_variable_history = df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ).unwrap();
                let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                let mut cell_culture_experiment = Experiment::new(
                    format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                    states,
                    2,
                    shared_variable_history,
                );
    
                let new_task = cell_culture_experiment.generate_task_of_the_state();
                schedule.experiments.push(cell_culture_experiment);
                schedule.tasks.push(new_task);
            }
    
            let initial_df = match df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
            maholo_simulator.append(&mut maholo_simulator2);

            schedule.show_experiment_names_and_state_names();
            // unimplemented!("Implement the FIFO_scheduler");
            println!("tasks: {:?}", schedule.tasks);
            schedule.assign_task_id();
            println!("tasks with assigned task id: {:?}", schedule.tasks);
            match pattern {
                &"SA" => {
                    let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                    println!("schedule_task:");
                    for task in &schedule_task{
                        println!("{:?}", task);
                    }
                    // std::thread::sleep(std::time::Duration::from_secs(1000));
                    run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                },
                &"FIFO" => {
                    let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                    println!("schedule_task:");
                    for task in &schedule_task{
                        println!("{:?}", task);
                    }
                    // std::thread::sleep(std::time::Duration::from_secs(1000));
                    run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                    break;
                },
                _ => panic!(),
        }
    }





}
}





#[test]
fn test_mix_SA_long_vs_FIFO_with_reagent_change() {
    // /home/cab314/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_mix_SA_vs_FIFO --exact --nocapture  > testcase/aaa.txt
    // Reset global time

    let patterns = ["SA", "FIFO"];
    let sim_num = 3;
    for pattern in patterns.iter(){
        for i in 0..sim_num{
            let global_time = 0;
            overwrtite_global_time_manualy(global_time);


            let ips_num = 5;
            let normal_num = 5;
            let all_num = ips_num + normal_num + 1;

            let dir = Path::new(RESULT_PATH);
            let dir = &dir.join("2024-04-15/sa_vs_fifo_reagent_chenge/mix").join(pattern).join(format!("sim_{}", i));
            println!("dir: {:?}", dir);
            match create_dir_all(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            let mut schedule = OneMachineExperimentManager::new(
                Vec::with_capacity(all_num),
                Vec::with_capacity(all_num),
                PathBuf::new(),
            );

            for i in 0..ips_num{

                // Define the transition functions
                let states = iPS_culture_experiment_states();
                let shared_variable_history = df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "operation" => ["PASSAGE"],
                    "error" => [false]
                ).unwrap();
                let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                let mut cell_culture_experiment = Experiment::new(
                    format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                    states,
                    2,
                    shared_variable_history,
                );

                let new_task = cell_culture_experiment.generate_task_of_the_state();
                schedule.experiments.push(cell_culture_experiment);
                schedule.tasks.push(new_task);
            }

            let initial_df = match df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

            for i in 0..normal_num{

                // Define the transition functions
                let states = normal_culture_experiment_states();
                let shared_variable_history = df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ).unwrap();
                let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                let mut cell_culture_experiment = Experiment::new(
                    format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                    states,
                    2,
                    shared_variable_history,
                );
    
                let new_task = cell_culture_experiment.generate_task_of_the_state();
                schedule.experiments.push(cell_culture_experiment);
                schedule.tasks.push(new_task);
            }
    
            let initial_df = match df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
            maholo_simulator.append(&mut maholo_simulator2);

            let reagent_states = reagent_experiment_states();
            let shared_variable_history = DataFrame::empty();
            let mut reagent_experiment = Experiment::new(
                REAGENT_EXPERIMENT_NAME.to_string(),
                reagent_states,
                1,
                SharedVariableHistoryInput::DataFrame(shared_variable_history),
            );

            let new_task = reagent_experiment.generate_task_of_the_state();
            schedule.experiments.push(reagent_experiment);
            schedule.tasks.push(new_task);

            let initial_df = match df!(
                "density" => [0.00], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            let mut reagent_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.00, 0.00, 1.0, 0.00)); 1];
            maholo_simulator.append(&mut reagent_simulator);


            schedule.show_experiment_names_and_state_names();
            // unimplemented!("Implement the FIFO_scheduler");
            println!("tasks: {:?}", schedule.tasks);
            schedule.assign_task_id();
            println!("tasks with assigned task id: {:?}", schedule.tasks);
            match pattern {
                &"SA" => {
                    let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                    println!("schedule_task:");
                    for task in &schedule_task{
                        println!("{:?}", task);
                    }
                    // std::thread::sleep(std::time::Duration::from_secs(1000));
                    run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                },
                &"FIFO" => {
                    let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                    println!("schedule_task:");
                    for task in &schedule_task{
                        println!("{:?}", task);
                    }
                    // std::thread::sleep(std::time::Duration::from_secs(1000));
                    run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                    break;
                },
                _ => panic!(),
        }
    }





}
}




    #[test]
    fn test_mix_SA_long_vs_FIFO() {
        // /home/cab314/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_mix_SA_vs_FIFO --exact --nocapture  > testcase/aaa.txt
        // Reset global time

        let patterns = ["SA", "FIFO"];
        let sim_num = 3;
        for pattern in patterns.iter(){
            for i in 0..sim_num{
                let global_time = 0;
                overwrtite_global_time_manualy(global_time);


                let ips_num = 5;
                let normal_num = 5;
                let all_num = ips_num + normal_num;

                let dir = Path::new(RESULT_PATH);
                let dir = &dir.join("2024-04-10/sa_vs_fifo/mix").join(pattern).join(format!("sim_{}", i));
                println!("dir: {:?}", dir);
                match create_dir_all(&dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }

                let mut schedule = OneMachineExperimentManager::new(
                    Vec::with_capacity(all_num),
                    Vec::with_capacity(all_num),
                    PathBuf::new(),
                );

                for i in 0..ips_num{

                    // Define the transition functions
                    let states = iPS_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "operation" => ["PASSAGE"],
                        "error" => [false]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );

                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }

                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

                for i in 0..normal_num{

                    // Define the transition functions
                    let states = normal_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "tag" => ["PASSAGE"]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );
        
                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }
        
                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
                maholo_simulator.append(&mut maholo_simulator2);

                schedule.show_experiment_names_and_state_names();
                // unimplemented!("Implement the FIFO_scheduler");
                println!("tasks: {:?}", schedule.tasks);
                schedule.assign_task_id();
                println!("tasks with assigned task id: {:?}", schedule.tasks);
                match pattern {
                    &"SA" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    &"FIFO" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                        break;
                    },
                    _ => panic!(),
            }
        }





    }
}


    #[test]
    fn test_mix_SA_vs_FIFO() {
        //  time /Users/yuyaarai/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPS_normal_mixed --exact --nocapture > testcase/aaa.txt
        // Reset global time

        let patterns = ["SA", "FIFO"];
        let sim_num = 10;
        for pattern in patterns.iter(){
            for i in 0..sim_num{
                let global_time = 0;
                overwrtite_global_time_manualy(global_time);


                let ips_num = 5;
                let normal_num = 5;
                let all_num = ips_num + normal_num;

                let dir = Path::new(RESULT_PATH);
                let dir = &dir.join("2024-03-26/sa_vs_fifo/mix").join(pattern).join(format!("sim_{}", i));
                println!("dir: {:?}", dir);
                match create_dir_all(&dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }

                let mut schedule = OneMachineExperimentManager::new(
                    Vec::with_capacity(all_num),
                    Vec::with_capacity(all_num),
                    PathBuf::new(),
                );

                for i in 0..ips_num{

                    // Define the transition functions
                    let states = iPS_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "operation" => ["PASSAGE"],
                        "error" => [false]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );

                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }

                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

                for i in 0..normal_num{

                    // Define the transition functions
                    let states = normal_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "tag" => ["PASSAGE"]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );
        
                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }
        
                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
                maholo_simulator.append(&mut maholo_simulator2);

                schedule.show_experiment_names_and_state_names();
                // unimplemented!("Implement the FIFO_scheduler");
                println!("tasks: {:?}", schedule.tasks);
                schedule.assign_task_id();
                println!("tasks with assigned task id: {:?}", schedule.tasks);
                match pattern {
                    &"SA" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    &"FIFO" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    _ => panic!(),
            }
        }





    }
}

    #[test]
    fn test_iPS_SA_vs_FIFO() {
        //  time /Users/yuyaarai/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPS_normal_mixed --exact --nocapture > testcase/aaa.txt
        // Reset global time

        let patterns = ["SA", "FIFO"];
        let sim_num = 10;
        for pattern in patterns.iter(){
            for i in 0..sim_num{
                let global_time = 0;
                overwrtite_global_time_manualy(global_time);


                let ips_num = 10;

                let dir = Path::new(RESULT_PATH);
                let dir = &dir.join("2024-03-26/sa_vs_fifo/ips").join(pattern).join(format!("sim_{}", i));
                println!("dir: {:?}", dir);
                match create_dir_all(&dir){
                    Ok(_) => (),
                    Err(err) => println!("{}", err),
                }

                let mut schedule = OneMachineExperimentManager::new(
                    Vec::with_capacity(ips_num),
                    Vec::with_capacity(ips_num),
                    PathBuf::new(),
                );

                for i in 0..ips_num{

                    // Define the transition functions
                    let states = iPS_culture_experiment_states();
                    let shared_variable_history = df!(
                        "density" => [0.05], 
                        "time" => [0.0],
                        "operation" => ["PASSAGE"],
                        "error" => [false]
                    ).unwrap();
                    let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
                    let mut cell_culture_experiment = Experiment::new(
                        format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                        states,
                        2,
                        shared_variable_history,
                    );

                    let new_task = cell_culture_experiment.generate_task_of_the_state();
                    schedule.experiments.push(cell_culture_experiment);
                    schedule.tasks.push(new_task);
                }

                let initial_df = match df!(
                    "density" => [0.05], 
                    "time" => [0.0],
                    "tag" => ["PASSAGE"]
                ) {
                    Ok(it) => it,
                    Err(err) => panic!("{}", err),
                };
                let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

                schedule.show_experiment_names_and_state_names();
                // unimplemented!("Implement the FIFO_scheduler");
                println!("tasks: {:?}", schedule.tasks);
                schedule.assign_task_id();
                println!("tasks with assigned task id: {:?}", schedule.tasks);
                match pattern {
                    &"SA" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    &"FIFO" => {
                        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
                        println!("schedule_task:");
                        for task in &schedule_task{
                            println!("{:?}", task);
                        }
                        // std::thread::sleep(std::time::Duration::from_secs(1000));
                        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);
                    },
                    _ => panic!(),
            }
        }





    }
}

    #[test]
    fn test_iPS_normal_mixed_SA() {
        //  time /Users/yuyaarai/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPS_normal_mixed --exact --nocapture > testcase/aaa.txt
        // Reset global time
        let global_time = 0;
        overwrtite_global_time_manualy(global_time);


        let ips_num = 5;
        let normal_num = 5;
        let all_num = ips_num + normal_num;

        let dir = Path::new(RESULT_PATH);
        let dir = &dir.join("2024-03-26/sa_vs_fifo/mix/SA");
        println!("dir: {:?}", dir);
        match create_dir_all(&dir){
            Ok(_) => (),
            Err(err) => println!("{}", err),
        }

        let mut schedule = OneMachineExperimentManager::new(
            Vec::with_capacity(all_num),
            Vec::with_capacity(all_num),
            PathBuf::new(),
        );

        for i in 0..ips_num{

            // Define the transition functions
            let states = iPS_culture_experiment_states();
            let shared_variable_history = df!(
                "density" => [0.05], 
                "time" => [0.0],
                "operation" => ["PASSAGE"],
                "error" => [false]
            ).unwrap();
            let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
            let mut cell_culture_experiment = Experiment::new(
                format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                states,
                2,
                shared_variable_history,
            );

            let new_task = cell_culture_experiment.generate_task_of_the_state();
            schedule.experiments.push(cell_culture_experiment);
            schedule.tasks.push(new_task);
        }

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

        for i in 0..normal_num{

            // Define the transition functions
            let states = normal_culture_experiment_states();
            let shared_variable_history = df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ).unwrap();
            let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
            let mut cell_culture_experiment = Experiment::new(
                format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                states,
                2,
                shared_variable_history,
            );

            let new_task = cell_culture_experiment.generate_task_of_the_state();
            schedule.experiments.push(cell_culture_experiment);
            schedule.tasks.push(new_task);
        }

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
        maholo_simulator.append(&mut maholo_simulator2);


        schedule.show_experiment_names_and_state_names();
        // unimplemented!("Implement the FIFO_scheduler");
        println!("tasks: {:?}", schedule.tasks);
        schedule.assign_task_id();
        println!("tasks with assigned task id: {:?}", schedule.tasks);
        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::simulated_annealing_scheduler_absolute(schedule.tasks.clone());
        println!("schedule_task:");
        for task in &schedule_task{
            println!("{:?}", task);
        }
        // std::thread::sleep(std::time::Duration::from_secs(1000));
        run_simulation_SA(schedule_task, schedule, maholo_simulator, 1000, dir);



    }

    #[test]
    fn test_iPS_normal_mixed_FIFO() {
        //  time /Users/yuyaarai/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPS_normal_mixed --exact --nocapture > testcase/aaa.txt
        // Reset global time
        let global_time = 0;
        overwrtite_global_time_manualy(global_time);


        let ips_num = 5;
        let normal_num = 5;
        let all_num = ips_num + normal_num;

        let dir = Path::new(RESULT_PATH);
        let dir = &dir.join("2024-03-28/volatile/mix/FIFO");
        match remove_dir_all(&dir){
            Ok(_) => (),
            Err(err) => println!("{}", err),
        }
        std::thread::sleep(std::time::Duration::from_secs(10));
        match create_dir_all(&dir){
            Ok(_) => (),
            Err(err) => println!("{}", err),
        }

        let mut schedule = OneMachineExperimentManager::new(
            Vec::with_capacity(all_num),
            Vec::with_capacity(all_num),
            PathBuf::new(),
        );

        for i in 0..ips_num{

            // Define the transition functions
            let states = iPS_culture_experiment_states();
            let shared_variable_history = df!(
                "density" => [0.05], 
                "time" => [0.0],
                "operation" => ["PASSAGE"],
                "error" => [false]
            ).unwrap();
            let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
            let mut cell_culture_experiment = Experiment::new(
                format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                states,
                2,
                shared_variable_history,
            );

            let new_task = cell_culture_experiment.generate_task_of_the_state();
            schedule.experiments.push(cell_culture_experiment);
            schedule.tasks.push(new_task);
        }

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];

        for i in 0..normal_num{

            // Define the transition functions
            let states = normal_culture_experiment_states();
            let shared_variable_history = df!(
                "density" => [0.05], 
                "time" => [0.0],
                "tag" => ["PASSAGE"]
            ).unwrap();
            let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
            let mut cell_culture_experiment = Experiment::new(
                format!("{}_{}", NORMAL_EXPERIMENT_NAME.to_string(), i),
                states,
                2,
                shared_variable_history,
            );

            let new_task = cell_culture_experiment.generate_task_of_the_state();
            schedule.experiments.push(cell_culture_experiment);
            schedule.tasks.push(new_task);
        }

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator2 = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.0012, 0.05, 1.0, 0.05)); normal_num];
        maholo_simulator.append(&mut maholo_simulator2);


        schedule.show_experiment_names_and_state_names();
        // unimplemented!("Implement the FIFO_scheduler");
        println!("tasks: {:?}", schedule.tasks);
        schedule.assign_task_id();
        println!("tasks with assigned task id: {:?}", schedule.tasks);
        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
        println!("schedule_task:");
        for task in &schedule_task{
            println!("{:?}", task);
        }
        // std::thread::sleep(std::time::Duration::from_secs(1000));
        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);



    }



    
    #[test]
    fn test_iPSs() {
        // /Users/yuyaarai/.cargo/bin/cargo test -r --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPSs --exact --nocapture
        // /home/cab314/.cargo/bin/cargo test --package ExperimentManagementSystem --bin ExperimentManagementSystem -- ccds::tests::test_iPSs --exact --nocapture 
        // Reset global time
        let global_time = 0;
        overwrtite_global_time_manualy(global_time);


        let ips_num = 10;

        let dir = Path::new("testcase/volatile");
        match create_dir(&dir){
            Ok(_) => (),
            Err(err) => println!("{}", err),
        }

        let mut schedule = OneMachineExperimentManager::new(
            Vec::with_capacity(ips_num),
            Vec::with_capacity(ips_num),
            PathBuf::new(),
        );

        for i in 0..ips_num{

            // Define the transition functions
            let states = iPS_culture_experiment_states();
            let shared_variable_history = df!(
                "density" => [0.05], 
                "time" => [0.0],
                "operation" => ["PASSAGE"],
                "error" => [false]
            ).unwrap();
            let shared_variable_history = SharedVariableHistoryInput::DataFrame(shared_variable_history);
            let mut cell_culture_experiment = Experiment::new(
                format!("{}_{}", IPS_EXPERIMENT_NAME.to_string(), i),
                states,
                2,
                shared_variable_history,
            );

            let new_task = cell_culture_experiment.generate_task_of_the_state();
            schedule.experiments.push(cell_culture_experiment);
            schedule.tasks.push(new_task);
        }

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator = vec![Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)); ips_num];


        schedule.show_experiment_names_and_state_names();
        // unimplemented!("Implement the FIFO_scheduler");
        println!("tasks: {:?}", schedule.tasks);
        schedule.assign_task_id();
        println!("tasks with assigned task id: {:?}", schedule.tasks);
        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler_absolute(schedule.tasks.clone());
        println!("schedule_task:");
        for task in &schedule_task{
            println!("{:?}", task);
        }
        // std::thread::sleep(std::time::Duration::from_secs(1000));
        run_simulation_fifo(schedule_task, schedule, maholo_simulator, 1000, dir);



    }


    
    #[test]
    fn test_CCDS() {

        // Reset global time
        let global_time = 0;
        overwrtite_global_time_manualy(global_time);

        
        let dir = Path::new("testcase/volatile");
        // Define the transition functions
        let states = iPS_culture_experiment_states();
    
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
        println!("current time before the simulation: {}", get_current_absolute_time());


        let mut schedule = OneMachineExperimentManager::new(
            vec![cell_culture_experiment],
            vec![new_task],
            std::path::PathBuf::new(),
        );

        let initial_df = match df!(
            "density" => [0.05], 
            "time" => [0.0],
            "tag" => ["PASSAGE"]
        ) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };
        let mut maholo_simulator = vec![
            Simulator::new(initial_df, NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05)),
        ];

        

        println!("schedule.tasks: {:?}", schedule.tasks);


        schedule.show_experiment_names_and_state_names();
        let mut schedule_task = crate::task_scheduler::one_machine_schedule_solver::FIFO_scheduler(schedule.tasks.clone());
        println!("schedule_task: {:?}", schedule_task);

        println!("Test the simulator --------------------------\n\n");
        for step in 0..100 {
            println!("step: {}=================", step);
            println!("Current state name: {}", schedule.experiments[0].states[schedule.experiments[0].current_state_index].state_name);
            println!("Current history: {}", schedule.experiments[0].shared_variable_history);
            println!("Current time: {}", get_current_absolute_time());

            let step_dir = dir.join(format!("step_{}", step));

            match create_dir(&step_dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }

            // // Get the earliest task
            let (task_id, new_result_of_experiment, update_type) = SimpleTaskSimulator::new(schedule_task.clone())
            .process_earliest_task(&schedule, &mut maholo_simulator);

            println!("task_id: {}, new_result_of_experiment: {:?}, update_type: {}", task_id, new_result_of_experiment, update_type);
            schedule_task = schedule.update_state_and_reschedule(task_id, new_result_of_experiment, update_type, 'f');
            


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
        }

        // Save the simulation result df
        for experiment_index in 0..schedule.experiments.len(){
            let mut experiment = &mut schedule.experiments[experiment_index];
            let name = experiment.experiment_name.clone();
            // Make the directory
            let dir = dir.join(name);
            match create_dir(&dir){
                Ok(_) => (),
                Err(err) => println!("{}", err),
            }
            let mut file = std::fs::File::create(dir.join("simulation_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut experiment.shared_variable_history)
                .unwrap();
            let mut file = std::fs::File::create(dir.join("simulator_result.csv")).unwrap();
            CsvWriter::new(&mut file)
                .finish(&mut maholo_simulator[experiment_index].cell_history().clone())
                .unwrap();    
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
            Ok(10)  
        };
        let timing_function_state_0 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::None))  
        };
        let transition_func_state_0 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(0)  
        };
    

        let experiment_operation_state_1 = "PASSAGE".to_string();
        let processing_time_function_state_1 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(10)  
        };
        let timing_function_state_1 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((0, PenaltyType::Linear { coefficient: 1 }))
        };
        let transition_func_state_1 = |variable_history: &mut DataFrame| -> Result<StateIndex, Box<dyn Error>> {
            Ok(2)  
        };


        let experiment_operation_state_2 = "GET_IMAGE".to_string();
        let processing_time_function_state_2 = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
            Ok(10)  
        };
        let timing_function_state_2 = |variable_history: &DataFrame| -> Result<(OptimalTiming, PenaltyType), Box<dyn Error>> {
            Ok((24*60, PenaltyType::Linear { coefficient: 1 }))
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
                .select([col("density").cast(DataType::Float64)])
                ;
            let latest_density = q.collect().unwrap();
            let latest_density = latest_density.column("density").unwrap().get(0).unwrap();
            let latest_density = match latest_density{
                AnyValue::Float64(it) => it,
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

        let mut schedule_task = schedule.update_state_and_reschedule(0, new_result_of_experiment, 'a', 'f');
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

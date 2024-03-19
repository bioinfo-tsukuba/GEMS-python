use std::{arch::global_asm, fs::File, io::{self, BufReader, Read, Write}, os::unix::{process, raw::uid_t}, path::Path, vec};

use polars::{functions::concat_df_diagonal, lazy::dsl::max, prelude::*};
use polars::lazy::dsl::col;

use crate::{ccds::IPS_CULTURE_STATE_NAMES, common_param_type::{self, get_current_absolute_time}};

use super::{overwrtite_global_time_manualy, ScheduledTask, TaskId};

#[derive(Debug, Clone)]
pub(crate) struct Simulator{
    cell_history: DataFrame,
    normal_cell_simulator: NormalCellSimulator,
}

impl Simulator {
    pub(crate) fn new(cell_history: DataFrame, normal_cell_simulator: NormalCellSimulator) -> Self {
        Self {
            cell_history,
            normal_cell_simulator,
        }
    }

    /// Simulate cell growth
    /// 1. Get the current time
    /// 2. Get the initial time and density, passaged time and density
    /// 3. Calculate the delta time
    /// 4. Simulate the cell growth
    /// 5. Update the cell_history
    pub(crate) fn simulate(&mut self) {
        let current_time = super::get_current_absolute_time();
        let q = self.cell_history
                .clone()
                .lazy()
                .filter(
                    col("time").is_not_null()
                    .and(col("density").is_not_null())
                    .and(col("tag").eq(lit("PASSAGE")))
                )
                .select(&[col("time"), col("density")])
                .sort("time", SortOptions{
                    descending: true,
                    nulls_last: false,
                    multithreaded: true,
                    maintain_order: false,
                    }
                )
                .limit(1)
                ;
        let df = q.collect().unwrap();
        println!("criteria :{:?}", df);
        let start_time = df["time"].f64().unwrap().get(0).unwrap();
        let density = df["density"].f64().unwrap().get(0).unwrap();
        let delta_time = current_time as f64 - start_time;
        println!("delta_time: {}", delta_time);

        let mut normal_cell_simulator = self.normal_cell_simulator.clone();
        normal_cell_simulator.n0 = density as f32;
        let current_cell_density = normal_cell_simulator.logistic_function(delta_time as f32);

        // Update the cell_history
        let simulate_log = df!(
            "time" => [current_time as f64], 
            "density" => [current_cell_density as f64], 
            "tag" => ["SIMULATE"]
        ).unwrap();
        self.cell_history = concat_df_diagonal(&[
            self.cell_history.clone(),
            simulate_log,
        ]).unwrap();

        self.normal_cell_simulator.current_cell_density = current_cell_density;

    }

    pub(crate) fn process_earliest_task(&mut self, one_machine_experiment: &super::OneMachineExperimentManager) -> (TaskId, DataFrame, char) {
        todo!()
    }
    
    pub(crate) fn cell_history(&self) -> &DataFrame {
        &self.cell_history
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CellHistory {
    pub(crate) absolute_time: common_param_type::OptimalTiming,
    pub(crate) cell_density: f32,
    pub(crate) cell_id: usize,
    pub(crate) passage: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct NormalCellSimulator {
    id: usize,
    r: f32,
    n0: f32,
    k: f32,
    current_cell_density: f32,
}

impl NormalCellSimulator {
    pub(crate) fn new(id: usize, r: f32, n0: f32, k: f32, current_cell_density: f32) -> Self {
        Self {
            id,
            r,
            n0,
            k,
            current_cell_density
        }
    }

    /// Calculate value of t(passed time) from cell number
    pub(crate) fn calculate_logistic_rev(&self, cell_density: f32) -> f32{
        let current_time =
        ((self.k-self.n0).ln() - self.n0.ln() + cell_density.ln() - (self.k - cell_density).ln())/self.r;
        current_time
    }

    fn logistic_function(&self, x: f32) -> f32 {
        let r = self.r;
        let N0 = self.n0;
        let K = self.k;
        K / (1.0 + (K / N0 - 1.0) * (-r * x).exp())
    }

    pub(crate) fn simulate_cell_growth(&self, delta_time: f32, current_cell_density: f32) -> f32 {
        let current_time = self.calculate_logistic_rev(current_cell_density);
        let time = current_time + delta_time;

        println!("Estimated time: {} + delta_time: {} = {}", current_time, delta_time, time);
        self.logistic_function(time)
    }
}

pub(crate) struct SimpleTaskSimulator{
    scheduled_tasks: Vec<ScheduledTask>,
}

impl SimpleTaskSimulator {
    pub(crate) fn new(scheduled_tasks: Vec<ScheduledTask>) -> Self {
        Self {
            scheduled_tasks,
        }
    }

    /// The time unit is minute
    pub(crate) fn put_a_clock_forward(&self, time: i64) {
        match time {
            0.. => println!("put a clock forward: {} minutes", time),
            _ => todo!("Found bug that causes the minus scheduled task!"),
        }
        let mut global_time = get_current_absolute_time();
        global_time += time;
        overwrtite_global_time_manualy(global_time);
    }

    pub(crate) fn simulate(&self, delta_time: i64, simulators:&mut Vec<Simulator>) {
        println!("================SIMULATION================");
        for simulator in simulators {
            simulator.simulate();
            println!("{:?}", simulator);
        }
    }

    pub(crate) fn process_earliest_task(
        &mut self, 
        one_machine_experiment: &super::OneMachineExperimentManager, 
        simulators: &mut Vec<Simulator>
    ) -> (TaskId, DataFrame, char) {
        let mut simulation_dir = one_machine_experiment.dir.clone();
        let output_dir = simulation_dir.join("simulation");
        println!("********Processing the earliest task********");
        // Sort the scheduled_tasks by schedule_timing
        self.scheduled_tasks.sort_by(|a, b| a.schedule_timing.cmp(&b.schedule_timing));
        // Get the earliest task, without removing it
        let earliest_task = self.scheduled_tasks.first().unwrap();
        println!("Earliest task: {:?}", earliest_task);
        // Process the earliest task
        let task_id = earliest_task.task_id;
        let earliest_task_uuid = earliest_task.experiment_uuid.clone();
        let experiment_index_of_earliest_task = one_machine_experiment.experiments.iter().position(|experiment| experiment.experiment_uuid == earliest_task_uuid).unwrap();
        let mut new_result_of_experiment: DataFrame = DataFrame::empty();
        let mut update_type: char = 'a';

        // Put a clock forward to the time of the earliest task
        let mut earliest_task_time = earliest_task.schedule_timing;
        let delta_time = earliest_task_time as i64 - get_current_absolute_time();
        self.put_a_clock_forward(delta_time);

        // Simulate cell growth
        self.simulate(delta_time, simulators);


        let operation_name = earliest_task.experiment_operation.as_str();
        if operation_name == IPS_CULTURE_STATE_NAMES[0] {
            // EXPIRE
            // Implement the processing of the task
            unimplemented!("Implement the processing of the task EXPIRE")
        } else if operation_name == IPS_CULTURE_STATE_NAMES[1] {
            // PASSAGE special tag
            let simulate_log = df!(
                "time" => [get_current_absolute_time() as f64], 
                "density" => [0.05 as f64], 
                "tag" => ["PASSAGE"]
            ).unwrap();
            simulators[experiment_index_of_earliest_task].cell_history = concat_df_diagonal(&[
                simulators[experiment_index_of_earliest_task].cell_history.clone(),
                simulate_log,
            ]).unwrap();
            println!("***SPECIAL TASK*** Passage occurs in {:?}", simulators[experiment_index_of_earliest_task]);

            
            new_result_of_experiment = match df!(
                "density" => [0.05 as f64], 
                ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            update_type = 'a';
        } else if operation_name == IPS_CULTURE_STATE_NAMES[2] {
            // GET_IMAGE_1
            new_result_of_experiment = match df!(
                "density" => [simulators[experiment_index_of_earliest_task].normal_cell_simulator.current_cell_density as f64], 
            ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            update_type = 'a';
        } else if operation_name == IPS_CULTURE_STATE_NAMES[3] {
            // MEDIUM_CHANGE_1
            update_type = 'a';
        } else if operation_name == IPS_CULTURE_STATE_NAMES[4] {
            // GET_IMAGE_2
            new_result_of_experiment = match df!(
                "density" => [simulators[experiment_index_of_earliest_task].normal_cell_simulator.current_cell_density as f64], 
                ) {
                Ok(it) => it,
                Err(err) => panic!("{}", err),
            };
            update_type = 'a';
        } else if operation_name == IPS_CULTURE_STATE_NAMES[5] {
            // MEDIUM_CHANGE_2
            update_type = 'a';
        } else if operation_name == IPS_CULTURE_STATE_NAMES[6] {
            // PLATE_COATING
            update_type = 'a';
        } else {
            unreachable!();
        }

        // Add "operation" column to new_result_of_experiment, and fill it with the operation_name
        let column_length = new_result_of_experiment.height().max(1);
        let operation_column = Series::new("operation", vec![operation_name; column_length]);
        let time_column = Series::new("time", vec![get_current_absolute_time() as f64; column_length]);
        let error_column = Series::new("error", vec![false; column_length]);
        new_result_of_experiment.with_column(operation_column).unwrap();
        new_result_of_experiment.with_column(time_column).unwrap();
        new_result_of_experiment.with_column(error_column).unwrap();


        let processing_time = earliest_task.processing_time;
        println!("Processed task: {:?}", earliest_task);
        println!("processing_time: {processing_time}");
        println!("Now:{}", common_param_type::get_current_absolute_time());
        self.put_a_clock_forward(processing_time);
        (task_id, new_result_of_experiment, update_type)
    }

    fn update_global_time(&self, processing_time: i64){
        self.put_a_clock_forward(processing_time);
    }
}


// Test
#[cfg(test)]
mod tests {
    use serde::Serialize;

    use crate::common_param_type::{try_read_tsv_struct, write_struct};

    use super::*;

    // Test NormalCellSimulator
    #[test]
    fn test_get_time(){
        let chrono_time = chrono::Utc::now();
        let time = chrono_time.timestamp();
        println!("{:?}", time);
        println!("{:?}", chrono_time.to_rfc3339());
        let rfc_3339 = chrono_time.to_rfc3339();
        let chrono_time = chrono::DateTime::parse_from_rfc3339(&rfc_3339).unwrap();
        println!("Check: {:?}", chrono_time.timestamp());

        // Wait for 1 second
        std::thread::sleep(std::time::Duration::from_secs(60));

        let chrono_time_after = chrono::Utc::now();
        let time_after = chrono_time_after.timestamp();
        println!("{:?}", time_after);

        

        // YYYY-MM-DD HH:MM:SS
        println!("{}", chrono_time);

        // Output chrono_time as a file with deserialize
        let output_path = Path::new("test_time.txt");
        // https://rust-lang-nursery.github.io/rust-cookbook/datetime/parse.html

    }

    #[test]
    fn create_test_params(){
        let mut normal_cell_simulators = vec![
            NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05),
            NormalCellSimulator::new(1, 0.000252219650877879, 0.05, 1.0, 0.05),
            NormalCellSimulator::new(2, 0.000252219650877879, 0.05, 1.0, 0.05),
            NormalCellSimulator::new(3, 0.000252219650877879, 0.05, 1.0, 0.05),
            ];
        let output_path = Path::new("/Users/yuyaarai/Downloads/ExperimentManagementSystem/testcase/cell_params");
        write_struct(&output_path, &normal_cell_simulators);

        let mut normal_cell_simulators_check = 
            try_read_tsv_struct::<NormalCellSimulator>(&output_path).unwrap();


        for normal_cell_simulator in normal_cell_simulators_check {
            println!("{:?}", normal_cell_simulator);
        }


    }

    #[test]
    fn test_simulate(){
        let passed_time = 6*24*60;
        overwrtite_global_time_manualy(passed_time);
        let cell_history = df!(
            "time" => [0.0, ],
            "density" => [0.05, ],
            "tag" => ["PASSAGE", ]
        ).unwrap();
        let normal_cell_simulator = NormalCellSimulator::new(0, 0.000252219650877879, 0.05, 1.0, 0.05);
        let mut simulator = Simulator::new(cell_history, normal_cell_simulator);
        simulator.simulate();
        println!("{:?}", simulator.cell_history);
    }

    #[test]
    fn test_get_initial_density(){
        let cell_history = df!(
            "time" => [0.0, 1.0, 2.0, 3.0, 4.0],
            "density" => [0.05, 0.1, 0.2, 0.3, 0.4],
            "operation" => ["PASSAGE", "PASSAGE", "PASSAGE", "PASSAGE", "GGGGG"]
        ).unwrap();

        println!("{:?}", cell_history);
        let q = cell_history
                .clone()
                .lazy()
                .filter(
                    col("time").is_not_null()
                    .and(col("density").is_not_null())
                    .and(col("operation").eq(lit("PASSAGE")))
                )
                .select(&[col("time"), col("density")])
                .sort("time", SortOptions{
                    descending: true,
                    nulls_last: false,
                    multithreaded: true,
                    maintain_order: false,
                    }
                )
                .limit(1)
                ;
        let df = q.collect().unwrap();

        let time = df["time"].f64().unwrap().get(0).unwrap();
        let density = df["density"].f64().unwrap().get(0).unwrap();
        println!("{:?}", df);

        println!("Time: {}, Density: {}", time, density);
    }
}
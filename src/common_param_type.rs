use std::{collections::HashMap, error::Error, fs::{self, File}, io::Write, os::unix::process, path::Path};

use polars::{datatypes::DataType, df, frame::DataFrame, series::Series};
use regex::Regex;
use serde::{Deserialize, Serialize};

pub static GLOBAL_TIME_PATH: &str = "/Users/yuyaarai/Downloads/ExperimentManagementSystem/testcase/global_time";

// 共通のパラメータ型を定義
pub type StateIndex = usize;
pub type StateName = String;
pub type ExperimentIndex = usize;
pub type ExperimentName = String;

pub type OptimalTiming = i64;
pub type ScheduleTiming = OptimalTiming;
pub type PenaltyParameter = OptimalTiming;
pub type ProcessingTime = OptimalTiming;

/// TaskResult is the data structure for data frame.
/// HashMap< Column name, data>
/// The data should have the same length
pub type TaskResult = DataFrame;

pub type ExperimentOperation = String;
pub type TaskId = usize;
#[derive(Debug, Clone, PartialEq)]
pub struct Task{
    pub optimal_timing: OptimalTiming,
    pub processing_time: ProcessingTime,
    pub penalty_type: PenaltyType,
    pub experiment_operation: ExperimentOperation,
    pub experiment_name: ExperimentName,
    pub task_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct  ScheduledTask {
    pub optimal_timing: OptimalTiming,
    pub processing_time: ProcessingTime,
    pub penalty_type: PenaltyType,
    pub experiment_operation: ExperimentOperation,
    pub experiment_name: ExperimentName,
    pub schedule_timing: ScheduleTiming,
    pub task_id: usize,
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct  ScheduledTaskSerde {
    pub optimal_timing: OptimalTiming,
    pub processing_time: ProcessingTime,
    pub penalty_type: String,
    pub experiment_operation: ExperimentOperation,
    pub experiment_name: ExperimentName,
    pub schedule_timing: ScheduleTiming,
    pub task_id: usize,
}

pub(crate) fn scheduled_task_convert_to_csv(output_path: &Path, tasks: &Vec<ScheduledTask>)-> Result<String, Box<dyn Error>> {
    // Convert ScheduledTask to ScheduledTaskSerde
    let mut items = Vec::new();
    for i in 0..tasks.len(){
        items.push(
            ScheduledTaskSerde{
                optimal_timing: tasks[i].optimal_timing,
                processing_time: tasks[i].processing_time,
                penalty_type: tasks[i].penalty_type.to_string_format(),
                experiment_operation: tasks[i].experiment_operation.clone(),
                experiment_name: tasks[i].experiment_name.clone(),
                schedule_timing: tasks[i].schedule_timing,
                task_id: tasks[i].task_id,
            }
        )
    }

    write_struct(&output_path, &items)

}

pub(crate) fn read_scheduled_task(path: &Path) -> Result<Vec<ScheduledTask>, Box<dyn Error>> {
    let scheduled_task_serde: Vec<ScheduledTaskSerde> = try_read_tsv_struct(path)?;
    let mut scheduled_tasks = Vec::new();
    for i in 0..scheduled_task_serde.len(){
        scheduled_tasks.push(
            ScheduledTask{
                optimal_timing: scheduled_task_serde[i].optimal_timing,
                processing_time: scheduled_task_serde[i].processing_time,
                penalty_type: PenaltyType::from_string_format(&scheduled_task_serde[i].penalty_type),
                experiment_operation: scheduled_task_serde[i].experiment_operation.clone(),
                experiment_name: scheduled_task_serde[i].experiment_name.clone(),
                schedule_timing: scheduled_task_serde[i].schedule_timing,
                task_id: scheduled_task_serde[i].task_id,
            }
        )
    }
    println!("Read the schedule: {:?}", path);
    Ok(scheduled_tasks)

}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyType {
    /// 0: None
    None,
    /// 1: Linear, let an earliness or a delay be "diff" (ScheduleTiming - OptimalTiming), then the penalty is abs(diff * coefficient)
    Linear(PenaltyParameter),

    /// 2: LinearWithRange, let an earliness or a delay be "diff", 
    /// then the penalty of LinearWithRange(lower, lower_coefficient, upper, upper_coefficient) is
    /// if diff < lower, then (lower-diff) * lower_coefficient
    /// if lower <= diff <= upper, then 0
    /// if upper < diff, then (diff-upper) * upper_coefficient
    LinearWithRange(PenaltyParameter, PenaltyParameter, PenaltyParameter, PenaltyParameter),
    
}

impl PenaltyType {
    pub fn get_penalty(&self, diff:PenaltyParameter ) -> PenaltyParameter {
        match self {
            PenaltyType::None => 0,
            PenaltyType::Linear(coefficient) => diff * coefficient,
            PenaltyType::LinearWithRange(lower, lower_coefficient, upper, upper_coefficient) => {
                if &diff < lower {
                    (lower-diff) * lower_coefficient
                } else if lower <= &diff && &diff <= upper {
                    0
                } else {
                    (diff-upper) * upper_coefficient
                }
            },
            _ => panic!("The penalty type is not supported."),
        }
    }

    /// Convert the penalty type to a string format
    pub fn to_string_format(&self) -> String {
        match self {
            PenaltyType::None => "None()".to_string(),
            PenaltyType::Linear(coefficient) => format!("Linear({})", coefficient),
            PenaltyType::LinearWithRange(lower, lower_coefficient, upper, upper_coefficient) => format!("LinearWithRange({},{},{},{})", lower, lower_coefficient, upper, upper_coefficient),
            _ => panic!("The penalty type is not supported."),
        }
    }

    ///Split the string by "title" and "parameters", e.g., "Linear(1)" -> "Linear" and [1]
    fn parse_param_string(param_string: &str) -> Result<(String, Vec<PenaltyParameter>), Box<dyn Error>> {
        let re = Regex::new(r"(\w+)\((.*)\)").unwrap();
        let captures = re.captures(param_string).unwrap();
        let title = captures.get(1).unwrap().as_str();
        let parameters = captures.get(2).unwrap().as_str();

        // Convert the parameters to Vec<PenaltyParameter>
        // e.g., "1,2,3,4" -> [1,2,3,4]

        // None does not have parameters
        if title == "None" {
            return Ok((title.to_string(), vec![]));
        }
        let parameters: Vec<PenaltyParameter> = parameters.split(",").map(|x| x.parse::<PenaltyParameter>().unwrap()).collect();

        Ok((title.to_string(), parameters))
    }

    /// Convert a string format to the penalty type
    pub fn from_string_format(penalty_type_string: &str) -> Self {
        let (title, parameters) = Self::parse_param_string(penalty_type_string).unwrap();
        match title.as_str() {
            "None" => PenaltyType::None,
            "Linear" => PenaltyType::Linear(parameters[0]),
            "LinearWithRange" => PenaltyType::LinearWithRange(parameters[0], parameters[1], parameters[2], parameters[3]),
            _ => panic!("The penalty type is not supported."),
        }
    }


}


pub struct Protocol {
    pub protocol_name: String,
    pub protocol_penalties: Vec<f64>,
}


pub(crate) fn try_read_tsv_struct<T: for<'de> Deserialize<'de>>(file_path: &Path) -> Result<Vec<T>, Box<dyn Error>> {
    let tsv_text = fs::read_to_string(&file_path)?;
    let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(tsv_text.as_bytes());
    let mut data_list = Vec::new();
    
    for result in rdr.deserialize() {
        let record: T = result?;
        data_list.push(record);
    }

    Ok(data_list)
}

pub(crate) fn write_struct<T: Serialize>(output_path: &Path, items: &Vec<T>) -> Result<String, Box<dyn Error>> {
    let path = Path::new(output_path);
    let mut wtr = csv::WriterBuilder::new().delimiter(b'\t').from_path(&path)?;

    for item in items {
        wtr.serialize(item)?;
    }
    wtr.flush()?;
    Ok(path.to_str().unwrap().to_string())
}

// jsonファイル書き出し
pub(crate) fn output_json<T: Serialize>(output_fn: &str, written_struct: Vec<T>) -> std::io::Result<()> {
    // シリアライズ
    let serialized: String = serde_json::to_string(&written_struct).unwrap();

    // ファイル出力
    let mut file = File::create(output_fn)?;
    file.write_all(serialized.as_bytes())?;
    Ok(())
}

pub(crate) fn overwrtite_global_time_manualy(current_time: i64){
    // Global time in the file, GLOBAL_TIME_PATH, is rewritten manually.

    // Get the global time as i64
    let global_time = fs::read_to_string(GLOBAL_TIME_PATH).unwrap();
    println!("global time: {:?}", global_time);
    let global_time: i64 = chrono::DateTime::parse_from_rfc3339(&global_time).unwrap().timestamp();
    println!("global time: {:?}", global_time);
    let elapsed_time = current_time - global_time;

    // current_time(i64, timestamp) -> chrono::DateTime
    let current_time = chrono::TimeZone::timestamp_opt(&chrono::Utc, current_time, 0);
    println!("overwrite time: {:?}", current_time);

    let current_time = current_time.unwrap().to_rfc3339();

    // Elapsed time
    
    println!("Elapsed time: {}", elapsed_time);

    // Rewrite the global time
    let mut file = File::create(GLOBAL_TIME_PATH).unwrap();
    file.write_all(format!("{}", current_time).as_bytes()).unwrap();
}


pub(crate) fn get_current_absolute_time() -> i64 {
    let global_time = std::fs::read_to_string(GLOBAL_TIME_PATH).unwrap();
    chrono::DateTime::parse_from_rfc3339(&global_time).unwrap().timestamp()
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    // Test penalty_type
    #[test]
    fn test_penalty_type() {
        let penalty_type_string = "Linear(1)";
        let penalty_type = PenaltyType::from_string_format(penalty_type_string);
        assert_eq!(penalty_type, PenaltyType::Linear(1));

        let penalty_type_string = "LinearWithRange(1,2,3,4)";
        let penalty_type = PenaltyType::from_string_format(penalty_type_string);
        assert_eq!(penalty_type, PenaltyType::LinearWithRange(1,2,3,4));

        let penalty_type_string = "None()";
        let penalty_type = PenaltyType::from_string_format(penalty_type_string);
        assert_eq!(penalty_type, PenaltyType::None);


        let penalty_type = PenaltyType::Linear(1);
        assert_eq!(penalty_type.to_string_format(), "Linear(1)");

        let penalty_type = PenaltyType::LinearWithRange(1,2,3,4);
        assert_eq!(penalty_type.to_string_format(), "LinearWithRange(1,2,3,4)");

        let penalty_type = PenaltyType::None;
        assert_eq!(penalty_type.to_string_format(), "None()");
    }

    #[test]
    fn overwrtite_global_time_manualy_test(){
        overwrtite_global_time_manualy(0);
    }
}
use std::{collections::HashMap, error::Error, fs::{self, File}, io::Write, os::unix::process, path::Path};

use csv::DeserializeError;
use polars::{datatypes::DataType, df, frame::DataFrame, series::Series};
use regex::Regex;
use serde::{ser, Deserialize, Serialize};

pub static GLOBAL_TIME_PATH: &str = "testcase/global_time";
pub static RESULT_PATH: &str = "/data01/cab314/GEMS";
pub static PENALTY_MAXIMUM: i64 = 1_000_000_000_000;

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
    pub experiment_uuid: String,
    pub task_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct  ScheduledTask {
    pub optimal_timing: OptimalTiming,
    pub processing_time: ProcessingTime,
    pub penalty_type: PenaltyType,
    pub experiment_operation: ExperimentOperation,
    pub experiment_name: ExperimentName,
    pub experiment_uuid: String,
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
    pub experiment_uuid: String,
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
                experiment_uuid: tasks[i].experiment_uuid.clone(),
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
                experiment_uuid: scheduled_task_serde[i].experiment_uuid.clone(),
                schedule_timing: scheduled_task_serde[i].schedule_timing,
                task_id: scheduled_task_serde[i].task_id,
            }
        )
    }
    println!("Read the schedule: {:?}", path);
    Ok(scheduled_tasks)

}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PenaltyType {
    /// 0: None
    None,
    /// 1: Linear, let an earliness or a delay be "diff" (ScheduleTiming - OptimalTiming), then the penalty is abs(diff * coefficient)
    Linear{coefficient:PenaltyParameter},

    /// 2: LinearWithRange, let an earliness or a delay be "diff", 
    /// then the penalty of LinearWithRange(lower, lower_coefficient, upper, upper_coefficient) is
    /// if diff < lower, then (lower-diff) * lower_coefficient
    /// if lower <= diff <= upper, then 0
    /// if upper < diff, then (diff-upper) * upper_coefficient
    LinearWithRange{lower:PenaltyParameter, lower_coefficient:PenaltyParameter, upper:PenaltyParameter, upper_coefficient:PenaltyParameter},

    /// 3: CyclicalRestPenalty, this penalty is used for the rest time, like holidays.
    /// The penalty is calculated by the following steps:
    /// 1. Calculate the diff = ScheduleTiming - start_minute, which is the time from the start_minute.
    /// 2. Calculate the diff = diff % cycle_minute, which is the time from the start_minute in the cycle.
    /// 3. If diff is in the ranges, then the penalty is 0, otherwise the penalty is PENALTY_MAXIMUM.
    /// The ranges are defined by the vector of (start, end) in the ranges.
    CyclicalRestPenalty { start_minute: i64, cycle_minute: i64, ranges: Vec<(i64, i64)>},

    /// 4: CyclicalRestPenaltyWithLinear, this penalty is used for the rest time, like holidays.
    /// The penalty is calculated by the following steps:
    /// 1. Calculate the diff = ScheduleTiming - start_minute, which is the time from the start_minute.
    /// 2. Calculate the diff = diff % cycle_minute, which is the time from the start_minute in the cycle.
    /// 3. If diff is in the ranges, then the penalty is PENALTY_MAXIMUM, otherwise the penalty is abs(diff * coefficient).
    /// The ranges are defined by the vector of (start, end) in the ranges.
    CyclicalRestPenaltyWithLinear { start_minute: i64, cycle_minute: i64, ranges: Vec<(i64, i64)>, coefficient: i64},
    
}

impl PenaltyType {
    /// Get the penalty
    /// diff: ScheduleTiming - OptimalTiming
    pub fn get_penalty(&self, scheduled_timing: ScheduleTiming, optimal_timing: OptimalTiming) -> i64 {
        let diff = scheduled_timing - optimal_timing;
        match self {
            PenaltyType::None => 0,
            PenaltyType::Linear{coefficient: coefficient} => diff.abs() * coefficient,
            PenaltyType::LinearWithRange{lower: lower, lower_coefficient: lower_coefficient, upper: upper, upper_coefficient: upper_coefficient} => {
                if &diff < lower {
                    (lower-diff) * lower_coefficient
                } else if lower <= &diff && &diff <= upper {
                    0
                } else {
                    (diff-upper) * upper_coefficient
                }
            },
            PenaltyType::CyclicalRestPenalty { start_minute: start_minute, cycle_minute: cycle_minute, ranges: ranges} => {
                let mut penalty = 0;
                let mut in_the_rest_time: bool = false;
                let diff = scheduled_timing - start_minute;
                if diff < 0 {
                    return 0;
                }
                let diff = diff % cycle_minute;
                for (start, end) in ranges {
                    if start <= &diff && &diff <= end {
                        in_the_rest_time = true;
                        break;
                    }
                }
                match in_the_rest_time {
                    // Scheduler has to respect the rest time
                    true => PENALTY_MAXIMUM,
                    // Thank you for your work
                    false => 0,
                }
            },
            PenaltyType::CyclicalRestPenaltyWithLinear { start_minute: start_minute, cycle_minute: cycle_minute, ranges: ranges, coefficient: coefficient} => {
                let mut penalty = 0;
                let mut in_the_rest_time: bool = false;
                let diff = scheduled_timing - start_minute;
                if diff < 0 {
                    return 0;
                }
                let diff = diff % cycle_minute;
                for (start, end) in ranges {
                    if start <= &diff && &diff <= end {
                        in_the_rest_time = true;
                        break;
                    }
                }
                match in_the_rest_time {
                    // Scheduler has to respect the rest time
                    true => PENALTY_MAXIMUM,
                    // Thank you for your work
                    false => diff.abs() * coefficient,
                }
            },
            _ => panic!("The penalty type is not supported."),
        }
    }

    /// Convert the penalty type to a string format
    pub fn to_string_format(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    /// Convert a string format to the penalty type
    pub fn from_string_format(penalty_type_string: &str) -> Self {
        serde_json::from_str(penalty_type_string).unwrap()
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

/// The time unit is minute
pub(crate) fn overwrtite_global_time_manualy(current_time: i64){
    // Global time in the file, GLOBAL_TIME_PATH, is rewritten manually.


    // current_time(i64, timestamp) -> chrono::DateTime
    let current_time = chrono::TimeZone::timestamp_opt(&chrono::Utc, current_time*60, 0);
    println!("overwrite time sec: {:?}", current_time);

    let current_time = current_time.unwrap().to_rfc3339();

    // Elapsed time
    


    // Rewrite the global time
    let mut file = File::create(GLOBAL_TIME_PATH).unwrap();
    file.write_all(format!("{}", current_time).as_bytes()).unwrap();
}

/// Get the current absolute time
/// The time unit is minute
pub(crate) fn get_current_absolute_time() -> i64 {
    let global_time = std::fs::read_to_string(GLOBAL_TIME_PATH).unwrap();
    let global_time = chrono::DateTime::parse_from_rfc3339(&global_time).unwrap().timestamp()/60;
    global_time
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    // Test penalty_type
    #[test]
    fn test_penalty_type_to_string() {
        let penalty_type = PenaltyType::Linear { coefficient: 1 };
        println!("{:?}", penalty_type.to_string_format());

        let penalty_type = PenaltyType::LinearWithRange { lower: 1, lower_coefficient: 1, upper: 2, upper_coefficient: 2 };
        println!("{:?}", penalty_type.to_string_format());

        let penalty_type = PenaltyType::CyclicalRestPenalty { start_minute: 1, cycle_minute: 2, ranges: vec![(1, 2)]};
        println!("{:?}", penalty_type.to_string_format());
    }

    #[test]
    fn test_penalty_type_cyclical_rest_penalty() {
        let penalty_type = PenaltyType::CyclicalRestPenalty { 
            start_minute: 0, cycle_minute: 10, ranges: vec![(1, 2)]};

        let optimal_timing = 0;

        for scheduled_timing in 0..10 {
            println!("scheduled_timing: {:?}, penalty: {:?}", scheduled_timing, penalty_type.get_penalty(scheduled_timing, optimal_timing));   
        }
    }

    #[test]
    fn test_penalty_type_cyclical_rest_penalty_with_linear() {
        let penalty_type = PenaltyType::CyclicalRestPenaltyWithLinear { 
            start_minute: 0, cycle_minute: 10, ranges: vec![(1, 8)], coefficient: 1};

        let optimal_timing = 0;

        for scheduled_timing in 0..10 {
            println!("scheduled_timing: {:?}, penalty: {:?}", scheduled_timing, penalty_type.get_penalty(scheduled_timing, optimal_timing));   
        }
    }

    #[test]
    fn overwrtite_global_time_manualy_test(){
        overwrtite_global_time_manualy(0);
    }
}
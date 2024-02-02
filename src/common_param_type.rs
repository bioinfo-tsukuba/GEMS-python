use std::{collections::HashMap, error::Error, os::unix::process};

use polars::{datatypes::DataType, frame::DataFrame};
use regex::Regex;

// 共通のパラメータ型を定義
pub type StateIndex = usize;
pub type StateName = String;
pub type ExperimentIndex = usize;
pub type ExperimentName = String;

pub type OptimalTiming = i32;
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
}

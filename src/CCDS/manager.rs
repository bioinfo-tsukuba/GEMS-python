use crate::common_param_type::*;
use crate::transition_manager::*;
use crate::task_generator::*;
use crate::task_scheduler::*;

use super::Experiment;


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




pub(crate) static CELL_CULTURE_EXPERIMENT_NAME:&str = "CELL_CULTURE";

pub(crate) static CELL_CULTURE_STATE_NAMES:[&str; 4] = [
    "EXPIRE",
    "PASSAGE",
    "GET_IMAGE",
    "MEDIUM_CHANGE",
];


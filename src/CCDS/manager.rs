use crate::common_param_type::*;
use crate::transition_manager::*;
use crate::task_generator::*;
use crate::task_scheduler::*;

use super::Experiment;

pub(crate) struct CellGrowthParam{
    /// [0]: Cell growth rate, r
    /// [1]: Initial cell density, N0
    /// [2]: Maximum cell density, K
    cell_growth_params: [f32; 3],
}


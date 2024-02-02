use super::{ScheduledTask, TaskId};

pub(crate) struct NormalCellSimulator {
    /// [0]: Cell growth rate, r
    /// [1]: Initial cell density, N0
    /// [2]: Maximum cell density, K
    cell_growth_params: [f32; 3],
}

impl NormalCellSimulator {
    fn new(cell_growth_params: [f32; 3]) -> Self { Self { cell_growth_params } }

    fn logistic_function(&self, x: f32) -> f32 {
        let r = self.cell_growth_params[0];
        let N0 = self.cell_growth_params[1];
        let K = self.cell_growth_params[2];
        K / (1.0 + (K / N0 - 1.0) * (-r * x).exp())
    }
    pub(crate) fn simulate_cell_growth(&self, current_time: f32, initial_time: f32) -> f32 {
        let delta_time = current_time - initial_time;
        self.logistic_function(delta_time)
        
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

    pub(crate) fn process_earliest_task(&mut self)-> Vec<TaskId>{
        // Sort the scheduled_tasks by schedule_timing
        self.scheduled_tasks.sort_by(|a, b| a.schedule_timing.cmp(&b.schedule_timing));

        // Get the earliest task, without removing it
        let earliest_task = self.scheduled_tasks.first().unwrap();

        // Get the task_id of the earliest task
        vec![earliest_task.task_id]
    }
}
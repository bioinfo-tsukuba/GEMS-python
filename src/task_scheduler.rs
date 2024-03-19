/// This module contains the task scheduler for one machine scheduling problem.
/// Common input: the vector of 'common_param_type::Task'
/// Let's review the definition of 'common_param_type::Task'
/// pub struct Task{
///   pub optimal_timing: OptimalTiming,
///   pub processing_time: ProcessingTime,
///   pub penalty_type: PenaltyType,
///   pub experiment_operation: ExperimentOperation,
/// }
/// Common output: the vector of 'common_param_type::ScheduledTask'
/// Let's review the definition of 'common_param_type::ScheduledTask'
/// pub struct  ScheduledTask {
///   pub schedule_timing: ScheduleTiming,
///   pub optimal_timing: OptimalTiming,
///   pub processing_time: ProcessingTime,
///   pub penalty_type: PenaltyType,
///   pub experiment_operation: ExperimentOperation,
/// }
/// 
/// Constraint1: the schedule timing of the next task must be equal to or greater than the schedule timing of the previous task.
/// Let the schedule timing of the previous task be 'previous_schedule_timing' and the schedule timing of the next task be 'next_schedule_timing'.
pub(crate) mod one_machine_schedule_solver{
    // Define the common parameter types
    use crate::common_param_type::*;

    /// The scheduler manage the schedule task
    /// Input: Task with "RELATIVE" optimal time
    /// Output: Scheduled task with "RELATIVE" time
    pub(crate) fn FIFO_scheduler_relative(tasks: Vec<Task>) -> Vec<ScheduledTask> {
        let mut scheduled_tasks: Vec<(usize, ScheduledTask)> = Vec::with_capacity(tasks.len());

        // Move the tasks to scheduled_tasks
        for (index, task) in tasks.into_iter().enumerate() {
            scheduled_tasks.push((index, ScheduledTask {
                schedule_timing: 0,
                optimal_timing: task.optimal_timing,
                processing_time: task.processing_time,
                penalty_type: task.penalty_type,
                experiment_operation: task.experiment_operation,
                experiment_name: task.experiment_name,
                task_id: task.task_id,
            }));
        }

        // Sort the scheduled_tasks by OptimalTiming
        // If the OptimalTiming is the same, sort by ProcessingTime
        scheduled_tasks.sort_by(|a, b| {
            if a.1.optimal_timing == b.1.optimal_timing {
                a.1.processing_time.cmp(&b.1.processing_time)
            } else {
                a.1.optimal_timing.cmp(&b.1.optimal_timing)
            }
        });

        // Assign the schedule timing
        let mut previous_task_completion_time = 0;
        for index in 0..scheduled_tasks.len() {
            let task = &mut scheduled_tasks[index].1;
            let schedule_timing = previous_task_completion_time.max(task.optimal_timing);
            println!("previous_task_completion_time: {}, 
            task.optimal_timing: {}, 
            schedule_timing: {}", 
            previous_task_completion_time, task.optimal_timing, schedule_timing);
            task.schedule_timing = schedule_timing;
            previous_task_completion_time = task.schedule_timing+task.processing_time;
        }

        // Sort the scheduled_tasks by index
        scheduled_tasks.sort_by(|a, b| a.0.cmp(&b.0));

        let scheduled_tasks: Vec<ScheduledTask> = scheduled_tasks.into_iter().map(|it| it.1).collect();

        println!("scheduled_tasks (relative)");
        for scheduled_task in &scheduled_tasks {
            println!("{:?}", scheduled_task);
        }

        scheduled_tasks
    }


    /// The scheduler manage the schedule task
    /// Input: Task with "ABSOLUTE" optimal time
    /// Output: Scheduled task with "ABSOLUTE" time
    /// Note: The schedule must finish in Unit time(min)
    pub(crate) fn FIFO_scheduler_absolute(tasks: Vec<Task>) -> Vec<ScheduledTask> {

        let current_absolute_time = get_current_absolute_time();
        let tasks: Vec<Task> = tasks.into_iter().map(|mut task| {
            task.optimal_timing = task.optimal_timing - current_absolute_time;
            task
        }).collect();

        let scheduled_tasks = FIFO_scheduler_relative(tasks);

        let scheduled_tasks: Vec<ScheduledTask> = scheduled_tasks.into_iter().map(|mut task| {
            task.optimal_timing = task.optimal_timing + current_absolute_time;
            task.schedule_timing = task.schedule_timing + current_absolute_time;
            task
        }).collect();

        println!("scheduled_tasks (absolute)");
        for scheduled_task in &scheduled_tasks {
            println!("{:?}", scheduled_task);
        }
        scheduled_tasks
    }


    /// The scheduler manage the schedule task
    /// Input: Task with "RELATIVE" optimal time
    /// Output: Scheduled task with "ABSOLUTE" time
    /// Note: The schedule must finish in Unit time(min)
    pub(crate) fn FIFO_scheduler(tasks: Vec<Task>) -> Vec<ScheduledTask> {
        println!("Now fifo scheduling");

        let scheduled_tasks = FIFO_scheduler_relative(tasks);

        let current_absolute_time = get_current_absolute_time();

        let scheduled_tasks: Vec<ScheduledTask> = scheduled_tasks.into_iter().map(|mut task| {
            task.optimal_timing = task.optimal_timing + current_absolute_time;
            task.schedule_timing = task.schedule_timing + current_absolute_time;
            task
        }).collect();

        scheduled_tasks
    }
    

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_FIFO_scheduler() {
            // Define the tasks
            let tasks = vec![
                Task {
                    optimal_timing: 0,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear(100),
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    task_id: 0,
                },
                Task {
                    optimal_timing: 1,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear(100),
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    task_id: 1,
                },
                Task {
                    optimal_timing: 500,
                    processing_time: 1,
                    penalty_type: PenaltyType::Linear(100),
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    task_id: 2,
                },
            ];

            // Schedule the tasks
            let scheduled_tasks = FIFO_scheduler(tasks);

            // Check the scheduled_tasks
            for scheduled_task in scheduled_tasks {
                println!("{:?}", scheduled_task);
            }
        }
    }


}
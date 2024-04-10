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
    use argmin::{core::CostFunction, solver::simulatedannealing::Anneal};
    use polars::chunked_array::ops::sort;
    use rand::{Rng, SeedableRng};

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
                experiment_uuid: task.experiment_uuid,
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

    /// The scheduler manage the schedule task
    /// Input: Task with "ABSOLUTE" optimal time
    /// Output: Scheduled task with "ABSOLUTE" time
    /// Note: The schedule must finish in Unit time(min)
    pub(crate) fn simulated_annealing_scheduler_absolute(tasks: Vec<Task>)-> Vec<ScheduledTask> {

        let current_absolute_time = get_current_absolute_time();
        let tasks: Vec<Task> = tasks.into_iter().map(|mut task| {
            task.optimal_timing = task.optimal_timing - current_absolute_time;
            task.penalty_type = match task.penalty_type {
                PenaltyType::CyclicalRestPenalty { start_minute, cycle_minute, ranges }
                => {
                    let start_minute = start_minute - current_absolute_time;
                    PenaltyType::CyclicalRestPenalty { start_minute, cycle_minute, ranges }
                },
                _ => {task.penalty_type}
            };
            task
        }).collect();

        let scheduled_tasks = simulated_annealing_scheduler_relative(tasks);

        let scheduled_tasks: Vec<ScheduledTask> = scheduled_tasks.into_iter().map(|mut task| {
            task.optimal_timing = task.optimal_timing + current_absolute_time;
            task.schedule_timing = task.schedule_timing + current_absolute_time;
            task.penalty_type = match task.penalty_type {
                PenaltyType::CyclicalRestPenalty { start_minute, cycle_minute, ranges }
                => {
                    let start_minute = start_minute + current_absolute_time;
                    PenaltyType::CyclicalRestPenalty { start_minute, cycle_minute, ranges }
                },
                _ => {task.penalty_type}
            };
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
    /// Output: Scheduled task with "RELATIVE" time
    pub(crate) fn simulated_annealing_scheduler_relative(tasks: Vec<Task>)-> Vec<ScheduledTask>{
        struct TaskSASchedule {
            /// The task
            task: Vec<Task>,
            /// lower bound of the task
            lower_bound: Vec<ScheduleTiming>,
            /// upper bound of the task
            upper_bound: Vec<ScheduleTiming>,

            /// Random number generator. We use a `Arc<Mutex<_>>` here because `Anneal` requires
            /// `self` to be passed as an immutable reference. This gives us thread safe interior
            /// mutability.
            rng: std::sync::Arc<std::sync::Mutex<rand_xoshiro::Xoshiro256PlusPlus>>,
        }

        impl TaskSASchedule {
            /// Create a new `TaskSASchedule` instance.
            fn new(task: Vec<Task>, lower_bound: Vec<ScheduleTiming>, upper_bound: Vec<ScheduleTiming>) -> Self {
                Self {
                    task,
                    lower_bound,
                    upper_bound,
                    rng: std::sync::Arc::new(std::sync::Mutex::new(rand_xoshiro::Xoshiro256PlusPlus::from_entropy())),
                }
            }
        }

        impl CostFunction for TaskSASchedule{
            type Param = Vec<ScheduleTiming>;
            type Output = f64;

            fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
                let mut cost :PenaltyParameter = 0;
                for (index, task) in self.task.iter().enumerate() {
                    let schedule_timing = param[index];
                    let penalty = task.penalty_type.get_penalty(schedule_timing, task.optimal_timing);
                    if penalty < 0 {
                        panic!("Penalty must be greater than or equal to 0")
                    }
                    cost += penalty;
                }

                // Overlapping penalty
                let mut sorted_tasks = param.iter().enumerate().collect::<Vec<_>>();
                sorted_tasks.sort_by(|a, b| a.1.cmp(b.1));
                for i in 0..sorted_tasks.len()-1 {
                    let task1 = &self.task[sorted_tasks[i].0];
                    let task2 = &self.task[sorted_tasks[i+1].0];
                    let schedule_timing1 = sorted_tasks[i].1;
                    let schedule_timing2 = sorted_tasks[i+1].1;
                    let overlap = schedule_timing1+task1.processing_time-schedule_timing2;
                    if overlap > 0 {
                        cost += overlap * 100000;
                    }
                }
                Ok(cost as f64)
            }
        }

        impl Anneal for TaskSASchedule {
            type Param = Vec<ScheduleTiming>;
            type Output = Vec<ScheduleTiming>;
            type Float = f64;

            /// Anneal a parameter vector
            fn anneal(&self, param: &Self::Param, temp: Self::Float) -> Result<Self::Param, argmin::core::Error> {
                let mut param_n = param.clone();
                let mut rng = self.rng.lock().unwrap();
                let distr = rand::distributions::Uniform::from(0..param.len());

                // println!("Temp: {}", temp);
                // Perform modifications to a degree proportional to the current temperature `temp`.
                for _ in 0..(temp.floor() as u64 + 1){
                    // Compute random index of the parameter vector using the supplied random number
                    // generator.
                    let idx = rng.sample(distr);

                    // Compute random number in [0.1, 0.1].
                    let val = rng.sample(rand::distributions::Uniform::new_inclusive(-120, 120));

                    // modify previous parameter value at random position `idx` by `val`
                    param_n[idx] += val;

                    // check if bounds are violated. If yes, project onto bound.
                    param_n[idx] = param_n[idx].clamp(self.lower_bound[idx], self.upper_bound[idx]);
                }
                Ok(param_n)
            }
        }
        fn initialise_param(tasks: Vec<Task>) -> Vec<ScheduleTiming> {
            let mut ordered_tasks = tasks;
            ordered_tasks.sort_by(|a, b| a.optimal_timing.cmp(&b.optimal_timing));
            let mut previous_release_timing = 0;
            let mut param = Vec::with_capacity(ordered_tasks.len());
            for task in ordered_tasks {
                let schedule_timing = previous_release_timing.max(task.optimal_timing);
                param.push(schedule_timing);
                previous_release_timing = schedule_timing + task.processing_time;
            }
            param
        }

        let lower_bound: Vec<ScheduleTiming> = tasks.iter().map(|task| (task.optimal_timing-1000).max(0)).collect();
        let upper_bound: Vec<ScheduleTiming> = tasks.iter().map(|task| (task.optimal_timing+1000).max(0)).collect();

        // Define the initial parameter
        let init_param: Vec<ScheduleTiming> = initialise_param(tasks.clone());

        // Define the initial temperature
        let temp = (tasks.len()*10) as f64;

        // Define the cost function
        let operator = TaskSASchedule::new(tasks.clone(), lower_bound, upper_bound);

        
        let solver = argmin::solver::simulatedannealing::SimulatedAnnealing::new(temp).unwrap()
            // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
            .with_temp_func(argmin::solver::simulatedannealing::SATempFunc::Boltzmann)
            /////////////////////////
            // Stopping criteria   //
            /////////////////////////
            // Optional: stop if there was no new best solution after 1000 iterations
            .with_stall_best(1_000_000)
            // Optional: stop if there was no accepted solution after 1000 iterations
            .with_stall_accepted(1_000_000)
            /////////////////////////
            // Reannealing         //
            /////////////////////////
            // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
            .with_reannealing_fixed(100_000)
            // Optional: Reanneal after no accepted solution has been found for `iter` iterations
            .with_reannealing_accepted(5_000)
            // Optional: Start reannealing after no new best solution has been found for 800 iterations
            .with_reannealing_best(8_000);

            /////////////////////////
            // Run solver          //
            /////////////////////////
        let res = 
        match  argmin::core::Executor::new(operator, solver)
        .configure(|state| {
            state
                .param(init_param)
                // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
                .max_iters(10_000_000)
                // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
                .target_cost(0.0)
        })
        // Optional: Attach a observer
        // .add_observer(argmin::core::observers::SlogLogger::term(), argmin::core::observers::ObserverMode::Always)
        .run(){
            Ok(res) => res,
            Err(e) => {
                println!("Simulated Annealing Error: {}", e);
                return FIFO_scheduler_relative(tasks);
            }
        };

        // Print result
        println!("Simulated Annealing Result: {}", res);
        let best_param = res.state.best_param.unwrap();

        tasks.into_iter().enumerate().map(|(index, task)| {
            ScheduledTask {
                schedule_timing: best_param[index],
                optimal_timing: task.optimal_timing,
                processing_time: task.processing_time,
                penalty_type: task.penalty_type,
                experiment_operation: task.experiment_operation,
                experiment_name: task.experiment_name,
                experiment_uuid: task.experiment_uuid,
                task_id: task.task_id,
            }
        }).collect()

    }

    pub(crate) fn greedy_optimisation_relative(pre_scheduled_tasks: Vec<ScheduledTask>)-> Vec<ScheduledTask>{

        todo!()
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
                    penalty_type: PenaltyType::Linear { coefficient: 100 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 0,
                },
                Task {
                    optimal_timing: 1,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear { coefficient: 100 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 1,
                },
                Task {
                    optimal_timing: 500,
                    processing_time: 1,
                    penalty_type: PenaltyType::Linear { coefficient: 100 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
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

        #[test]
        fn test_FIFO_scheduler_bad_pattern() {
            // Define the tasks
            let tasks = vec![
                Task {
                    optimal_timing: 0,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear { coefficient: 1 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 0,
                },
                Task {
                    optimal_timing: 1,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear { coefficient: 1000 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 1,
                },
                Task {
                    optimal_timing: 500,
                    processing_time: 1,
                    penalty_type: PenaltyType::Linear { coefficient: 100 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
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

        #[test]
        fn test_sa_scheduler() {
            // Set absolute time
            let current_absolute_time = 0;
            overwrtite_global_time_manualy(current_absolute_time);
            // Define the tasks
            let tasks = vec![
                Task {
                    optimal_timing: 0,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear { coefficient: 1 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 0,
                },
                Task {
                    optimal_timing: 1,
                    processing_time: 100,
                    penalty_type: PenaltyType::Linear { coefficient: 1000 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 1,
                },
                Task {
                    optimal_timing: 500,
                    processing_time: 1,
                    penalty_type: PenaltyType::Linear { coefficient: 100 },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 2,
                },
                Task {
                    optimal_timing: 100,
                    processing_time: 1,
                    penalty_type: PenaltyType::CyclicalRestPenalty{
                        start_minute: 0,
                        cycle_minute: 24*60,
                        ranges: vec![(0, 9*60), (18*60, 24*60)],
                    },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 3,
                },
                Task {
                    optimal_timing: 100,
                    processing_time: 1,
                    penalty_type: PenaltyType::CyclicalRestPenaltyWithLinear {
                        start_minute: 0,
                        cycle_minute: 24*60,
                        ranges: vec![(0, 9*60), (18*60, 24*60)],
                        coefficient: 100
                    },
                    experiment_operation: "A".to_string(),
                    experiment_name: "A".to_string(),
                    experiment_uuid: uuid::Uuid::new_v4().to_string(),
                    task_id: 4,
                },
            ];

            // Schedule the tasks
            let scheduled_tasks = simulated_annealing_scheduler_relative(tasks);

            // Check the scheduled_tasks
            for scheduled_task in scheduled_tasks {
                println!("{:?}", scheduled_task);
            }
        }
    }


}
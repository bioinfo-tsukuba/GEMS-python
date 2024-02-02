use std::{error::Error, result};
use crate::{common_param_type::{self, TaskId, TaskResult}, transition_manager, task_generator, task_scheduler::{self, one_machine_schedule_solver}};
use polars::{functions::concat_df_diagonal, prelude::*};



/// State has a transition rule (TransitionManager), 
/// a Fomula for ideal schedule timing and experiment operation (TaskGenerator)
/// Define the struct of State
pub(crate) struct State {
    pub(crate) transition_function: transition_manager::TransitionManager,
    pub(crate) task_generator: task_generator::TaskGenerator,
    pub(crate) state_index: common_param_type::StateIndex,
    pub(crate) state_name: common_param_type::StateName,
}

impl State {
    pub(crate) fn new(
        transition_function: transition_manager::TransitionManager,
        task_generator: task_generator::TaskGenerator,
        state_index: common_param_type::StateIndex,
        state_name: common_param_type::StateName,
    ) -> Self {
        Self {
            transition_function,
            task_generator,
            state_index,
            state_name,
        }
    }
}

pub(crate) enum SharedVariableHistoryInput{
    Path(String),
    DataFrame(polars::frame::DataFrame),
}

/// Experiment has a several states and a shared variable history.
/// Define the struct of Experiment
pub(crate) struct Experiment {
    pub(crate) experiment_name: common_param_type::ExperimentName,
    pub(crate) states: Vec<State>,
    pub(crate) current_state_index: common_param_type::StateIndex,
    pub(crate) shared_variable_history: polars::frame::DataFrame,// mutability is required
}

impl Experiment {
    pub(crate) fn new(
        experiment_name: common_param_type::ExperimentName,
        states: Vec<State>,
        current_state_index: common_param_type::StateIndex,
        shared_variable_history: SharedVariableHistoryInput,
    ) -> Self {

        let shared_variable_history = match shared_variable_history{
            SharedVariableHistoryInput::Path(path) => {
                CsvReader::from_path(path).unwrap()
                .has_header(true)
                .finish()
                .unwrap()
            },
            SharedVariableHistoryInput::DataFrame(data_frame) => data_frame,
        };


        Self {
            experiment_name,
            states,
            current_state_index,
            shared_variable_history,
        }
    }

    /// Show the experiment name and the state names
    pub(crate) fn show_experiment_name_and_state_names(&self) {
        println!("Experiment name: {}", self.experiment_name);
        println!("State names:");
        print!("{}", self.states[0].state_name);
        for state in &self.states[1..] {
            print!(" => {}", state.state_name);
        }
    }

    /// Show the current state name
    pub(crate) fn show_current_state_name(&self) {
        self.show_experiment_name_and_state_names();
        println!("\nCurrent state name: {}", self.states[self.current_state_index].state_name);
    }
}

impl Experiment { 
    pub(crate) fn execute_one_step(
        &mut self,
    ) -> common_param_type::Task {
        // Determine the next state index
        let next_state_index = match self.states[self.current_state_index].transition_function.determine_next_state_index(&mut self.shared_variable_history) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        // Update the current state index
        self.current_state_index = next_state_index;

        // Generate a task
        let task = match self
        .states[self.current_state_index]
        .task_generator
        .generate_task(&mut self.shared_variable_history, self.experiment_name.clone()) {
            Ok(it) => it,
            Err(err) => panic!("{}", err),
        };

        task
    }
}

/// ExperimentManager has 
/// a several experiments,
/// task scheduler,
/// Define the struct of ExperimentManager
pub(crate) struct OneMachineExperimentManager {
    pub(crate) experiments: Vec<Experiment>,
    pub(crate) tasks: Vec<common_param_type::Task>,
}

impl OneMachineExperimentManager {
    pub(crate) fn new(
        experiments: Vec<Experiment>,
        tasks: Vec<common_param_type::Task>,
    ) -> Self {
        Self {
            experiments,
            tasks,
        }
    }
}

impl OneMachineExperimentManager {

    /// Delete the tasks with task_id
    pub(crate) fn delete_tasks_with_task_id(
        &mut self, 
        task_id: TaskId,
    ){
        // Delete the tasks with task_id
        self.tasks.retain(|task| task.task_id != task_id);
    }

    /// Update the state and reschedule
    pub(crate) fn update_state_and_reschedule(
        &mut self, 
        task_id: TaskId,
        new_result_of_experiment: TaskResult,
        update_type: char,
    ) -> Vec<common_param_type::ScheduledTask> {

        let experiment_name = self.tasks[task_id].experiment_name.clone();
        eprintln!("Update the state of experiment: {}", experiment_name);
        let experiment_index = self.experiments.iter().position(|experiment| experiment.experiment_name == experiment_name).unwrap();

        self.delete_tasks_with_task_id(task_id);
        
        // Update the shared variable history
        match update_type {
            'a' => {
                // Add the new result of experiment to the shared variable history
                self.experiments[experiment_index].shared_variable_history = concat_df_diagonal(&[
                    self.experiments[experiment_index].shared_variable_history.clone(),
                    new_result_of_experiment,
                ]).unwrap();
            },
            'w' => {
                // Replace the shared variable history with the new result of experiment
                self.experiments[experiment_index].shared_variable_history = new_result_of_experiment;
            },
            _ => panic!("The update type is not supported."),
        }

        // Execute one step
        let task = self.experiments[experiment_index].execute_one_step();

        // Add the task to the task list
        self.tasks.push(task);

        // Reindex the tasks
        for (index, task) in self.tasks.iter_mut().enumerate() {
            task.task_id = index;
        }

        // Reschedule
        one_machine_schedule_solver::FIFO_scheduler(self.tasks.clone())
    }
}

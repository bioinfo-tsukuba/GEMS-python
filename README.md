# What is GEMS?


# How to Use

This guide outlines the necessary steps to define and run a cell culture experiment simulation using Rust. It focuses on setting up the experiment without providing direct code examples.

## Prerequisites
- Ensure Rust is installed on your system.
- Familiarize yourself with the Rust testing framework, specifically the `#[test]` attribute.
- Understand basic Rust concepts such as closures, enums, structs, and error handling.

## Steps to Define an Experiment

1. **Import Required Crates**: Include crates that provide functionalities like data manipulation (e.g., `DataFrame`) and error handling.

2. **Define Transition Functions**: For each state of the experiment (e.g., "Expire", "PASSAGE", "GET_IMAGE"), define functions to handle:
   - Processing time calculation.
   - Optimal timing and penalty assessment.
   - State transition logic based on experiment conditions.

3. **Create State Objects**: For each state, instantiate a `State` object with:
   - A `TransitionManager` to manage state transitions.
   - A `TaskGenerator` to generate tasks based on the experiment operation, processing time function, and timing function.
   - An initial state index and a descriptive state name.

4. **Prepare Shared Variable History**: Initialize a `DataFrame` with experiment variables (e.g., "density" and "time") that will be shared across states.

5. **Instantiate the Experiment**: Create an `Experiment` object with:
   - A unique experiment name.
   - A list of state objects.
   - An initial state index.
   - The shared variable history.

6. **Execute the Experiment**: Call the experiment's method to execute one step of the experiment, generating a new task.

7. **Manage Experiment Schedule**: Create an `OneMachineExperimentManager` to manage the experiment's schedule, including:
   - Adding the experiment and its tasks to the schedule.
   - Updating the state and rescheduling tasks based on new experiment results.

8. **Simulate Experiment Execution**: Implement a loop to simulate the execution of experiment tasks, where:
   - The earliest task is processed.
   - The state is updated with new experiment results.
   - The experiment's current state name is printed.

9. **Testing**: Wrap the experiment definition and execution logic within a Rust test function (annotated with `#[test]`) to facilitate automated testing and validation of the simulation logic.

## Notes
- The example provided simplifies certain aspects, such as always returning a fixed value for processing times and transitions, for illustrative purposes.
- The actual logic for processing time calculations, timing assessments, and state transitions should be adapted based on the specific requirements of the cell culture experiment being simulated.
- Error handling is crucial for robust simulation. Ensure that all functions properly handle errors, potentially using custom error types.

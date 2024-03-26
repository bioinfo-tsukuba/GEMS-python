extern crate argmin;
extern crate argmin_testfunctions;
use argmin::core::{State, Error, Executor, CostFunction};
use argmin::solver::neldermead::NelderMead;
use polars::frame::DataFrame;
use polars::lazy::frame::IntoLazy;
use polars::prelude::*;


#[derive(Clone, Default, Debug)]
struct CellGrowthHistory {
    time: f64,
    cell_count: f64,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct MyProblem {
    history: Vec<CellGrowthHistory>,
}

/// Make alias of MyProblem, cell_growth_problem
pub(crate) type CellGrowthProblem = MyProblem;

pub(crate) fn logistic_growth(t: f64, n0: f64, r: f64, k: f64) -> f64 {
    k / (1.0 + (k / n0 - 1.0) * (-r * t).exp())
}

/// Calculate value of t(passed time) from cell number
pub(crate) fn calculate_logistic_inverse_function(cell_number: f64, n0: f64, r: f64, k: f64) -> f64{
    ((k-n0).ln() - n0.ln() + cell_number.ln() - (k - cell_number).ln())/r
}

/// Calculate value of t(passed time) from cell number
pub(crate) fn calculate_logistic_inverse_binary_search(cell_number: f64, n0: f64, r: f64, k: f64) -> f64{
    let mut small_bound = 0.0;
    let mut large_bound = 100.0;

    let mut large_value ;
    let mut small_value ;

    // eprintln!("cell_number: {}", cell_number);
    // eprintln!("n0: {}, r: {}, k: {}", n0, r, k);
// 
    while large_bound - small_bound > 0.0001 {
        // std::thread::sleep(std::time::Duration::from_millis(100));
        large_value = logistic_growth(large_bound, n0, r, k);
        small_value = logistic_growth(small_bound, n0, r, k);
        // eprintln!("small_bound: {}, large_bound: {}", small_bound, large_bound);
        // eprintln!("small_value: {}, large_value: {}", small_value, large_value);
        if large_value < cell_number {
            large_bound = large_bound + (large_bound - small_bound);
        } else if  cell_number < small_value {
            small_bound = small_bound - (large_bound - small_bound);
        } else {
            let mid = (large_bound + small_bound) / 2.0;
            let mid_value = logistic_growth(mid, n0, r, k);
            if mid_value > cell_number {
                large_bound = mid;
            } else if mid_value < cell_number {
                small_bound = mid;
            } else {
                return mid;
            }
        }
    }
    (large_bound + small_bound) / 2.0
}

fn clamp_param(p: &[f64], min: f64, max: f64) -> Vec<f64> {
    p.iter().map(|&x| x.min(max).max(min)).collect()
}

// Default Parameter
impl MyProblem {
    pub(crate) fn new_from_df(variable_history: DataFrame, time_col_name: &str, cell_count_col_name: &str) -> Result<Self, Error> {
        let filtered_df = variable_history
        .lazy()
        .filter(col(time_col_name).is_not_null().and(col(cell_count_col_name).is_not_null()))
        .collect()?;

        
        let cell_growth_history: Vec<CellGrowthHistory> = filtered_df
        .column(time_col_name)?.f64()?
        .into_iter()
        .zip(filtered_df.column(cell_count_col_name)?.f64()?.into_iter())
        .filter_map(|(time, density)| match (time, density) {
            (Some(time), Some(density)) => Some(CellGrowthHistory { time, cell_count: density }),
            _ => None,
        })
        .collect();

        Ok(Self { history: cell_growth_history })
    }

    pub(crate) fn default_parameter(&self) -> Result<Vec<Vec<f64>>, Error> {
        // Generally, n0, r, k
        // 0 < n0 <= 0.1
        // 0 < r <= 1
        // n0 < k <= 1
        Ok(vec![
            vec![0.0001, 0.0001, 1.0],
            vec![0.1, 0.1, 1.0],
            vec![0.1, 0.00001, 1.0],
            vec![0.05, 0.5, 1.0],
        ])
    }
}

// Implement `CostFunction` for `MyProblem`
impl CostFunction for MyProblem {
    // [...]
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        assert_eq!(p.len(), 3);
        let mut cost = 0.0;
        for history in &self.history {
            let n = logistic_growth(history.time, p[0], p[1], p[2]);
            cost += (n - history.cell_count).powi(2);
        }
        Ok(cost)
    }
}



fn run() -> Result<(), Error> {

    let mut history = Vec::new();

    // Generate sample data
    // Let n0 = 1.0, r = 0.5, k = 1.0
    let n0 = 0.05;
    let r = 0.02;
    let k = 1.0;

    // - 0.01 < r' < 0.01
    let randomnoise = vec![
        -0.01, -0.01, 0.0, 0.01, 0.01,
    ];

    // Let t = 0.0, 0.1, 0.2, ..., 10.0
    for i in 0..10 {
        let t = i as f64 / 10.0;
        let cell_count = logistic_growth(t, n0, r + randomnoise[i%randomnoise.len()], k);
        history.push(CellGrowthHistory {
            time: t,
            cell_count: cell_count,
        });
    }

    println!("history: {:?}", history);

    // Create new instance of cost function
    let cost = MyProblem { history: history };

    // Define initial parameter vector
    let init_param = cost.default_parameter()?;

    // Using Nelder-Mead
    let solver = NelderMead::new(init_param);

    // Create an `Executor` object 
    let res = Executor::new(cost, solver)
    // Via `configure`, one has access to the internally used state.
    // This state can be initialized, for instance by providing an
    // initial parameter vector.
    // The maximum number of iterations is also set via this method.
    // In this particular case, the state exposed is of type `IterState`.
    // The documentation of `IterState` shows how this struct can be
    // manipulated.
    // Population based solvers use `PopulationState` instead of 
    // `IterState`.
    .configure(|state|
        state
            // Set maximum iterations to 10
            // (optional, set to `std::u64::MAX` if not provided)
            .max_iters(1000)
            // Set target cost. The solver stops when this cost
            // function value is reached (optional)
            .target_cost(0.0)
    )
    // run the solver on the defined problem
    .run()?;

    // print result
    println!("{}", res);

    // Extract results from state

    // Best parameter vector
    let best = res.state().get_best_param().unwrap();

    // Cost function value associated with best parameter vector
    let best_cost = res.state().get_best_cost();

    // Check the execution status
    let termination_status = res.state().get_termination_status();

    // Optionally, check why the optimizer terminated (if status is terminated) 
    let termination_reason = res.state().get_termination_reason();

    // Time needed for optimization
    let time_needed = res.state().get_time().unwrap();

    // Total number of iterations needed
    let num_iterations = res.state().get_iter();

    // Iteration number where the last best parameter vector was found
    let num_iterations_best = res.state().get_last_best_iter();

    // Number of evaluation counts per method (Cost, Gradient)
    let function_evaluation_counts = res.state().get_func_counts();
    Ok(())
    }

    pub(crate) fn example() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}

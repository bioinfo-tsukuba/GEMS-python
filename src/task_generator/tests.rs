use super::*;

#[test]
fn test_task_generator() {
    // 例として、常に状態ID 1 を返す遷移関数
    let experiment_operation = "Hello, world!".to_string();

    // データフレームの例（実際にはここに実験データが入る）
    let mut variable_history = match df!("variable" => [1, 2, 3, 4, 5], "time" => [1, 2, 3, 4, 5]) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };

    // Define a timing_func as linear regression of variable and estimate the time for the variable becoming 10

    let timing_func = |variable_history: &DataFrame| -> Result<(ExperimentOperation, OptimalTiming, PenaltyType), Box<dyn Error>> {
        // x and y are series of i32
        fn linear_regression(x: &Series, y: &Series, y_desired: f64) -> OptimalTiming {

            // Confirm the x and y have the same length
            assert_eq!(x.len(), y.len());

            // Cast x and y to f64 Series
            let x = x.cast(&DataType::Float64).unwrap();
            let y = y.cast(&DataType::Float64).unwrap();

            // Calculate the mean of x and y
            let x_mean = x.mean().unwrap();
            let y_mean = y.mean().unwrap();                // Calculate the covariance of x and y
            let cov_xy = x.f64()
                                .unwrap()
                                .into_iter()
                                .zip(y.f64().unwrap())
                                .map(|(xi, yi)| {
                                    xi.map(|xi| {
                                        yi.map(|yi| (xi - x_mean) * (yi - y_mean))
                                            .unwrap_or(0.0)
                                    })
                                    .unwrap_or(0.0)
                                })
                                .sum::<f64>() / x.len() as f64;

            // Calculate the variance of x
            let var_x = x.f64()
                                .unwrap()
                                .into_iter()
                                .map(|xi| {
                                    xi.map(|xi| (xi - x_mean).powi(2))
                                        .unwrap_or(0.0)
                                })
                                .sum::<f64>() / x.len() as f64;
                            
            // Calculate the slope (b1) and y-intercept (b0) of the regression line
            let slope = cov_xy / var_x;
            let y_intercept = y_mean - slope * x_mean;

            // Print the equation of the line
            println!("Equation of the line: y = {:.2}x + {:.2}", slope, y_intercept);

            // Predict the optimal timing based on the desired y value
            let optimal_timing = (y_desired - y_intercept) / slope;

            // Return the optimal timing
            optimal_timing as OptimalTiming
        }

        let variable = variable_history.column("variable").unwrap();
        let time = variable_history.column("time").unwrap();
        let optimal_timing = linear_regression(&variable, &time, 10.0);
        Ok(("test_operation".to_string() , optimal_timing, PenaltyType::Linear { coefficient: 10 }))
    };

    let processing_time_func = |variable_history: &DataFrame| -> Result<ProcessingTime, Box<dyn Error>> {
        Ok(1)
    };

    
    // TransitionManagerのインスタンスを作成
    let manager = TaskGenerator::new(Box::new(experiment_operation), Box::new(processing_time_func), Box::new(timing_func));
    
    // Check the variable_history
    println!("variable_history:\n{}", variable_history);

    // Generate a task
    let task = match manager.generate_task(&mut variable_history, "experiment1".to_string(), uuid::Uuid::new_v4().to_string()) {
        Ok(it) => it,
        Err(err) => panic!("{}", err),
    };

    print!("task: {:?}", task)
    
}

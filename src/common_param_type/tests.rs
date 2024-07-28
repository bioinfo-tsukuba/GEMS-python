use super::*;

// Test penalty_type
#[test]
fn test_penalty_type_to_string() {
    let penalty_type = PenaltyType::Linear { coefficient: 1 };
    println!("{:?}", penalty_type.to_string_format());

    let penalty_type = PenaltyType::LinearWithRange { lower: 1, lower_coefficient: 1, upper: 2, upper_coefficient: 2 };
    println!("{:?}", penalty_type.to_string_format());

    let penalty_type = PenaltyType::CyclicalRestPenalty { start_minute: 1, cycle_minute: 2, ranges: vec![(1, 2)]};
    println!("{:?}", penalty_type.to_string_format());
}

#[test]
fn test_penalty_type_cyclical_rest_penalty() {
    let penalty_type = PenaltyType::CyclicalRestPenalty { 
        start_minute: 0, cycle_minute: 10, ranges: vec![(1, 2)]};

    let optimal_timing = 0;

    for scheduled_timing in 0..10 {
        println!("scheduled_timing: {:?}, penalty: {:?}", scheduled_timing, penalty_type.get_penalty(scheduled_timing, optimal_timing));   
    }
}

#[test]
fn test_penalty_type_cyclical_rest_penalty_with_linear() {
    let penalty_type = PenaltyType::CyclicalRestPenaltyWithLinear { 
        start_minute: 0, cycle_minute: 10, ranges: vec![(1, 8)], coefficient: 1};

    let optimal_timing = 0;

    for scheduled_timing in 0..10 {
        println!("scheduled_timing: {:?}, penalty: {:?}", scheduled_timing, penalty_type.get_penalty(scheduled_timing, optimal_timing));   
    }
}

#[test]
fn overwrtite_global_time_manualy_test(){
    overwrtite_global_time_manualy(0);
}

#[test]
fn plot_test(){
    let scheduled_tasks = ScheduledTask::test_samples();
    ScheduledTask::plot(&scheduled_tasks, Path::new("test.png")).unwrap();
}

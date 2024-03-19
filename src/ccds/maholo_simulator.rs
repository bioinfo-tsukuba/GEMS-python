use std::{error::Error, fs, process, collections::HashMap};

use crate::{operation_mod::*, operation_information_module_for_schedule::ScheduleInformation, state_mod::{iPSStateController, Response}, plan_processor::{PassedTime, AbsoluteTime, read_passed_time, read_absolute_time}, common::*};

const TEST_RESPONSE:&str = "success_flag:1;cell_number:1.0;cell_quality:1.0";
///
/// 
const N_0_CONST: f32 = 0.05;
const K_CONST: f32 = 1.0;
const R_CONST: f32 = 0.000252219650877879;
const NOISE_DEVIATION: f32 = 0.005;
const NOISE_MEAN: f32 = 0.00;

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone, Copy)]
struct Params {
    state_id: usize,
    n_0: f32,
    k: f32,
    r: f32,
}

struct ParamsManager {
    params_list: Vec<Params>,
    params_map: HashMap<usize, Params>,
}
impl ParamsManager {
    fn new(params_list: Vec<Params>) -> Self {
        let params_map: HashMap<usize, Params> = params_list.iter().cloned()
            .map(|params| (params.state_id, params))
            .collect();

        ParamsManager {
            params_list,
            params_map,
        }
    }


    fn get_params(&self, state_id: usize) -> Option<&Params> {
        self.params_map.get(&state_id)
    }

}

use serde::Deserialize;
use csv;

fn try_read_csv_struct<T: for<'de> Deserialize<'de>>(file_path: String) -> Result<Vec<T>, Box<dyn Error>> {
    let mut data_list = Vec::new();
    let csv_text = fs::read_to_string(file_path)?;
    let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
    for result in rdr.deserialize() {
        let record: T = result?;
        data_list.push(record);
    }
    Ok(data_list)
}
use serde::Serialize;

fn write_struct<T: Serialize>(output_path: &String, items: &Vec<T>) -> Result<String, Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(output_path.clone())?;
    for item in items {
        wtr.serialize(item)?;
    }
    wtr.flush()?;
    Ok(output_path.clone())
}



pub(crate) fn run_maholo_simulator(input_directory: &String, output_directory: &String){
    let raw_operation_information_list = read_raw_operation_information_list(input_directory);
    let schedule_information_list = read_schedule_information_list(input_directory);
    let mut real_states = read_real_state_csv(input_directory);

    let earliest_scheduled_plan_index = get_earliest_scheduled_plan_index(&schedule_information_list);
    let earliest_scheduled_plan_id = schedule_information_list[earliest_scheduled_plan_index].operation_id;
    let earliest_operation_information_index = get_earliest_operation_information_index(&raw_operation_information_list, earliest_scheduled_plan_index, earliest_scheduled_plan_id);
    let earliest_scheduled_plan_state_id = raw_operation_information_list[earliest_operation_information_index].state_id;
    let earliest_state_index = get_earliest_state_index(&real_states, earliest_scheduled_plan_index, earliest_scheduled_plan_state_id);
    let earliest_scheduled_plan_type = raw_operation_information_list[earliest_operation_information_index].operation_type;


    let data_list: Vec<Params> = try_read_csv_struct(format!("{}/params.csv", input_directory)).unwrap();
    let params_manager = ParamsManager::new(data_list);


    let (passed_time, mut response) = simulate_one_plan(&raw_operation_information_list, &schedule_information_list, &mut real_states, &params_manager);
    if earliest_scheduled_plan_type == PASSAGE_OPERATION_ID{
        real_states[earliest_state_index].cell_number = N_0_CONST;
        response.1 = N_0_CONST;
    }
    let response = response_maker(response.0, response.1, response.2);
    let responses = vec![Response{ state_id: earliest_scheduled_plan_state_id, response: response}];

    match write_real_state(output_directory, &real_states) {
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(output_path)=>println!("wrote {}", output_path),
    };

    match write_maholo_response(output_directory, responses) {
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(output_path)=>println!("wrote {}", output_path),
    };
    let passed_time_struct = PassedTime{ passed_time };
    match write_passed_time(output_directory, passed_time_struct){
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(output_path)=>println!("wrote {}", output_path),
    };

    let mut absolute_time = read_absolute_time(input_directory);
    absolute_time.absolute_time += passed_time_struct.passed_time;

    match write_absolute_time(output_directory, absolute_time){
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(output_path)=>println!("wrote {}", output_path),
    };

    //Write params
    let params_list = params_manager.params_list;
    match write_struct(&format!("{}/params.csv", output_directory), &params_list) {
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(output_path)=>println!("wrote {}", output_path),
    };
}

fn write_maholo_response(output_directory: &String, reponses: Vec<Response>)->Result<String, Box<dyn Error>>{
    let output_path = format!("{}/response.csv", output_directory);
    let mut wtr = match csv::Writer::from_path(output_path.clone()){
        Err(why) => panic!("Couldn't create {}: {}", output_directory, why),
        Ok(file) => file,
    };
    for reponse_index in 0..reponses.len(){
        wtr.serialize(reponses[reponse_index].clone())?;
    }
    wtr.flush()?;
    Ok(output_path)
}

fn write_passed_time(output_directory: &String, passed_time_struct: PassedTime)->Result<String, Box<dyn Error>>{
    let output_path = format!("{}/passed_time.csv", output_directory);
    let mut wtr = match csv::Writer::from_path(output_path.clone()){
        Err(why) => panic!("Couldn't create {}: {}", output_directory, why),
        Ok(file) => file,
    };
    wtr.serialize(passed_time_struct)?;
    wtr.flush()?;
    Ok(output_path)
}

fn write_absolute_time(output_directory: &String, absolute_time_struct: AbsoluteTime)->Result<String, Box<dyn Error>>{
    let output_path = format!("{}/absolute_time.csv", output_directory);
    let mut wtr = match csv::Writer::from_path(output_path.clone()){
        Err(why) => panic!("Couldn't create {}: {}", output_directory, why),
        Ok(file) => file,
    };
    wtr.serialize(absolute_time_struct)?;
    wtr.flush()?;
    Ok(output_path)
}

fn response_maker(success_flag: usize, cell_number: f32, cell_quality: f32)->String{
    format!("success_flag:{};cell_number:{};cell_quality:{}", success_flag, cell_number, cell_quality)
}

fn update_cell_numbers(real_states: &mut Vec<iPSStateController>, passed_time: f32, params_manager: &ParamsManager){
    for state_index in 0..real_states.len(){
        let cell_number_before = real_states[state_index].cell_number;
        // real_states[state_index].cell_number = calculate_cell_proliferation_with_noise(cell_number_before, passed_time,  N_0_CONST, K_CONST, R_CONST);
        let params = params_manager.get_params(real_states[state_index].state_id).unwrap();
        let n_0 = params.n_0;
        let k = params.k;
        let r = params.r;
        real_states[state_index].cell_number = calculate_cell_proliferation(cell_number_before, passed_time,  n_0, k, r);
    }
}

fn update_cell_numbers_without_params(real_states: &mut Vec<iPSStateController>, passed_time: f32){
    for state_index in 0..real_states.len(){
        let cell_number_before = real_states[state_index].cell_number;
        // real_states[state_index].cell_number = calculate_cell_proliferation_with_noise(cell_number_before, passed_time,  N_0_CONST, K_CONST, R_CONST);
        real_states[state_index].cell_number = calculate_cell_proliferation(cell_number_before, passed_time,  N_0_CONST, K_CONST, R_CONST);
    }
}

fn update_cell_qualities(real_states: &mut Vec<iPSStateController>, passed_time: f32){
}

fn set_passed_time(state: &mut iPSStateController, passed_time: i64){
    state.passed_time = passed_time;
}

fn reset_all_passed_time(real_states: &mut Vec<iPSStateController>){
    for state_index in 0..real_states.len(){
        set_passed_time(&mut real_states[state_index], 0);
    }
}

fn update_all_passed_time(real_states: &mut Vec<iPSStateController>, passed_time: i64){
    for state_index in 0..real_states.len(){
        let current_passed_time = real_states[state_index].passed_time;
        set_passed_time(&mut real_states[state_index], current_passed_time + passed_time);
    }
}

fn simulate_one_plan(raw_operation_information_list: &Vec<RawOperationInformation>, schedule_information_list: &Vec<ScheduleInformation>, real_states: &mut Vec<iPSStateController>, params_manager:  &ParamsManager)-> (i64, (usize, f32, f32)) {
    let earliest_scheduled_plan_index = get_earliest_scheduled_plan_index(schedule_information_list);
    let earliest_scheduled_plan_id = schedule_information_list[earliest_scheduled_plan_index].operation_id;
    let earliest_operation_information_index = get_earliest_operation_information_index(raw_operation_information_list, earliest_scheduled_plan_index, earliest_scheduled_plan_id);
    let earliest_scheduled_plan_state_id = raw_operation_information_list[earliest_operation_information_index].state_id;
    let earliest_state_index = get_earliest_state_index(real_states, earliest_scheduled_plan_index, earliest_scheduled_plan_state_id);

    
    //The end point of earliest_scheduled_plan
    let passed_time = 
    OperationTypeToOperationInformation::OPERATION_TYPE_TO_OPERATION_INFORMATION_LIST[raw_operation_information_list[earliest_operation_information_index].operation_type].processing_time+//Processing time of earliest scheduled plan
    schedule_information_list[earliest_scheduled_plan_index].schedule//Start point of earliest scheduled plan
    ;
    update_cell_numbers(real_states, passed_time as f32, &params_manager);
    update_cell_qualities(real_states, passed_time as f32);
    reset_all_passed_time(real_states);

    let mut random_generator = rand::thread_rng();
    let gaussian_noise = rand_distr::Normal::new(NOISE_MEAN, NOISE_DEVIATION).unwrap();
    let noise: f32 = rand::Rng::sample(&mut random_generator, gaussian_noise);
    let mut cell_number_with_noise = real_states[earliest_state_index].cell_number + noise;
    if cell_number_with_noise < 0.00{
        cell_number_with_noise = 0.00;
    }
    if cell_number_with_noise > 1.00{
        cell_number_with_noise = 1.00;
    }
    let success_flag = 1;
    let cell_number = cell_number_with_noise;
    let cell_quality = real_states[earliest_state_index].cell_quality;
    (passed_time, (success_flag, cell_number, cell_quality))
}

// fn simulate_one_plan_v0(raw_operation_information_list: &Vec<RawOperationInformation>, schedule_information_list: &Vec<ScheduleInformation>, real_states: &mut Vec<iPSStateController>)->(i64, String){
//     let earliest_scheduled_plan_index = get_earliest_scheduled_plan_index(schedule_information_list);
//     let earliest_scheduled_plan_id = schedule_information_list[earliest_scheduled_plan_index].operation_id;
//     let earliest_operation_information_index = get_earliest_operation_information_index(raw_operation_information_list, earliest_scheduled_plan_index, earliest_scheduled_plan_id);
//     let earliest_scheduled_plan_state_id = raw_operation_information_list[earliest_operation_information_index].state_id;
//     let earliest_state_index = get_earliest_state_index(real_states, earliest_scheduled_plan_index, earliest_scheduled_plan_state_id);

    
//     //The end point of earliest_scheduled_plan
//     let passed_time = 
//     OperationTypeToOperationInformation::OPERATION_TYPE_TO_OPERATION_INFORMATION_LIST[raw_operation_information_list[earliest_operation_information_index].operation_type].processing_time+//Processing time of earliest scheduled plan
//     schedule_information_list[earliest_scheduled_plan_index].schedule//Start point of earliest scheduled plan
//     ;
//     update_cell_numbers(real_states, passed_time as f32);
//     update_cell_qualities(real_states, passed_time as f32);
//     reset_all_passed_time(real_states);

//     let response = response_maker(1, real_states[earliest_state_index].cell_number, real_states[earliest_state_index].cell_quality);
//     (passed_time, response)
// }

fn get_earliest_state_index(real_states: &Vec<iPSStateController>, earliest_scheduled_plan_index: usize, earliest_scheduled_plan_state_id: usize) -> usize {
    let mut earliest_state_index = earliest_scheduled_plan_index;
    if earliest_scheduled_plan_state_id == real_states[earliest_state_index].state_id{
    }
    else{
            earliest_state_index = earliest_scheduled_plan_index;
        for state_index in 0..real_states.len(){
            if real_states[state_index].state_id == earliest_scheduled_plan_state_id{
                earliest_state_index = state_index;
            }
        }
            assert_ne!(earliest_state_index, earliest_scheduled_plan_index);
    }
    earliest_state_index
}

fn get_earliest_scheduled_plan_index(schedule_information_list: &Vec<ScheduleInformation>)->usize{
    let mut earliest_schedule = i64::MAX;
    let mut earliest_scheduled_plan_index = 0;
    for schedule_information_index in 0..schedule_information_list.len(){
        if earliest_schedule > schedule_information_list[schedule_information_index].schedule{
            earliest_schedule = schedule_information_list[schedule_information_index].schedule;
            earliest_scheduled_plan_index = schedule_information_index;
        }
    }
    earliest_scheduled_plan_index
}

fn get_earliest_operation_information_index(raw_operation_information_list: &Vec<RawOperationInformation>, earliest_scheduled_plan_index: usize, earliest_scheduled_plan_id: usize)->usize{
    let mut earliest_operation_information_index = earliest_scheduled_plan_index;
    if earliest_scheduled_plan_id == raw_operation_information_list[earliest_operation_information_index].operation_id{
    }
    else{
        earliest_operation_information_index = earliest_scheduled_plan_index;
        for raw_opetation_information_index in 0..raw_operation_information_list.len(){
            if raw_operation_information_list[raw_opetation_information_index].operation_id == earliest_scheduled_plan_id{
                earliest_operation_information_index = raw_opetation_information_index;
            }
        }
        assert_ne!(earliest_operation_information_index, earliest_scheduled_plan_index);
    }
    earliest_operation_information_index
}

pub(crate) fn read_schedule_information_list(input_directory: &String)->Vec<ScheduleInformation>{
    fn try_read_csv(file_path: String) -> Result<Vec<ScheduleInformation>, Box<dyn Error>> {
        let mut schedule_information_list = Vec::new();
        let csv_text = fs::read_to_string(file_path)?;
        let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
        for result in rdr.deserialize() {
            // Notice that we need to provide a type hint for automatic
            // deserialization.
            let record = result?;
            schedule_information_list.push(record);
        }
        Ok(schedule_information_list)
    }
    let file_path = format!("{}/schedule.csv", input_directory);

    let schedule_information_list = match try_read_csv(file_path.to_owned()) {
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(schedule_information_list)=>schedule_information_list,
    };
    schedule_information_list
    
}



fn read_real_state_csv(input_directory: &String)->Vec<iPSStateController>{
    fn try_read_csv(file_path: String) -> Result<Vec<iPSStateController>, Box<dyn Error>> {
        let mut states = Vec::new();
        let csv_text = fs::read_to_string(file_path)?;
        let mut rdr = csv::Reader::from_reader(csv_text.as_bytes());
        for result in rdr.deserialize() {
            // Notice that we need to provide a type hint for automatic
            // deserialization.
            let record = result?;
            states.push(record);
        }
        Ok(states)
    }
    let file_path = format!("{}/real_state.csv", input_directory);

    let states = match try_read_csv(file_path.to_owned()) {
        Err(err)=>{
            println!("error running example: {}", err);
            process::exit(1);
        },
        Ok(states)=>states,
    };
    states
}

fn write_real_state(output_directory: &String, states: &Vec<iPSStateController>)->Result<String, Box<dyn Error>>{
    let output_path = format!("{}/real_state.csv", output_directory);
    let mut wtr = match csv::Writer::from_path(output_path.clone()){
        Err(why) => panic!("Couldn't create {}: {}", output_directory, why),
        Ok(file) => file,
    };
    for state_index in 0..states.len(){
        wtr.serialize(states[state_index])?;
    }
    wtr.flush()?;
    Ok(output_path)
}
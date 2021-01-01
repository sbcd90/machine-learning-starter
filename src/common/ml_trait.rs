use std::collections::HashMap;

pub struct Model {
    pub(crate) weights: Vec<f32>
}

pub trait MLAlgorithm {

    fn cross_validation_split(&self, dataset: &mut Vec<Vec<f32>>) -> HashMap<i32, Vec<Vec<f32>>>;

    fn load_csv(&self, csv_file_name: &str, positive_class: &str) -> Vec<Vec<f32>>;

    fn evaluate_algorithm(&self, dataset: &mut Vec<Vec<f32>>) -> Vec<f32>;

    fn train_model(&self, dataset: &Vec<Vec<f32>>) -> Model;

    fn predict_model(&self, dataset: &mut Vec<Vec<f32>>, model: &Model);

    fn accuracy_metric(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> f32;
}
use std::collections::HashMap;
use csv::ReaderBuilder;
use rand::prelude::thread_rng;
use rand::Rng;

pub struct Model {
    pub(crate) weights: Vec<f32>
}

pub trait MLAlgorithm {

    fn cross_validation_split(&self, dataset: &mut Vec<Vec<f32>>, n_folds: &i32) -> HashMap<i32, Vec<Vec<f32>>> {
        let mut dataset_split = HashMap::new();

        let mut fold_size: usize = dataset.len() / (*n_folds as usize);
        if fold_size == 0 {
            fold_size = 1;
        }

        for i in 1..=*n_folds {
            let mut fold: Vec<Vec<f32>> = vec![];
            while fold.len() < fold_size && dataset.len() > 0 {
                let idx = thread_rng().gen_range(0..dataset.len());

                let mut fold_elem = vec![];
                for elem in &dataset[idx] {
                    fold_elem.push(*elem);
                }
                dataset.remove(idx);
                fold.push(fold_elem);
            }
            dataset_split.insert(i, fold);
        }
        dataset_split
    }

    fn load_csv(&self, csv_file_name: &str, positive_class: &str) -> Vec<Vec<f32>> {
        let mut csv_vec = vec![];

        let mut reader = ReaderBuilder::new()
            .delimiter(b',')
            .from_path(csv_file_name)
            .unwrap();

        let mut idx = 0;
        for result in reader.records() {
            let record = result.unwrap();
            let record_iter = record.iter();

            csv_vec.push(vec![]);

            record_iter
                .for_each(|elem| {
                    let feature_value = String::from(elem).parse::<f32>();
                    let feature_value_str = String::from(elem);

                    match feature_value {
                        Ok(feature) =>
                            csv_vec[idx].push(feature),
                        _ => {
                            if feature_value_str.eq(positive_class) {
                                csv_vec[idx].push(1.0);
                            } else {
                                csv_vec[idx].push(0.0);
                            }
                        }
                    }
                });
            idx = idx+1;
        }

//        println!("{}", csv_vec[1][2]);
        csv_vec
    }

    fn evaluate_algorithm(&self, dataset: &mut Vec<Vec<f32>>) -> Vec<f32>;

    fn train_model(&self, dataset: &Vec<Vec<f32>>) -> Model;

    fn predict_model(&self, dataset: &mut Vec<Vec<f32>>, model: &Model);

    fn accuracy_metric(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> f32;
}
use crate::common::ml_trait;
use std::collections::HashMap;
use crate::common::ml_trait::Model;

pub struct Perceptron<'a> {
    n_folds: &'a i32, // n-fold cross validation
    l_rate: &'a f32, // learning rate
    n_epoch: &'a i32, // no. of epochs
}

impl<'a> Perceptron<'a> {
    pub fn new(n_folds: &'a i32,
               l_rate: &'a f32,
               n_epoch: &'a i32) -> Self {
        Perceptron {
            n_folds,
            l_rate,
            n_epoch
        }
    }

    fn predict(&self, row: &Vec<f32>, weights: &Vec<f32>) -> f32 {
        let mut activation = weights[0];

        for idx in 0..(row.len()-1) {
            activation += weights[idx+1] * row[idx];
        }

        if activation >= 0.0 {
            return 1.0
        }
        0.0
    }

    fn train_weights(&self, train_set: &Vec<Vec<f32>>) -> Vec<f32> {
        let mut weights = vec![];

        for _ in 0..train_set[0].len() as i32 {
            weights.push(0.0)
        }

        for _ in 1..=*self.n_epoch {
            for row in train_set {
                let prediction = (&self).predict(row, &weights);
//                println!("pred: {}, last elem: {}", prediction, row.last().unwrap());
                let error = row.last().unwrap() - prediction;
                weights[0] = weights[0] + self.l_rate * error;
                for idx in 0..(row.len()-1) {
                    weights[idx+1] = weights[idx+1] + self.l_rate * error * row[idx];
                }
            }
        }

        for idx in 0..train_set[0].len() {
 //           println!("{}", weights[idx]);
        }
        return weights
    }

    pub fn perceptron(&self, train_set: &Vec<Vec<f32>>, test_set: &Vec<Vec<f32>>) -> Vec<f32> {
        let mut predictions = vec![];
        let weights = self.train_weights(train_set);

        for row in test_set {
            let prediction = self.predict(row, &weights);
            predictions.push(prediction);
        }

        return predictions;
    }
}

impl<'a> ml_trait::MLAlgorithm for Perceptron<'a> {

    fn evaluate_algorithm(&self, dataset: &mut Vec<Vec<f32>>) -> Vec<f32> {
        let folds = self.cross_validation_split(dataset, self.n_folds);
        let mut scores = vec![];

        for fold in &folds {
            let mut train_set = vec![];

            for train_fold in &folds {
                if train_fold.0 != fold.0 {
                    train_set.extend_from_slice(&train_fold.1);
                }
            }

            let test_set = fold.1;
            let mut actual = vec![];
            for row in test_set {
                actual.push(row[row.len()-1]);
            }

            let predicted = self.perceptron(&train_set, &test_set);
            let accuracy = self.accuracy_metric(&actual, &predicted);

            scores.push(accuracy);
        }

        scores
    }

    fn train_model(&self, dataset: &Vec<Vec<f32>>) -> Model {
        let weights = self.train_weights(dataset);

        Model {
            weights
        }
    }

    fn predict_model(&self, dataset: &mut Vec<Vec<f32>>, model: &Model) {
        for row in dataset {
            let pred = self.predict(row, &model.weights);

            let len = row.len()-1;
            row[len] = pred;
        }
    }

    fn accuracy_metric(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> f32 {
        let mut correct = 0.0;

        let mut idx = 0;
        while idx < actual.len() {
            if actual[idx] == predicted[idx] {
                correct = correct + 1.0;
            }

            idx = idx + 1;
        }

        (correct / actual.len() as f32) * 100.0
    }
}
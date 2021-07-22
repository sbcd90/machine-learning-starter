use crate::common::ml_trait;
use crate::MLAlgorithm;

use rand::prelude::thread_rng;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::cmp::max;
use crate::common::ml_trait::Model;
use std::str::FromStr;
use std::rc::Rc;

pub struct RandomForest<'a> {
    n_folds: &'a i32,
    max_depth: &'a i32,
    min_size: &'a i32,
    sample_size: &'a f32
}

struct GiniMetrics {
    b_index: i32,
    b_value: f32,
    b_score: f32,
    b_groups: (Vec<Vec<f32>>, Vec<Vec<f32>>),
    left: Option<Rc<GiniMetrics>>,
    right: Option<Rc<GiniMetrics>>,
    left_leaf: Option<f32>,
    right_leaf: Option<f32>
}

impl<'a> RandomForest<'a> {
    fn new(n_folds: &'a i32,
           max_depth: &'a i32,
           min_size: &'a i32,
           sample_size: &'a f32) -> Self {
        RandomForest {
            n_folds,
            max_depth,
            min_size,
            sample_size
        }
    }

    fn test_split(&self, idx: &usize, value: &f32,
                  dataset: &Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut left = vec![];
        let mut right = vec![];

        for row in dataset {
            if row[*idx] < *value {
                left.push(row.clone());
            } else {
                right.push(row.clone());
            }
        }

        (left, right)
    }

    fn gini_index(&self, groups: Vec<&Vec<Vec<f32>>>, classes: &Vec<f32>) -> f32 {
        let n_instances = groups[0].len() + groups[1].len();

        let mut gini = 0.0;

        for group in groups {
            let size = group.len() as f32;

            if size == 0.0 {
                continue;
            }

            let mut score = 0.0;
            for class_val in classes {
                let mut cnt = 0.0;
                for row in group {
                    if row[row.len()-1] == *class_val {
                        cnt += 1.0;
                    }
                }

                let p = cnt / size;
                score += p * p;
            }

            gini += (1.0 - score) * (size / n_instances as f32);
        }

        gini
    }

    fn get_split(&self, dataset: &Vec<Vec<f32>>, n_features: &'a f32) -> GiniMetrics {
        let mut class_set = HashSet::new();

        for row in dataset {
            class_set.insert(row[row.len() - 1].to_string());
        }
        let class_values = Vec::from_iter(class_set.iter().map(|f| f32::from_str(f.as_str()).unwrap()));

        let mut gini_metrics = GiniMetrics {
            b_index: 999,
            b_value: 999.0,
            b_score: 999.0,
            b_groups: (vec![], vec![]),
            left: None,
            right: None,
            left_leaf: None,
            right_leaf: None
        };

        let mut features = vec![];
        while features.len() < *n_features as usize {
            let idx = thread_rng().gen_range(0..dataset[0].len());

            if !features.contains(&idx) {
                features.push(idx);
            }
        }

        for idx in features {
            for row in dataset {
                let groups = self.test_split(&idx, &row[idx], dataset);
                let gini = self.gini_index(vec![&groups.0, &groups.1], &class_values);

                if gini < gini_metrics.b_score {
                    gini_metrics = GiniMetrics {
                        b_index: idx as i32,
                        b_value: row[idx],
                        b_score: gini,
                        b_groups: groups,
                        left: None,
                        right: None,
                        left_leaf: None,
                        right_leaf: None
                    };
                }
            }
        }
        gini_metrics
    }

    fn to_terminal(&self, group: &Vec<Vec<f32>>) -> f32 {
        let mut outcome_counts: Vec<i32> = vec![];

        for row in group {
            if row[row.len() - 1] == 1.0 {
                outcome_counts[1] = outcome_counts[1] + 1;
            } else {
                outcome_counts[0] = outcome_counts[0] + 1;
            }
        }

        if outcome_counts[0] >= outcome_counts[1] {
            0.0
        } else {
            1.0
        }
    }

    fn split(&self, gini_metrics: &GiniMetrics, n_features: &'a f32,
             depth: &'a i32) -> GiniMetrics {
        let (left, right) = &gini_metrics.b_groups;
        let mut new_gini_metrics = GiniMetrics {
            b_index: gini_metrics.b_index.clone(),
            b_value: gini_metrics.b_value.clone(),
            b_score: gini_metrics.b_score.clone(),
            b_groups: (gini_metrics.b_groups.0.clone(), gini_metrics.b_groups.1.clone()),
            left: None,
            right: None,
            left_leaf: None,
            right_leaf: None
        };

        if left.len() == 0 || right.len() == 0 {
            let mut merge_all = vec![];

            for row in left {
                merge_all.push(row.clone());
            }

            for row in right {
                merge_all.push(row.clone());
            }

            new_gini_metrics.left_leaf = Some(self.to_terminal(&merge_all));
            new_gini_metrics.right_leaf = Some(self.to_terminal(&merge_all));
            return new_gini_metrics;
        }

        if depth >= self.max_depth {
            new_gini_metrics.left_leaf = Some(self.to_terminal(left));
            new_gini_metrics.right_leaf = Some(self.to_terminal(right));
            return new_gini_metrics;
        }

        if left.len() <= *self.min_size as usize {
            new_gini_metrics.left_leaf = Some(self.to_terminal(left));
        } else {
            let left_val = Box::new(self.get_split(left, n_features));
            new_gini_metrics.left =
                Some(Rc::new(self.split(&left_val, n_features, &(depth+1))));
        }

        if right.len() <= *self.min_size as usize {
            new_gini_metrics.right_leaf = Some(self.to_terminal(right));
        } else {
            let right_val = Box::new(self.get_split(right, n_features));
            new_gini_metrics.right =
                Some(Rc::new(self.split(&right_val, n_features, &(depth+1))));
        }

        new_gini_metrics
    }

    fn predict(&self, gini_metrics: Rc<GiniMetrics>, row: &Vec<f32>) -> f32 {
        if row[gini_metrics.b_index as usize] < gini_metrics.b_value {
            match gini_metrics.left_leaf {
                Some(left_leaf) => left_leaf,
                None => self.predict(Rc::clone(gini_metrics.left.as_ref().unwrap()), row)
            }
        } else {
            match gini_metrics.right_leaf {
                Some(right_leaf) => right_leaf,
                None => self.predict(Rc::clone(gini_metrics.right.as_ref().unwrap()), row)
            }
        }
    }

    fn build_tree(&self, train_set: &Vec<Vec<f32>>,
                  n_features: &'a f32) -> GiniMetrics {
        let gini_metrics = self.get_split(train_set, n_features);
        let new_gini_metrics = self.split(&gini_metrics, n_features, &1);
        new_gini_metrics
    }

    fn subsample(&self, train_set: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut sample = vec![];

        let n_sample = (train_set.len() as f32 / *self.sample_size) as i32;

        while sample.len() < n_sample as usize {
            let idx = thread_rng().gen_range(0..train_set.len());

            let mut row = vec![];
            for col in &train_set[idx] {
                row.push(*col);
            }

            sample.push(row);
        }

        sample
    }

    fn bagging_predict(&self, trees: &Vec<Rc<GiniMetrics>>, row: &Vec<f32>) -> f32 {
        let mut predictions = vec![];
        for tree in trees {
            let prediction = self.predict(Rc::clone(tree), row);
            predictions.push(prediction);
        }

        let mut cnt0 = 0;
        for prediction in &predictions {
            if *prediction == 0.0 {
                cnt0 = cnt0 + 1;
            }
        }

        if cnt0 >= (predictions.len() - cnt0) {
            0.0
        } else {
            1.0
        }
    }

    pub fn random_forest(&self, train_set: &Vec<Vec<f32>>, test_set: &Vec<Vec<f32>>,
                         n_trees: &'a i32, n_features: &'a f32) -> Vec<f32> {
        let mut trees = vec![];

        for i in 1..=*n_trees {
            let sample = self.subsample(train_set);
            let tree = self.build_tree(&sample, n_features);
            trees.push(Rc::new(tree));
        }

        let mut predictions = vec![];
        for row in test_set {
            let prediction = self.bagging_predict(&trees, row);
            predictions.push(prediction);
        }

        predictions
    }

    fn evaluate_random_forest(&self, dataset: &mut Vec<Vec<f32>>,
                              n_trees: &'a i32, n_features: &'a f32) -> Vec<f32> {
        let folds = self.cross_validation_split(dataset, self.n_folds);

        let mut scores = vec![];

        for fold in &folds {
            let mut train_set = vec![];

            for train_fold in &folds {
                if train_fold.0 != fold.0 {
                    train_set.extend_from_slice(train_fold.1);
                }
            }

            let test_set = fold.1;
            let mut actual = vec![];
            for row in test_set {
                actual.push(row[row.len() - 1]);
            }

            let predicted = self.random_forest(&train_set, test_set, n_trees, n_features);
            let accuracy = self.accuracy_metric(&actual, &predicted);
            scores.push(accuracy);
        }

        scores
    }
}

impl<'a> ml_trait::MLAlgorithm for RandomForest<'a> {

    fn evaluate_algorithm(&self, dataset: &mut Vec<Vec<f32>>) -> Vec<f32> {
        let n_features = ((dataset[0].len() - 1) as f32).sqrt();
        let n_trees_vals = [1, 5, 10];
        let mut acc_scores = vec![];

        for n_trees_val in &n_trees_vals {
            let scores = self.evaluate_random_forest(dataset, n_trees_val, &n_features);
            println!("Trees: {}", n_trees_val);
            println!("Scores: {:?}", scores);

            let sum_scores = |scores| {
                let mut sum = 0.0;
                for score in scores {
                    sum += score;
                }

                sum
            };

            println!("Mean Accuracy: {}", sum_scores(&scores) / scores.len() as f32);
            acc_scores.extend_from_slice(&scores);
        }

        acc_scores
    }

    fn train_model(&self, dataset: &Vec<Vec<f32>>) -> Model {
        Model {
            weights: vec![]
        }
    }

    fn predict_model(&self, dataset: &mut Vec<Vec<f32>>, model: &Model) {

    }

    fn accuracy_metric(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> f32 {
        let mut correct = 0.0;
        for i in 0..actual.len() {
            if actual[i] == predicted[i] {
                correct += 1.0;
            }
        }
        correct / actual.len() as f32 * 100.0
    }
}
#[macro_use]
mod common;
mod classification;
mod utils;

pub use classification::perceptron::Perceptron;
pub use common::ml_trait::MLAlgorithm;
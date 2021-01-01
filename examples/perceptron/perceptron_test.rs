use machine_learning_starter::Perceptron;
use machine_learning_starter::MLAlgorithm;

fn main() {
    let perceptron = &Perceptron::new(&4, &0.01, &100);
    let mut csv_vec = perceptron.load_csv("train_df.csv", "R");

    /*for row in &csv_vec {
        for col in row {
            print!("{} ", col);
        }
        println!();
    }*/

    let scores = perceptron.evaluate_algorithm(&mut csv_vec);

    for score in scores {
        println!("{}", score);
    }

    let training_set = perceptron.load_csv("train_df.csv", "R");
    let test_set = perceptron.load_csv("test_df.csv", "R");

    let mut pred_set = vec![];

    for row in &test_set {

        let mut pred_row = vec![];
        let mut idx = 0;
        for col in row {
            if idx > 0 {
                pred_row.push(*col);
            }
            idx = idx + 1;
        }
        pred_set.push(pred_row);
    }

    let model = perceptron.train_model(&training_set);
    perceptron.predict_model(&mut pred_set, &model);

    let mut submission_set = vec![];

    let mut xidx = 0;
    for row in &test_set {
        let mut submission_row = vec![];

        let mut yidx = 0;
        for col in row {
            if yidx == 0 {
                submission_row.push(*col);
            }
            if yidx == row.len()-2 {
                submission_row.push(pred_set[xidx][yidx]);
            }
            yidx = yidx + 1;
        }

        submission_set.push(submission_row);
        xidx = xidx + 1;
    }

    for row in &submission_set {
        println!("{},{}", row[0], row[1]);
    }
}
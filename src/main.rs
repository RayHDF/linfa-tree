use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::array;

fn main() {
    let features = array![
        [5.1, 3.5],
        [4.9, 3.0],
        [4.7, 3.2],
        [7.0, 3.2],
        [6.4, 3.2],
        [6.9, 3.1],
    ];

    let targets = array![
        0, 
        0, 
        0, 
        1,
        1,
        1,
    ];

    let dataset = Dataset::new(features, targets);

    let model = DecisionTree::params()
        .max_depth(Some(3))
        .fit(&dataset)
        .expect("Failed to fit decision tree model");


    println!("Decision tree model trained successfully");

    let new_observation = array![[6.9, 3.1]];
    let prediction = model.predict(&new_observation);

    println!("Prediction for [5.0, 3.3]: {:?}", prediction[0]);

    // println!("\nModel structure");
    // println!("{:?}", model);
}

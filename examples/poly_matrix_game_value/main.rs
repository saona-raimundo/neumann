//! In this example, the value of a polynomial matrix game is computed in two different ways:
//! 1) Evaluating it and solving the corresponding matrix game.
//! 2) Computing the functional form and evaluating this function.
use ndarray::array;
use neumann::{PolyMatrixGame, value_positivity::ValuePositivity};
use preexplorer::prelude::*;

fn main() {
    // Setting
    let poly_matrix_game =
        PolyMatrixGame::from(vec![array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]]]);

    // Computing
    let grid = ndarray::Array1::<f64>::linspace(-0., 10., 100);
    let values: Vec<f64> = grid.iter().map(|eps| poly_matrix_game.eval(*eps).value()).collect();

    // Exact form of the value function
    let value_function = poly_matrix_game.functional_form().output;
    let numer: polynomials::Polynomial<f64> = value_function.numer()
    	.iter()
    	.map(|v| *v as f64)
    	.collect::<Vec<f64>>()
    	.into();
    let denom: polynomials::Polynomial<f64> = value_function.denom()
    	.iter()
    	.map(|v| *v as f64)
    	.collect::<Vec<f64>>()
    	.into();
    let exact_values: Vec<f64> = grid.iter()
    	.map(|eps| numer.eval(*eps).unwrap() / denom.eval(*eps).unwrap())
    	.collect();

    // Drawing
    let mut computed_1 = (&grid, &values).preexplore();
    computed_1.set_title("matrix game");
    let mut computed_2 = (&grid, &exact_values).preexplore();
    computed_2.set_title("functional form");

    // On top of each other
    (computed_1 + computed_2)
    	.set_title("Comparing computations of the value")
        .set_xlabel("epsilon")
        .set_ylabel("value")
        .plot("value_positive")
        .unwrap();
    // Difference
    let difference: Vec<f64> =  values.iter().zip(&exact_values).map(|(v, u)| v - u).collect();
    (&grid, &difference).preexplore()
    	.set_title("Difference between the methods")
    	.set_xlabel("epsilon")
        .set_ylabel("value")
        .plot("difference_value")
        .unwrap();
}

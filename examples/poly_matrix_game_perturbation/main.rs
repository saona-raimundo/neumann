fn main() {
    unimplemented!();
}

// //! In this example, a linear matrix game is studied under positive and negative perturbations.
// use ndarray::array;
// use neumann::{value_positivity::ValuePositivity, PolyMatrixGame};
// use preexplorer::prelude::*;

// fn main() {
//     // Setting
//     let poly_matrix_game =
//         PolyMatrixGame::from(vec![array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]]]);

//     // Computing
//     let grid = ndarray::Array1::<f64>::linspace(-0.5, 0.5, 100);
//     let values: Vec<f64> = grid
//         .iter()
//         .map(|eps| poly_matrix_game.eval(*eps).value())
//         .collect();

//     // Exact form of the value function
//     let value_function = poly_matrix_game.functional_form().output;
//     let numer: polynomials::Polynomial<f64> = value_function
//         .numer()
//         .iter()
//         .map(|v| *v as f64)
//         .collect::<Vec<f64>>()
//         .into();
//     let denom: polynomials::Polynomial<f64> = value_function
//         .denom()
//         .iter()
//         .map(|v| *v as f64)
//         .collect::<Vec<f64>>()
//         .into();
//     let exact_values: Vec<f64> = grid
//         .iter()
//         .map(|eps| numer.eval(*eps).unwrap() / denom.eval(*eps).unwrap())
//         .collect();

//     // Drawing
//     let mut computed_1 = (&grid, &values).preexplore();
//     computed_1.set_title("matrix game");
//     let mut computed_2 = (&grid, &exact_values).preexplore();
//     computed_2.set_title("functional form");

//     // On top of each other
//     (computed_1 + computed_2)
//         .set_title("Comparing computations of the value")
//         .set_xlabel("epsilon")
//         .set_ylabel("value")
//         .plot("value_positive")
//         .unwrap();
//     // Difference
//     let difference: Vec<f64> = values
//         .iter()
//         .zip(&exact_values)
//         .map(|(v, u)| v - u)
//         .collect();
//     (&grid, &difference)
//         .preexplore()
//         .set_title("Difference between the methods")
//         .set_xlabel("epsilon")
//         .set_ylabel("value")
//         .plot("difference_value")
//         .unwrap();
// }

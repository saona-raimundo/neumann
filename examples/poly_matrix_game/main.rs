use ndarray::array;
use neumann::PolyMatrixGame;
use preexplorer::prelude::*;

fn main() {
    // Setting
    let poly_matrix_game =
        PolyMatrixGame::from(vec![array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]]]);

    // Computing
    let grid = ndarray::Array1::<f64>::linspace(-0., 1., 100);
    let values = grid.iter().map(|eps| poly_matrix_game.eval(*eps).value());

    // Drawing
    (&grid, values)
        .preexplore()
        .set_title("Value function")
        .set_xlabel("epsilon")
        .set_ylabel("value")
        .plot("value_positive")
        .unwrap();
}

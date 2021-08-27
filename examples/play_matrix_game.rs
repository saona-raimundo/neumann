use neumann::prelude::*;

fn main() {
    let matrix_game = MatrixGame::from([[0., 0.], [-1., 1.]]);
    matrix_game.play().unwrap();
}

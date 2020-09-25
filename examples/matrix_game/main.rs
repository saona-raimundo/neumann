use ndarray::array;
use neumann::{Playable, MatrixGame};


fn main() {
    let matrix_game =
        MatrixGame::from( array![[0, 0], [-1, 1]]);

    matrix_game.play()
}
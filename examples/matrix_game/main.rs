use nalgebra::Matrix2;
use neumann::{traits::CliPlayable, MatrixGame};

fn main() {
    let matrix = Matrix2::new(0, 0, -1, 1);
    let matrix_game = MatrixGame::from(matrix);
    matrix_game.play()
}

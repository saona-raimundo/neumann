//! All transformations between different types.

use nalgebra::Matrix;

use crate::MatrixGame;

impl<T, R, C, S> From<Matrix<T, R, C, S>> for MatrixGame<T, R, C, S> {
    fn from(matrix: Matrix<T, R, C, S>) -> Self {
        MatrixGame { matrix }
    }
}

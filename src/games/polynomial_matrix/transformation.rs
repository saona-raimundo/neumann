//! All generic on type transformations between different types.

use core::fmt::Debug;
use nalgebra::{Matrix, SMatrix};

use crate::games::polynomial_matrix::types::SPolynomialMatrixGame;
use crate::{MatrixGame, PolynomialMatrixGame, SMatrixGame};

impl<T, R, C, S> From<Vec<MatrixGame<T, R, C, S>>> for PolynomialMatrixGame<T, R, C, S> {
    fn from(matrices: Vec<MatrixGame<T, R, C, S>>) -> Self {
        PolynomialMatrixGame { matrices }
    }
}

impl<T, R, C, S> From<Vec<Matrix<T, R, C, S>>> for PolynomialMatrixGame<T, R, C, S> {
    fn from(matrices: Vec<Matrix<T, R, C, S>>) -> Self {
        let matrices = matrices.into_iter().map(|m| m.into()).collect();
        PolynomialMatrixGame { matrices }
    }
}

impl<T, const R: usize, const C: usize, const K: usize> From<[[[T; C]; R]; K]>
    for SPolynomialMatrixGame<T, R, C>
where
    T: Clone + PartialEq + Debug + 'static,
{
    fn from(matrices: [[[T; C]; R]; K]) -> Self {
        let matrices = matrices
            .iter()
            .map(|matrix| {
                let matrix = SMatrix::from_fn(|r, c| matrix[r][c].clone());
                SMatrixGame::from(matrix)
            })
            .collect();
        PolynomialMatrixGame { matrices }
    }
}

#[cfg(feature = "interoperability")]
mod ndarray {
    use ndarray::Array2;

    use crate::{DMatrixGame, DPolynomialMatrixGame};

    // # Note
    //
    // ndarray might implement constant generics in the future
    impl<T> From<Vec<Array2<T>>> for DPolynomialMatrixGame<T>
    where
        DMatrixGame<T>: From<Array2<T>>,
    {
        fn from(matrices: Vec<Array2<T>>) -> Self {
            let matrices: Vec<DMatrixGame<T>> = matrices.into_iter().map(|m| m.into()).collect();
            DPolynomialMatrixGame::from(matrices)
        }
    }
}

//! All generic on type transformations between different types.

use core::fmt::Debug;
use nalgebra::{DMatrix, Dim, Matrix, RawStorage, SMatrix, Scalar};
use polynomials::Polynomial;

use crate::{
    DPolynomialMatrixGame, MatrixGame, PolynomialMatrixGame, SMatrixGame, SPolynomialMatrixGame,
};

impl<T, R, C, S> PolynomialMatrixGame<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    pub fn into_dynamic(self) -> DPolynomialMatrixGame<T> {
        let (nrows, ncols) = self.shape();
        let matrices: Vec<_> = self
            .matrices
            .into_iter()
            .map(|matrix_game| {
                DMatrix::from_fn(nrows, ncols, |r, c| matrix_game.matrix[(r, c)].clone())
            })
            .collect();
        DPolynomialMatrixGame::from(matrices)
    }
}

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

impl<T, R, C, S> Into<DMatrix<Polynomial<T>>> for PolynomialMatrixGame<T, R, C, S>
where
    T: Clone,
    Polynomial<T>: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn into(self) -> DMatrix<Polynomial<T>> {
        let (nrows, ncols) = self.shape();
        DMatrix::<Polynomial<T>>::from_fn(nrows, ncols, |r, c| {
            let vec: Vec<_> = (0..self.matrices.len())
                .map(|k| (self.matrices[k].matrix)[(r, c)].clone())
                .collect();
            Polynomial::from(vec)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_matrix_of_polynomials() {
        let p = PolynomialMatrixGame::from([[[0]]]);
        let _matrix_of_polynomials: DMatrix<Polynomial<usize>> = p.into();
    }
}

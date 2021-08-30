//! All generic on type transformations between different types.

use core::fmt::Debug;
use nalgebra::{Matrix, SMatrix};

use crate::games::matrix::types::SMatrixGame;
use crate::MatrixGame;

impl<T, R, C, S> From<Matrix<T, R, C, S>> for MatrixGame<T, R, C, S> {
    fn from(matrix: Matrix<T, R, C, S>) -> Self {
        MatrixGame { matrix }
    }
}

impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for SMatrixGame<T, R, C>
where
    T: Clone + PartialEq + Debug + 'static,
{
    fn from(matrix: [[T; C]; R]) -> Self {
        let matrix = SMatrix::from_fn(|r, c| matrix[r][c].clone());
        MatrixGame { matrix }
    }
}

#[cfg(feature = "interoperability")]
mod ndarray {
    use crate::MatrixGame;
    use nalgebra::{DMatrix, Dynamic, Scalar};
    use ndarray::Array2;

    use crate::games::matrix::types::DMatrixGame;

    // # Note
    //
    // ndarray might implement constant generics in the future
    impl<T> From<Array2<T>> for DMatrixGame<T>
    where
        T: Scalar,
    {
        fn from(matrix: Array2<T>) -> Self {
            let matrix: DMatrix<T> = {
                // TODO fix, taken from nshare because of dependency issues
                let std_layout = matrix.is_standard_layout();
                let nrows = Dynamic::new(matrix.nrows());
                let ncols = Dynamic::new(matrix.ncols());
                let mut res = DMatrix::<T>::from_vec_generic(nrows, ncols, matrix.into_raw_vec());
                if std_layout {
                    // This can be expensive, but we have no choice since nalgebra VecStorage is always
                    // column-based.
                    res.transpose_mut();
                }
                res
            };
            MatrixGame { matrix }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::MatrixGame;

    #[test]
    fn from_array2() {
        let array = ndarray::array![[0, 1], [2, 3]];
        let matrix_game = MatrixGame::from(array);
        assert_eq!(matrix_game.matrix, nalgebra::matrix![0, 1; 2, 3]);
    }
}

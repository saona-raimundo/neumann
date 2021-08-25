use nalgebra::{
    allocator::Reallocator, Const, DefaultAllocator, Dim, DimDiff, DimSub, Matrix, RawStorage,
    Scalar, Storage, U1,
};
// use std::collections::HashSet;
// use std::fmt;

// More implementations
mod play;
mod solvable;
mod transformations;

/// [Matrix games](https://en.wikipedia.org/wiki/Zero-sum_game) are finite zero-sum two-player games.
///
/// # Examples
///
/// Rock-paper-scisors.
/// ```
/// # use nalgebra::Matrix3;
/// # use neumann::MatrixGame;
/// let rewards = Matrix3::new(0, 1, -1, 1, -1, 0, -1, 0, 1);
/// MatrixGame::from(rewards);
/// ```
#[derive(Clone)]
pub struct MatrixGame<T, R, C, S> {
    pub matrix: Matrix<T, R, C, S>,
}

impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> MatrixGame<T, R, C, S> {
    /// The shape of this matrix returned as the tuple (number of rows, number of columns).
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }
    /// Return `true` if the matrix game has no entries.
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    /// Returns `true` if both players have the same number of possible actions.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is a square game.
    /// ```
    /// # use nalgebra::Matrix3;
    /// # use neumann::MatrixGame;
    /// let rewards = Matrix3::new(0, 1, -1, 1, -1, 0, -1, 0, 1);
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert!(matrix_game.is_square());
    /// ```
    ///
    /// A 2x3 game is not square.
    /// ```
    /// # use nalgebra::Matrix2x3;
    /// # use neumann::MatrixGame;
    /// let rewards = Matrix2x3::new(0, 1, -1, 0, -1, 2);
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert!(!matrix_game.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        let shape = self.matrix.shape();
        shape.0 == shape.1
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> MatrixGame<T, R, C, S>
where
    R: DimSub<U1>,
    DefaultAllocator: Reallocator<T, R, C, DimDiff<R, U1>, C>,
{
    /// Returns a matrix game with one action less for the row player.
    ///
    /// # Examples
    ///
    /// Forgetting about the first action for the row player.
    /// ```
    /// # use nalgebra::{Matrix3, Matrix2x3};
    /// # use neumann::MatrixGame;
    /// let rewards = Matrix3::new(0, 1, -1, 1, -1, 0, -1, 0, 1);
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.remove_row(0);
    /// assert_eq!(sub_matrix_game.matrix, Matrix2x3::new(1, -1, 0, -1, 0, 1));
    /// ```
    pub fn remove_row(
        self,
        i: usize,
    ) -> MatrixGame<
        T,
        <R as DimSub<Const<1_usize>>>::Output,
        C,
        <DefaultAllocator as nalgebra::allocator::Allocator<
            T,
            <R as DimSub<Const<1_usize>>>::Output,
            C,
        >>::Buffer,
    > {
        let sub_matrix = self.matrix.remove_row(i);
        MatrixGame { matrix: sub_matrix }
    }
}

impl<T: Scalar, R: Dim, C: Dim, S: Storage<T, R, C>> MatrixGame<T, R, C, S>
where
    C: DimSub<U1>,
    DefaultAllocator: Reallocator<T, R, C, R, DimDiff<C, U1>>,
{
    /// Returns a matrix game with one action less for the column player.
    ///
    /// # Examples
    ///
    /// Forgetting about the last action for the column player.
    /// ```
    /// # use nalgebra::{Matrix3, Matrix3x2};
    /// # use neumann::MatrixGame;
    /// let rewards = Matrix3::new(0, 1, -1, 1, -1, 0, -1, 0, 1);
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.remove_column(2);
    /// assert_eq!(sub_matrix_game.matrix, Matrix3x2::new(0, 1, 1, -1, -1, 0));
    /// ```
    pub fn remove_column(
        self,
        i: usize,
    ) -> MatrixGame<
        T,
        R,
        <C as DimSub<Const<1_usize>>>::Output,
        <DefaultAllocator as nalgebra::allocator::Allocator<
            T,
            R,
            <C as DimSub<Const<1_usize>>>::Output,
        >>::Buffer,
    > {
        let sub_matrix = self.matrix.remove_column(i);
        MatrixGame { matrix: sub_matrix }
    }
}

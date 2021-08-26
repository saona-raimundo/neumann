use nalgebra::{
    allocator::{Allocator, Reallocator},
    Const, DefaultAllocator, Dim, DimDiff, DimSub, Matrix, OMatrix, Owned, RawStorage, Scalar,
    Storage, U1,
};
// use std::collections::HashSet;
// use std::fmt;
use core::fmt;

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
/// # use neumann::MatrixGame;
/// MatrixGame::from([[0, 1, -1], [1, -1, 0], [-1, 0, 1]]);
/// ```
#[derive(Clone)]
pub struct MatrixGame<T, R, C, S> {
    pub matrix: Matrix<T, R, C, S>,
}

impl<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>> MatrixGame<T, R, C, S> {
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
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0, 1, -1], [1, -1, 0], [-1, 0, 1]]);;
    /// assert!(matrix_game.is_square());
    /// ```
    ///
    /// A 2x3 game is not square.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0, 1, -1], [0, -1, 2]]);;
    /// assert!(!matrix_game.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        let shape = self.matrix.shape();
        shape.0 == shape.1
    }

    /// The number of rows of this matrix game.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    /// The number of columns of this matrix game.
    pub fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    /// The shape of this matrix game returned as the tuple (number of rows, number of columns).
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
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
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0, 1, -1], [1, -1, 0], [-1, 0, 1]]);
    /// let sub_matrix_game = matrix_game.remove_row(0);
    /// assert_eq!(sub_matrix_game.shape(), (2, 3));
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
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0, 1, -1], [1, -1, 0], [-1, 0, 1]]);
    /// let sub_matrix_game = matrix_game.remove_column(2);
    /// assert_eq!(sub_matrix_game.shape(), (3, 2));
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

impl<T, R, C, S> fmt::Display for MatrixGame<T, R, C, S>
where
    T: Scalar + fmt::Display,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: ,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
    DefaultAllocator: Allocator<T, R, C>,
{
    pub fn map<T2: Scalar, F: FnMut(T) -> T2>(&self, f: F) -> MatrixGame<T2, R, C, Owned<T2, R, C>>
    where
        T: Scalar,
        DefaultAllocator: Allocator<T2, R, C>,
    {
        let matrix: OMatrix<T2, R, C> = self.matrix.map(f);
        MatrixGame { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map() {
        MatrixGame::from([[1], [2]]).map(|v| v as f64);
    }
}

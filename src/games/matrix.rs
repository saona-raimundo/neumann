use core::{
    fmt,
    ops::{AddAssign, MulAssign},
};
use nalgebra::{
    allocator::{Allocator, Reallocator},
    constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    ClosedAdd, ClosedMul, Const, DefaultAllocator, Dim, DimDiff, DimSub, Matrix, OMatrix, Owned,
    RawStorage, Scalar, Storage, StorageMut, U1,
};
use num_traits::Zero;

// More implementations
#[cfg(feature = "play")]
mod play;
mod solvable;
mod transformation;
pub mod types;

/// [Matrix games](https://en.wikipedia.org/wiki/Zero-sum_game) are finite zero-sum two-player games.
///
/// # Implementation
///
/// It is a thin wrapper from a `nalgebra::Matrix`. See types definitions to make your life easier.
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
        let shape = self.shape();
        shape.0 == shape.1
    }

    /// The number of rows of this matrix game.
    pub fn nrows(&self) -> usize {
        let (nrows, _) = self.shape();
        nrows
    }

    /// The number of columns of this matrix game.
    pub fn ncols(&self) -> usize {
        let (_, ncols) = self.shape();
        ncols
    }

    /// The shape of this matrix game returned as the tuple (number of rows, number of columns).
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.shape()
    }
}

impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Returns `true` if `self` is equal to the additive identity.
    ///
    /// # Examples
    ///
    /// A zero matrix.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0, 0], [0, 0]]);
    /// assert!(matrix_game.is_zero());
    /// ```

    // # Note
    //
    // The Zero trait can not be implemented since zero() -> Self
    // has no meaning for Dynamic dimension.
    pub fn is_zero(&self) -> bool {
        let (nrows, ncols) = self.shape();
        for row in 0..nrows {
            for col in 0..ncols {
                if !self.matrix[(row, col)].is_zero() {
                    return false;
                }
            }
        }
        true
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

impl<T, R: Dim, C: Dim, S: fmt::Debug> fmt::Debug for MatrixGame<T, R, C, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        formatter
            .debug_struct("MatrixGame")
            .field("matrix", &self.matrix)
            .finish()
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

impl<T, R, R2, C, C2, S, S2> PartialEq<MatrixGame<T, R2, C2, S2>> for MatrixGame<T, R, C, S>
where
    T: Scalar + PartialEq,
    C: Dim,
    C2: Dim,
    R: Dim,
    R2: Dim,
    S: RawStorage<T, R, C>,
    S2: RawStorage<T, R2, C2>,
{
    fn eq(&self, right: &MatrixGame<T, R2, C2, S2>) -> bool {
        self.shape() == right.shape()
            && self
                .matrix
                .iter()
                .zip(right.matrix.iter())
                .all(|(l, r)| l == r)
    }
}

impl<T, R: Dim, C: Dim, S> MulAssign<T> for MatrixGame<T, R, C, S>
where
    T: Scalar + ClosedMul,
    S: StorageMut<T, R, C>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.matrix *= rhs;
    }
}

impl<'a, T, R1, C1, R2, C2, SA, SB> AddAssign<&'a MatrixGame<T, R2, C2, SB>>
    for MatrixGame<T, R1, C1, SA>
where
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    T: Scalar + ClosedAdd,
    SA: StorageMut<T, R1, C1>,
    SB: Storage<T, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    fn add_assign(&mut self, rhs: &'a MatrixGame<T, R2, C2, SB>) {
        self.matrix += &rhs.matrix;
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

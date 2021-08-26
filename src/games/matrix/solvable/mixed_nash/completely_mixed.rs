//! Completely mixed related functions

use nalgebra::{
    allocator::{Allocator, Reallocator},
    constraint::{AreMultipliable, ShapeConstraint},
    Const, DefaultAllocator, Dim, DimSub, Dynamic, Storage,
};

use crate::{traits::Solvable, MatrixGame};

impl<R, C, S> MatrixGame<f64, R, C, S>
where
    R: Dim + DimSub<Const<1>>,
    C: Dim,
    S: Storage<f64, R, C> + Clone,
    DefaultAllocator: Allocator<f64, R, C>
        + Allocator<f64, C, R>
        + Allocator<f64, R>
        + Allocator<f64, C>
        + Allocator<f64, <R as DimSub<Const<1_usize>>>::Output, C>
        + Allocator<f64, C, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, C>
        + Reallocator<f64, R, C, <R as DimSub<Const<1_usize>>>::Output, C>,
    ShapeConstraint: AreMultipliable<R, C, Dynamic, Dynamic>
        + AreMultipliable<C, R, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, R, C>
        + AreMultipliable<Dynamic, Dynamic, C, R>
        + AreMultipliable<<R as DimSub<Const<1_usize>>>::Output, C, Dynamic, Dynamic>
        + AreMultipliable<C, <R as DimSub<Const<1_usize>>>::Output, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, <R as DimSub<Const<1_usize>>>::Output, C>
        + AreMultipliable<Dynamic, Dynamic, C, <R as DimSub<Const<1_usize>>>::Output>,
{
    /// Returns `true` if both players have the same number of possible actions
    /// and a unique optimal strategy which has full support[^1].
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is a completely-mixed game.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0., 1., -1.], [1., -1., 0.], [-1., 0., 1.]]);
    /// assert!(matrix_game.is_completely_mixed());
    /// ```
    ///
    /// A game with dominant strategies is not completely-mixed.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0., 2., 1.], [2., 0., 1.], [-1., -1., -1.]]);
    /// assert!(!matrix_game.is_completely_mixed());
    /// ```
    ///
    /// [^1]: Kaplansky, I. (1945).
    ///       [*A Contribution to Von Neumann's Theory of Games*](https://www.jstor.org/stable/1969164).
    ///       Annals of Mathematics, 46(3), second series, 474-479.
    ///       doi:10.2307/1969164
    pub fn is_completely_mixed(&self) -> bool {
        if !self.is_square() {
            false
        } else if self.nrows() == 1 {
            true
        } else {
            let full_value = self.value();
            let sub_value = self.clone().remove_row(self.best_row_removal()).value();
            full_value > sub_value
        }
    }
}

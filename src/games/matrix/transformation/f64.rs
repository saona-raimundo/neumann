//! All transformations exclusive for `f64`.
// # Note
//
// This is mainly because the restriction implementing `Solvable` trait.

use nalgebra::{
    allocator::{Allocator, Reallocator},
    constraint::{AreMultipliable, ShapeConstraint},
    Const, DMatrix, DefaultAllocator, Dim, DimSub, Dynamic, RawStorage, Storage,
};
use std::collections::HashSet;

use crate::games::matrix::types::DMatrixGame;
use crate::{traits::Solvable, MatrixGame};

impl<R, C, S> MatrixGame<f64, R, C, S>
where
    R: Dim + DimSub<Const<1>>,
    <R as DimSub<Const<1_usize>>>::Output: Dim,
    C: Dim,
    S: Storage<f64, R, C> + Clone,
    DefaultAllocator: Allocator<f64, <R as DimSub<Const<1_usize>>>::Output, C>
        + Allocator<f64, C, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, C>
        + Reallocator<f64, R, C, <R as DimSub<Const<1_usize>>>::Output, C>,
    ShapeConstraint: AreMultipliable<<R as DimSub<Const<1_usize>>>::Output, C, Dynamic, Dynamic>
        + AreMultipliable<C, <R as DimSub<Const<1_usize>>>::Output, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, <R as DimSub<Const<1_usize>>>::Output, C>
        + AreMultipliable<Dynamic, Dynamic, C, <R as DimSub<Const<1_usize>>>::Output>,
{
    /// Returns the least beneficial action for the row player.
    ///
    /// In other words, if this action is prohibited for the row player,
    /// then the value of the restricted game diminishes the least.
    ///
    /// # Panics
    ///
    /// If the game is empty.
    ///
    /// # Examples
    ///
    /// Forgetting about the worst action for the row player.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0., 1.], [1., 0.], [-1., -1.]]);
    /// assert_eq!(matrix_game.best_row_removal(), 2);
    /// ```
    pub fn best_row_removal(&self) -> usize {
        assert!(!self.is_empty());
        (0..self.nrows())
            .map(|i| (i, self.clone().remove_row(i).value()))
            .max_by(|(_, v), (_, u)| {
                if v < u {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .map(|(i, _)| i)
            .unwrap()
    }
}

impl<R, C, S> MatrixGame<f64, R, C, S>
where
    C: Dim + DimSub<Const<1>>,
    <C as DimSub<Const<1_usize>>>::Output: Dim,
    R: Dim,
    S: Storage<f64, R, C> + Clone,
    DefaultAllocator: Allocator<f64, <C as DimSub<Const<1_usize>>>::Output, R>
        + Allocator<f64, R, <C as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, <C as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, R>
        + Reallocator<f64, R, C, R, <C as DimSub<Const<1_usize>>>::Output>,
    ShapeConstraint: AreMultipliable<R, <C as DimSub<Const<1_usize>>>::Output, Dynamic, Dynamic>
        + AreMultipliable<<C as DimSub<Const<1_usize>>>::Output, R, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, R, <C as DimSub<Const<1_usize>>>::Output>
        + AreMultipliable<Dynamic, Dynamic, <C as DimSub<Const<1_usize>>>::Output, R>,
{
    /// Returns the least beneficial action for the column player.
    ///
    /// In other words, if this action is prohibited for the column player,
    /// then the value of the restricted game increases the least.
    ///
    /// # Panics
    ///
    /// If the game is empty.
    ///
    /// # Examples
    ///
    /// Forgetting about the worst action for the column player.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[1., 0.], [1., -1.]]);
    /// assert_eq!(matrix_game.best_column_removal(), 0);
    /// ```
    pub fn best_column_removal(&self) -> usize {
        assert!(!self.is_empty());
        (0..self.ncols())
            .map(|i| (i, self.clone().remove_column(i).value()))
            .min_by(|(_, v), (_, u)| {
                if v < u {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .map(|(i, _)| i)
            .unwrap()
    }
}

/// # Mixed Nash equilirbium
///
/// Implementations related to this solution concept.
impl<R, C, S> MatrixGame<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorage<f64, R, C>,
{
    /// Reduces the matrix game to a square sub-game with the same value and
    /// whose optimal strategies are also optimal in the original game,
    /// together with the indexed dropped from the original game (in the corresponding dimension).
    ///
    /// If the matrix game has dimensions `m`x`n`,
    /// then the resulting matrix game has dimensions `min(m, n)`x`min(m, n)`.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors can not be reduce further, so it stays the same.
    /// ```
    /// # use neumann::MatrixGame;
    /// let (matrix_game, dropped_indexes) =
    ///     MatrixGame::from([[0., 1., -1.], [1., -1., 0.], [-1., 0., 1.]]).reduce_to_square();
    /// assert_eq!(matrix_game.shape(), (3, 3));
    /// assert!(dropped_indexes.is_empty());
    /// ```
    ///
    /// A game with a rectangular shape can be reduced.
    /// ```
    /// # use neumann::MatrixGame;
    /// # use nalgebra::dmatrix;
    /// let (matrix_game, dropped_indexes) =
    ///     MatrixGame::from([[0., 2.], [2., 0.], [-1., -1.]]).reduce_to_square();
    /// assert_eq!(matrix_game.matrix, dmatrix![0., 2.; 2., 0.]);
    /// assert_eq!(dropped_indexes, vec![2]);
    /// ```
    pub fn reduce_to_square(&self) -> (DMatrixGame<f64>, Vec<usize>) {
        let num_extra_actions = self.nrows().max(self.ncols()) - self.nrows().min(self.ncols());
        let mut dropped_indexes = Vec::with_capacity(num_extra_actions);
        let shape = self.shape();
        let mut matrix_game = MatrixGame::from(DMatrix::<f64>::from_iterator(
            shape.0,
            shape.1,
            self.matrix.iter().cloned(),
        ));
        let value = matrix_game.value();
        if self.nrows() > self.ncols() {
            for row in 0..self.nrows() {
                if matrix_game.clone().remove_row(row).value() >= value {
                    dropped_indexes.push(row);
                }
                if dropped_indexes.len() == num_extra_actions {
                    for row in &dropped_indexes {
                        matrix_game = matrix_game.remove_row(*row);
                    }
                    break;
                }
            }
        }
        if self.nrows() < self.ncols() {
            for column in 0..self.ncols() {
                if matrix_game.clone().remove_column(column).value() <= value {
                    dropped_indexes.push(column);
                }
                if dropped_indexes.len() == num_extra_actions {
                    for column in &dropped_indexes {
                        matrix_game = matrix_game.remove_column(*column);
                    }
                    break;
                }
            }
        }
        (matrix_game, dropped_indexes)
    }

    /// Returns the description of a sub-game with the following properties
    /// - *completely-mixed*[^1]
    /// - same value as the original matrix game
    /// - optimal strategies, extended by zeros, are also optimal in the original matrix game
    ///
    /// The representation is a tuple of `indexes_row`, `indexes_column` and `sub-game`,
    /// where `indexes_xxx` are the indexes of the actions in the original matrix game
    /// which lead to the sub-game.
    ///
    /// See [is_completely_mixed] method for an explanation of completely-mixed matrix games.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is already completely-mixed so its kernel is the whole game.
    /// ```
    /// # use neumann::MatrixGame;
    /// let matrix_game = MatrixGame::from([[0., 1., -1.], [1., -1., 0.], [-1., 0., 1.]]);
    /// let (indexes_rows, indexes_columns, sub_matrix_game) = matrix_game.kernel_completely_mixed();
    /// assert_eq!(indexes_rows, vec![0, 1, 2]);
    /// assert_eq!(indexes_columns, vec![0, 1, 2]);
    /// assert_eq!(sub_matrix_game.shape(), (3, 3));
    /// ```
    ///
    /// A game with a rectangular shape (row player has a dominated action).
    /// ```
    /// # use neumann::MatrixGame;
    /// # use nalgebra::dmatrix;
    /// let matrix_game = MatrixGame::from([[1., 2.], [2., 1.], [0., 0.]]);
    /// let (indexes_rows, indexes_columns, sub_matrix_game) = matrix_game.kernel_completely_mixed();
    /// assert_eq!(indexes_rows, vec![0, 1]);
    /// assert_eq!(indexes_columns, vec![0, 1]);
    /// assert_eq!(sub_matrix_game.matrix, dmatrix![1., 2.; 2., 1.]);
    /// ```
    ///
    /// A game where naive elimination of dominated strategies might not preserve optimal strategies.
    /// ```
    /// # use neumann::MatrixGame;
    /// # use nalgebra::dmatrix;
    /// let matrix_game = MatrixGame::from([[1., 1.], [1., 0.]]);
    /// let (indexes_rows, indexes_columns, sub_matrix_game) = matrix_game.kernel_completely_mixed();
    /// assert_eq!(indexes_rows, vec![0]);
    /// assert_eq!(indexes_columns, vec![0]);
    /// // assert_eq!(sub_matrix_game.matrix, dmatrix![1.]);
    /// ```
    ///
    /// [is_completely_mixed]: struct.MatrixGame.html#method.is_completely_mixed
    /// [^1]: Kaplansky, I. (1945).
    ///       [*A Contribution to Von Neumann's Theory of Games*](https://www.jstor.org/stable/1969164).
    ///       Annals of Mathematics, 46(3), second series, 474-479.
    ///       doi:10.2307/1969164
    // # Note
    //
    // This suffers from two step blindness
    // Doing the reduction in two-steps at a time allows to maintain optimal strategies in the original game!
    pub fn kernel_completely_mixed(&self) -> (Vec<usize>, Vec<usize>, DMatrixGame<f64>) {
        // Setting up kernel
        let mut indexes_row: HashSet<usize> = (0..self.nrows()).collect();
        let mut indexes_column: HashSet<usize> = (0..self.ncols()).collect();
        // Square matrix game
        let (mut sub_matrix_game, dropped_indexes) = self.reduce_to_square();
        if self.nrows() > self.ncols() {
            for row in dropped_indexes {
                indexes_row.remove(&row);
            }
        } else {
            for col in dropped_indexes {
                indexes_row.remove(&col);
            }
        }
        // Iterative two-step reduction
        while !sub_matrix_game.is_completely_mixed() {
            let row_index = sub_matrix_game.best_row_removal();
            let rectangular_sub_matrix_game = sub_matrix_game.clone().remove_row(row_index);
            let (row_strategy, _column_strategy, value) = sub_matrix_game.some_solution().unwrap(); // Never fails
            for col_index in 0..rectangular_sub_matrix_game.ncols() {
                let square_sub_matrix_game =
                    rectangular_sub_matrix_game.clone().remove_column(col_index);
                let column_strategy = {
                    let mut sub_column_strategy =
                        square_sub_matrix_game.some_solution_for_player(1).unwrap(); // Never fails
                    sub_column_strategy.insert(col_index, 0.0);
                    sub_column_strategy
                };
                if sub_matrix_game.is_solution((row_strategy.clone(), column_strategy, value)) {
                    sub_matrix_game = square_sub_matrix_game;
                    indexes_row.remove(&row_index);
                    indexes_column.remove(&col_index);
                    break;
                }
            }
        }
        // Sorting indexes
        let indexes_row = {
            let mut unsorted: Vec<usize> = indexes_row.drain().collect();
            unsorted.sort();
            unsorted
        };
        let indexes_column = {
            let mut unsorted: Vec<usize> = indexes_column.drain().collect();
            unsorted.sort();
            unsorted
        };
        (indexes_row, indexes_column, sub_matrix_game)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_completely_mixed() {
        use nalgebra::dmatrix;
        let matrix_game = MatrixGame::from([[1., 2.], [2., 1.], [0., 0.]]);
        let (indexes_rows, indexes_columns, sub_matrix_game) =
            matrix_game.kernel_completely_mixed();
        assert_eq!(indexes_rows, vec![0, 1]);
        assert_eq!(indexes_columns, vec![0, 1]);
        assert_eq!(sub_matrix_game.matrix, dmatrix![1., 2.; 2., 1.]);
    }
}

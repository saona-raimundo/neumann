// use itertools::Itertools;
// use ndarray::{Array2, Axis};
// use num_rational::Ratio;
use core::{
    fmt,
    ops::{AddAssign, MulAssign},
};
use nalgebra::{ClosedMul, Dim, RawStorage, Scalar, StorageMut};
use num_traits::Zero;

use crate::games::matrix::MatrixGame;

#[cfg(feature = "play")]
mod play;
mod solvable;
mod transformation;
pub mod types;

/// Polynomial matrix games are [Matrix Games](https://en.wikipedia.org/wiki/Zero-sum_game) whith perturbation terms in terms of polynomials.
///
/// # Examples
///
/// Error term pushes the optimal strategy in a different direction than the error free optimal strategy.
/// ```
/// # use neumann::PolynomialMatrixGame;
/// PolynomialMatrixGame::from([[[0, 1], [1, 0]], [[0, 2], [1, 0]]]);
/// ```
///
/// Error term diminishes the reward for the row player, but does not changes optimal strategies.
/// ```
/// # use neumann::PolynomialMatrixGame;
/// PolynomialMatrixGame::from([[[0, 1], [1, 0]], [[0, -1], [-1, 0]]]);
/// ```
pub struct PolynomialMatrixGame<T, R, C, S> {
    matrices: Vec<MatrixGame<T, R, C, S>>,
}

impl<T, R, C, S> PolynomialMatrixGame<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Return `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Returns `true` if both players have the same number of possible actions.
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
    ///
    /// # Panics
    ///
    /// If `self` is empty.
    pub fn shape(&self) -> (usize, usize) {
        assert!(!self.is_empty());
        self.matrices[0].shape()
    }
}

impl<T, R, C, S> PolynomialMatrixGame<T, R, C, S>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Returns the degree of `self`.
    ///
    /// The degree is the highest power with a non-zero Matrix game.
    ///
    /// # Examples
    ///
    /// Zero PolynomialMatrixGame have no degree.
    /// ```
    /// # use neumann::PolynomialMatrixGame;
    /// let p = PolynomialMatrixGame::from([[[0]]]);
    /// assert_eq!(p.degree(), None);
    /// ```
    ///
    /// Constant PolynomialMatrixGame, ie no perturbation term.
    /// ```
    /// # use neumann::PolynomialMatrixGame;
    /// let p = PolynomialMatrixGame::from([[[1]]]);
    /// assert_eq!(p.degree(), Some(0));
    /// ```
    ///
    /// Linearly perturbed matrix game.
    /// ```
    /// # use neumann::PolynomialMatrixGame;
    /// let p = PolynomialMatrixGame::from([[[1]], [[1]]]);
    /// assert_eq!(p.degree(), Some(1));
    /// ```
    pub fn degree(&self) -> Option<usize> {
        let mut degree = self.matrices.len();
        for _ in 0..self.matrices.len() {
            if self.matrices[degree - 1].is_zero() {
                degree -= 1;
            } else {
                return Some(degree - 1);
            }
        }
        None
    }

    /// Leading coefficient (a `MatrixGame`).
    pub fn leading_coefficient(&self) -> Option<&MatrixGame<T, R, C, S>> {
        self.degree()
            // Never fails since degree is less than len
            .map(|degree| self.matrices.get(degree).unwrap())
    }
}

impl<T, R, C, S> PolynomialMatrixGame<T, R, C, S> {
    /// Get Matrix games
    ///
    /// The order corresponds to increasing the order of perturbation.
    pub fn coefficients(self) -> Vec<MatrixGame<T, R, C, S>> {
        self.matrices
    }
}

impl<T, R, C, S> PolynomialMatrixGame<T, R, C, S>
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Returns the matrix game that corresponds to evaluate perturbation at `epsilon`.
    ///
    /// # Implementation
    ///
    /// [Horner's method](https://en.wikipedia.org/wiki/Horner%27s_method)
    ///
    /// # Panics
    ///
    /// If there are no coefficients.
    ///
    /// # Examples
    ///
    /// Evaluating at one is the same as summing the matrix games.
    /// ```
    /// # use neumann::{MatrixGame, PolynomialMatrixGame};
    /// let p = PolynomialMatrixGame::from([[[0, 1], [1, 0]], [[0, -1], [-1, 0]]]);
    /// assert_eq!(p.eval(&1), MatrixGame::from([[0, 0], [0, 0]]));
    /// ```
    pub fn eval<'a, 'b>(&'b self, epsilon: &'a T) -> MatrixGame<T, R, C, S>
    where
        T: ClosedMul + Clone,
        S: StorageMut<T, R, C>,
        MatrixGame<T, R, C, S>: Clone + MulAssign<T> + AddAssign<&'b MatrixGame<T, R, C, S>>,
    {
        assert!(!self.matrices.is_empty());
        let mut sum: MatrixGame<T, R, C, S> = self.matrices.last().unwrap().clone(); // Never fails by previous check
        for i in (0..self.matrices.len() - 1).rev() {
            sum *= epsilon.clone();
            sum += &self.matrices[i];
        }
        sum
    }
}

impl<T, R, C, S> fmt::Display for PolynomialMatrixGame<T, R, C, S>
where
    MatrixGame<T, R, C, S>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();
        for i in 0..self.matrices.len() {
            string += &format!("{}", self.matrices[i]);
            if i == 1 {
                string += "eps";
            }
            if i > 1 {
                string += &format!(" eps^{}", i);
            }
            if i < self.matrices.len() - 1 {
                string += "\n+\n"
            }
        }
        write!(f, "{}", string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatrixGame;

    #[test]
    fn eval() {
        let p = PolynomialMatrixGame::from([[[0, 1], [1, 0]], [[0, -1], [-1, 0]]]);
        assert_eq!(p.eval(&1), MatrixGame::from([[0, 0], [0, 0]]));
    }

    #[test]
    fn display() {
        let p = PolynomialMatrixGame::from([[[0, 1], [1, 0]], [[0, -1], [-1, 0]]]);
        println!("{}", p);
    }
}

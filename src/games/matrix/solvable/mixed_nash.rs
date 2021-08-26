use core::marker::PhantomData;
use minilp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use nalgebra::{
    allocator::{Allocator, Reallocator},
    constraint::{AreMultipliable, ShapeConstraint},
    Const, DMatrix, DVector, DefaultAllocator, Dim, DimSub, Dynamic, Scalar, Storage,
};

use crate::{equilibria::MixedNash, traits::Solvable, MatrixGame};

impl<T, R, C, S> Solvable<MixedNash<f64>> for MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone,
    S: Storage<T, R, C> + Clone,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<T, C, R>
        + Allocator<T, R, C>
        + Allocator<f64, R, C>
        + Allocator<f64, C, R>
        + Allocator<f64, R>
        + Allocator<f64, C>,
    ShapeConstraint: AreMultipliable<R, C, Dynamic, Dynamic>
        + AreMultipliable<C, R, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, R, C>
        + AreMultipliable<Dynamic, Dynamic, C, R>,
{
    type PlayerStrategy = Vec<f64>;
    type Solution = (Vec<f64>, Vec<f64>, f64);
    type SolutionIter = Iter<f64>;
    type Value = f64;

    /// By Nash, this is always true.
    fn is_solvable(&self) -> bool {
        true
    }

    /// # Implementation
    ///
    /// Uses the Linear Programming method.
    fn some_solution(&self) -> Option<<Self as Solvable<MixedNash<f64>>>::Solution> {
        let (row_strategy, value) = self.solve_row();
        let column_strategy = self.some_solution_for_player(1).unwrap(); // Never fails
        Some((row_strategy, column_strategy, value))
    }

    fn is_solution(&self, proposal: <Self as Solvable<MixedNash<f64>>>::Solution) -> bool {
        let value = self.value().unwrap();
        let matrix = self.matrix.map(|v| -> f64 { v.into() });
        let row_reward = {
            let strategy = DMatrix::from_vec(1, proposal.0.len(), proposal.0); // PR for RowDVector::from(proposal.0);
            (strategy * matrix.clone()).min()
        };
        let column_reward = {
            let strategy = DVector::from(proposal.1);
            (matrix * strategy).min()
        };

        (proposal.2 >= value) && (row_reward >= value) && (column_reward >= value)
    }
    /// # Implementation
    ///
    /// Uses the Linear Programming method.
    ///
    /// # Panics
    ///
    /// If `player` is not `0` or `1`.
    fn some_solution_for_player(
        &self,
        player: usize,
    ) -> Option<<Self as Solvable<MixedNash<f64>>>::PlayerStrategy> {
        assert!(player < 2);
        match player {
            0 => {
                let (strategy, _) = self.solve_row();
                Some(strategy)
            }
            1 => {
                let matrix = self.matrix.map(|v| -> f64 { -v.into() }).transpose();
                MatrixGame { matrix }.some_solution_for_player(0)
            }
            _ => unreachable!(),
        }
    }
    /// # Implementation
    ///
    /// TODO
    fn all_solutions(&self) -> Iter<f64> {
        todo!()
    }
    fn value(&self) -> Option<Self::Value> {
        let (_strategy, value) = self.solve_row();
        Some(value)
    }
}

#[derive(Debug)]
pub struct Iter<F> {
    _phantom: PhantomData<F>,
}
impl<F> Iterator for Iter<F> {
    type Item = (Vec<F>, Vec<F>, F);
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        todo!()
    }
}

/// # Mixed Nash equilirbium
///
/// Implementations related to this solution concept.
impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    /// Construct the optimization problem the row player has to solve.
    fn row_player_lp(&self) -> (Problem, (Vec<Variable>, Variable)) {
        // Define LP
        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Setting
        let (rows, columns) = self.shape();

        // Add row player strategy
        let row_strategy: Vec<_> = (0..rows)
            .map(|_| problem.add_var(0.0, (0.0, 1.0)))
            .collect();
        // Add value variable
        let value_variable = problem.add_var(1.0, (-std::f64::INFINITY, std::f64::INFINITY));

        // Probabiltiy constrains
        let ones = vec![1.0; rows];
        problem.add_constraint(
            row_strategy.clone().into_iter().zip(ones),
            ComparisonOp::Eq,
            1.0,
        );

        // Value constrains
        for column_action in 0..columns {
            let rewards: Vec<f64> = self
                .matrix
                .column(column_action)
                .into_iter()
                .map(|x| x.clone().into())
                .collect();
            let mut constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards)
                .collect::<Vec<(Variable, f64)>>();
            constrain.push((value_variable, -1.0));
            problem.add_constraint(constrain, ComparisonOp::Ge, 0.0);
        }

        (problem, (row_strategy, value_variable))
    }

    /// Returns an optimal strategy for the row player and the value of the game, i.e. the value this player can ensure.
    fn solve_row(&self) -> (Vec<f64>, f64) {
        let (problem, (strategy_variables, _)) = self.row_player_lp();

        // Solve
        let solution = problem.solve().unwrap();

        // Retrieve the solution
        let value = solution.objective();
        let mut optimal_row_strategy = Vec::new();
        for var in strategy_variables {
            optimal_row_strategy.push(solution[var]);
        }

        (optimal_row_strategy, value)
    }
}

impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone,
    R: Dim + DimSub<Const<1>> + Clone,
    <R as DimSub<Const<1_usize>>>::Output: Dim,
    C: Dim + Clone,
    S: Storage<T, R, C> + Clone,
    DefaultAllocator: Allocator<T, C, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<T, <R as DimSub<Const<1_usize>>>::Output, C>
        + Allocator<f64, <R as DimSub<Const<1_usize>>>::Output, C>
        + Allocator<f64, C, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, <R as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, C>
        + Reallocator<T, R, C, <R as DimSub<Const<1_usize>>>::Output, C>,
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

impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone,
    C: Dim + DimSub<Const<1>> + Clone,
    <C as DimSub<Const<1_usize>>>::Output: Dim,
    R: Dim + Clone,
    S: Storage<T, R, C> + Clone,
    DefaultAllocator: Allocator<T, R, <C as DimSub<Const<1_usize>>>::Output>
        + Allocator<T, <C as DimSub<Const<1_usize>>>::Output, R>
        + Allocator<f64, <C as DimSub<Const<1_usize>>>::Output, R>
        + Allocator<f64, R, <C as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, <C as DimSub<Const<1_usize>>>::Output>
        + Allocator<f64, R>
        + Reallocator<T, R, C, R, <C as DimSub<Const<1_usize>>>::Output>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use test_case::test_case;

    use core::fmt::Debug;

    #[test_case( [[0, 1], [1, 0]],  0.5 ; "positive value")]
    #[test_case( [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  0.0 ; "rock-paper-scisors")]
    fn value<T, const R: usize, const C: usize>(matrix: [[T; C]; R], expected_value: f64)
    where
        T: Clone + PartialEq + Debug + 'static + Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        let value = matrix_game.value().unwrap();
        assert_ulps_eq!(value, expected_value, max_ulps = 1);
    }

    #[test_case( [[0, 1], [1, 0]],  vec![0.5, 0.5] ; "positive value")]
    #[test_case( [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  vec![1./3., 1./3., 1./3.] ; "rock-paper-scisors")]
    fn some_solution_for_player_row<T, const R: usize, const C: usize>(
        matrix: [[T; C]; R],
        expected_strategy: Vec<f64>,
    ) where
        T: Clone + PartialEq + Debug + 'static + Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        let optimal_row_strategy = matrix_game.some_solution_for_player(0).unwrap();
        for i in 0..expected_strategy.len() {
            assert_ulps_eq!(optimal_row_strategy[i], expected_strategy[i], max_ulps = 1);
        }
    }
}

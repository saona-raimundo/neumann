use core::marker::PhantomData;
use minilp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use nalgebra::{
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
    DMatrix, DVector, DefaultAllocator, Dim, Dynamic, Storage,
};

use crate::{equilibria::MixedNash, traits::Solvable, MatrixGame};

mod completely_mixed;

/// Mixed Nash equilibrium solution for Matrix games.
///
/// # Known issues
///
/// The precision of the implementation does not allow to solve games
/// with small entries. For example, consider the following.
/// ```
/// # use neumann::{MatrixGame, traits::Solvable};
/// let small_entries =[[5e-10, -1e-10], [-1e-10, 5e-10]];
/// let matrix_game = MatrixGame::from(small_entries);
/// assert!(matrix_game.value().unwrap() >= 5e-10); // correct answer is 4e-10
/// ```
impl<R, C, S> Solvable<MixedNash<f64>> for MatrixGame<f64, R, C, S>
where
    S: Storage<f64, R, C> + Clone,
    R: Dim,
    C: Dim,
    DefaultAllocator:
        Allocator<f64, R, C> + Allocator<f64, C, R> + Allocator<f64, R> + Allocator<f64, C>,
    ShapeConstraint: AreMultipliable<R, C, Dynamic, Dynamic>
        + AreMultipliable<C, R, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, R, C>
        + AreMultipliable<Dynamic, Dynamic, C, R>,
{
    type PlayerStrategy = Vec<f64>;
    /// Row optimal strategy, column optimal strategy, value.
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

    /// Returns true if `proposal` is a mixed Nash exuilibrium.
    ///
    /// # Errors
    ///
    /// If dimensions do not match.
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
        assert!(
            player < 2,
            "There are only two players in a matrix game (0 and 1), you asked for player {}",
            player
        );
        match player {
            0 => {
                let (strategy, _) = self.solve_row();
                Some(strategy)
            }
            1 => {
                let matrix = -self.matrix.transpose();
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
impl<R, C, S> MatrixGame<f64, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<f64, R, C>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_ulps_eq};
    use test_case::test_case;

    use core::fmt::Debug;

    #[test_case( [[0, 1], [1, 0]],  0.5 ; "positive value")]
    #[test_case( [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  0.0 ; "rock-paper-scisors")]
    fn value<T, const R: usize, const C: usize>(matrix: [[T; C]; R], expected_value: f64)
    where
        T: Clone + PartialEq + Debug + 'static + Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix).map(|x| x.into());
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
        let matrix_game = MatrixGame::from(matrix).map(|x| x.into());
        let optimal_row_strategy = matrix_game.some_solution_for_player(0).unwrap();
        for i in 0..expected_strategy.len() {
            assert_ulps_eq!(optimal_row_strategy[i], expected_strategy[i], max_ulps = 1);
        }
    }

    #[test_case( [[0, 1], [1, 0]],  (vec![0.5, 0.5], vec![0.5, 0.5], 0.5) ; "positive value")]
    #[test_case( [[0, 1], [1, 0], [-1, -1]],  (vec![0.5, 0.5, 0.], vec![0.5, 0.5], 0.5) ; "positive value with extra strategy")]
    #[test_case( [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  (vec![1./3., 1./3., 1./3.], vec![1./3., 1./3., 1./3.], 0.0) ; "rock-paper-scisors")]
    #[test_case(
        [
            [1, 1, -2],
            [1, -2, 1],
            [-2, 1, 1]
        ],
        (vec![1./3., 1./3., 1./3.], vec![1./3., 1./3., 1./3.], 0.0);
        "symmetric game 3x3"
    )]
    #[test_case( [[1, 1, 1, -3], [1, 1, -3, 1], [1, -3, 1, 1], [-3, 1, 1, 1]],  (vec![1./4., 1./4., 1./4., 1./4.], vec![1./4., 1./4., 1./4., 1./4.], 0.0) ; "symmetric game 4x4")]
    #[test_case(
        [
            [11, 11, 11, -550],
            [11, 11, -550, 11],
            [11, -550, 11, 11],
            [-550, 11, 11, 11]
        ],
        (vec![1./4., 1./4., 1./4., 1./4.], vec![1./4., 1./4., 1./4., 1./4.], -129.25);
        "symmetric game with bigger entries"
    )]
    fn some_solution<T, const R: usize, const C: usize>(
        matrix: [[T; C]; R],
        expected_solution: (Vec<f64>, Vec<f64>, f64),
    ) where
        T: Clone + PartialEq + Debug + 'static + Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix).map(|x| x.into());
        let (optimal_row_strategy, optimal_column_strategy, value) =
            matrix_game.some_solution().unwrap();
        for i in 0..expected_solution.0.len() {
            assert_abs_diff_eq!(
                optimal_row_strategy[i],
                expected_solution.0[i],
                epsilon = 1e-10
            );
        }
        for j in 0..expected_solution.1.len() {
            assert_abs_diff_eq!(
                optimal_column_strategy[j],
                expected_solution.1[j],
                epsilon = 1e-10
            );
        }
        assert_ulps_eq!(value, expected_solution.2, max_ulps = 1);
    }
}

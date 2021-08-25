//! All implementations of `Solvable` trait.

use core::marker::PhantomData;
use minilp::{ComparisonOp, OptimizationDirection, Problem, Variable};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, RawStorage, Scalar, SimdComplexField};

use crate::{equilibria::MixedNash, traits::Solvable, MatrixGame};

impl<T, R, C, S> Solvable<MixedNash<f64>> for MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone + SimdComplexField,
    S: RawStorage<T, R, C>,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<T, C, R> + Allocator<T, R, C>,
{
    type PlayerStrategy = Vec<f64>;
    type Solution = (Vec<f64>, Vec<f64>, f64);
    type SolutionIter = Iter<f64>;

    fn is_solvable(&self) -> bool {
        true
    }

    fn some_solution(&self) -> Option<<Self as Solvable<MixedNash<f64>>>::Solution> {
        let (row_strategy, value) = self.solve_row();
        let column_strategy = self.some_solution_for_player(1).unwrap(); // Never fails
        Some((row_strategy, column_strategy, value))
    }

    fn is_solution(&self, _proposal: <Self as Solvable<MixedNash<f64>>>::Solution) -> bool {
        todo!()
    }
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
                let matrix = -self.matrix.adjoint();
                MatrixGame { matrix }.some_solution_for_player(0)
            }
            _ => unreachable!(),
        }
    }

    fn all_solutions(&self) -> Iter<f64> {
        todo!()
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

/// Implementation using `minilp` crate
impl<T, R, C, S> MatrixGame<T, R, C, S>
where
    T: Scalar + Into<f64> + Clone,
    R: Dim,
    C: Dim,
    S: RawStorage<T, R, C>,
{
    /// Construct the optimization problem the row player has to solve.
    pub fn row_player_lp(&self) -> (Problem, (Vec<Variable>, Variable)) {
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

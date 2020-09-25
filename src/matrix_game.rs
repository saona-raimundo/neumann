pub use polynomial::PolyMatrixGame;

mod polynomial;

// use crate::traits::Game;
use ndarray::{Array1, Array2, Axis};
use std::fmt;

/// [Matrix games](https://en.wikipedia.org/wiki/Zero-sum_game) are finite zero-sum two-player games.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixGame {
    matrix: Array2<f64>,
}

impl MatrixGame {
    /// Returns the reward matrix for the row player.
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Returns an optimal strategy for the row player and the value of the game, i.e. the value this player can ensure.
    pub fn row_solve(&self) -> (Vec<f64>, f64) {
        // Define LP
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);

        // Setting
        let dimensions = self.matrix.shape();

        // Add row player strategy
        let mut row_strategy = Vec::with_capacity(dimensions[0]);
        for _ in 0..dimensions[0] {
            row_strategy.push(problem.add_var(0.0, (0.0, 1.0)));
        }
        // Add value variable
        let value_variable = problem.add_var(1.0, (-std::f64::INFINITY, std::f64::INFINITY));

        // Probabiltiy constrains
        let mut ones = Vec::with_capacity(row_strategy.len());
        for _ in 0..row_strategy.len() {
            ones.push(1.0);
        }
        problem.add_constraint(
            row_strategy.clone().into_iter().zip(ones),
            minilp::ComparisonOp::Eq,
            1.0,
        );

        // Value constrains
        for column_action in 0..dimensions[1] {
            let rewards = self.matrix.index_axis(Axis(1), column_action);
            let mut constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            constrain.push((value_variable, -1.0));
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, 0.0);
        }

        // Solve
        let solution = problem.solve().unwrap();

        // Retrieve the solution
        let value = solution.objective();
        let mut optimal_row_strategy = Vec::new();
        for var in row_strategy {
            optimal_row_strategy.push(solution[var]);
        }

        (optimal_row_strategy, value)
    }

    /// Returns the value of the game.
    pub fn value(&self) -> f64 {
        self.row_solve().1
    }

    /// Returns a Nash equilibrium and the value of the game.
    pub fn solve(&self) -> (Vec<f64>, Vec<f64>, f64) {
        let column_matrix: Array2<f64> = self.matrix().t().map(|x| -x).to_owned();
        let column_matrix_game = MatrixGame::from(column_matrix);

        // Solve
        let (optimal_row_strategy, value) = self.row_solve();
        let (optimal_column_strategy, _) = column_matrix_game.row_solve();

        (optimal_row_strategy, optimal_column_strategy, value)
    }

    /// Number of row actions
    pub fn row_actions(&self) -> usize {
        self.matrix.shape()[0]
    }

    /// Number of column actions
    pub fn column_actions(&self) -> usize {
        self.matrix.shape()[1]
    }

    fn input_strategy(&self) -> Array1<f64> {
        println!("Enter you mixed strategy, one probability at a time.");

        let mut weights = Array1::from_elem(self.row_actions(), 0.);
        for i in 0..self.row_actions() {
            let mut weight = String::new();

            std::io::stdin()
                .read_line(&mut weight)
                .expect("Failed to read line");

            let mut ns = fasteval::EmptyNamespace;
            match fasteval::ez_eval(&weight, &mut ns) {
                Ok(val) => {
                	if val.is_sign_positive() {
                		weights[i] = val	
                	} else {
                		eprintln!("Probabilities must be greater or equal to zero");
                		break;
                	}
                },
                Err(e) => {
                    eprintln!("{}", e);
                    break;
                },
            }
        }

        weights
    }

}

impl crate::Playable for MatrixGame {

    /// Starts a REPL to play the game.
    /// 
    /// The user is asked to input a strategy, one probability at a time. 
    /// For robustness, inputs are read as weights: a renormalization is performed to obtain the mixed strategy.
    /// 
    /// # Remarks
    /// 
    /// Values are parsed using the [fasteval] crate, accepting a big range of inputs.
    /// 
    /// [fasteval]: https://crates.io/crates/fasteval
    fn play(&self) {
        println!("Welcome! You are playing the following matrix game:\n{}", self);

        loop {
        	let weights = self.input_strategy();

            // Reward
            let reward = weights.dot(&self.matrix).iter().cloned().fold(std::f64::NAN, f64::min) / weights.sum();
            println!("You obtained: {}\n", reward);

            // Repeating?
            println!("Keep playing?(y/n)");
            let mut repeat = String::new();
            std::io::stdin()
                .read_line(&mut repeat)
                .expect("Failed to read line");
            if !(repeat.trim() == "y") {
                println!("Thank you for playing!");
                break;
            }
        }
    }
}

impl<T: Into<f64>> From<Array2<T>> for MatrixGame {
    fn from(matrix: Array2<T>) -> Self {
        let converted_matrix = Array2::from_shape_vec(
            (matrix.shape()[0], matrix.shape()[1]),
            matrix
                .into_raw_vec()
                .into_iter()
                .map(|x: T| -> f64 { x.into() })
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        MatrixGame {
            matrix: converted_matrix,
        }
    }
}

impl Into<Array2<f64>> for MatrixGame {
    fn into(self) -> Array2<f64> {
        self.matrix
    }
}

impl fmt::Display for MatrixGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;
    use ndarray::array;
    use test_case::test_case;

    #[test]
    fn construction() {
        let matrix = array![[0, 1], [1, 0],];
        MatrixGame::from(matrix);
    }

    #[test_case( array![[0, 1], [1, 0]],  2, 2 ; "2x2")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1]],  2, 3 ; "2x3")]
    fn checking_dimensions<T>(matrix: Array2<T>, row_actions: usize, column_actions: usize)
    where
        T: Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        assert_eq!(row_actions, matrix_game.row_actions());
        assert_eq!(column_actions, matrix_game.column_actions());
    }

    #[test_case( array![[0, 1], [1, 0]],  0.5 ; "positive value")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  0.0 ; "rock-paper-scisors")]
    fn computing_value<T>(matrix: Array2<T>, expected_value: f64)
    where
        T: Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        let value = matrix_game.value();
        assert_ulps_eq!(value, expected_value, max_ulps = 1);
    }

    #[test_case( array![[0, 1], [1, 0]],  vec![0.5, 0.5] ; "positive value")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  vec![1./3., 1./3., 1./3.] ; "rock-paper-scisors")]
    fn row_solving<T>(matrix: Array2<T>, expected_strategy: Vec<f64>)
    where
        T: Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        let (optimal_row_strategy, _) = matrix_game.row_solve();
        for i in 0..expected_strategy.len() {
            assert_ulps_eq!(optimal_row_strategy[i], expected_strategy[i], max_ulps = 1);
        }
    }

    #[test_case( array![[0, 1], [1, 0]],  (vec![0.5, 0.5], vec![0.5, 0.5], 0.5) ; "positive value")]
    #[test_case( array![[0, 1], [1, 0], [-1, -1]],  (vec![0.5, 0.5, 0.], vec![0.5, 0.5], 0.5) ; "positive value with extra strategy")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  (vec![1./3., 1./3., 1./3.], vec![1./3., 1./3., 1./3.], 0.0) ; "rock-paper-scisors")]
    fn solving<T>(matrix: Array2<T>, expected_solution: (Vec<f64>, Vec<f64>, f64))
    where
        T: Into<f64>,
    {
        let matrix_game = MatrixGame::from(matrix);
        let (optimal_row_strategy, optimal_column_strategy, value) = matrix_game.solve();
        for i in 0..expected_solution.0.len() {
            assert_ulps_eq!(
                optimal_row_strategy[i],
                expected_solution.0[i],
                max_ulps = 1
            );
        }
        for j in 0..expected_solution.1.len() {
            assert_ulps_eq!(
                optimal_column_strategy[j],
                expected_solution.1[j],
                max_ulps = 1
            );
        }
        assert_ulps_eq!(value, expected_solution.2, max_ulps = 1);
    }
}

pub use polynomial::PolyMatrixGame;

mod polynomial;

// use crate::traits::Game;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashSet;
use std::fmt;

/// [Matrix games](https://en.wikipedia.org/wiki/Zero-sum_game) are finite zero-sum two-player games.
///
/// # Examples
///
/// Rock-paper-scisors.
/// ```
/// # use ndarray::array;
/// # use neumann::MatrixGame;
/// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
/// MatrixGame::from(rewards);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixGame {
    matrix: Array2<f64>,
}

impl MatrixGame {
    /// Return whether the array has any elements
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }

    /// Returns `true` if both players have the same number of possible actions.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is a square game.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert!(matrix_game.is_square());
    /// ```
    ///
    /// A 2x3 game is not square.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [0, -1, 2]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert!(!matrix_game.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        let shape = self.matrix.shape();
        shape[0] == shape[1]
    }

    /// Returns `true` if both players have the same number of possible actions
    /// and a unique optimal strategy which has full support[^1].
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is a completely-mixed game.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert!(matrix_game.is_completely_mixed());
    /// ```
    ///
    /// A game with dominant strategies is not completely-mixed.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 2, 1], [2, 0, 1], [-1, -1, -1]];
    /// let matrix_game = MatrixGame::from(rewards);
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
        } else {
            let full_value = self.value();
            let sub_value = self.reduce_row(self.reduce_row_best()).value();
            full_value > sub_value
        }
    }

    /// Reduces the matrix game to a square sub-game with the same value and
    /// whose optimal strategies are also optimal in the original game.
    ///
    /// If the matrix game has dimensions `m`x`n`,
    /// then the resulting matrix game has dimensions `min(m, n)`x`min(m, n)`.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors can not be reduce further, so it stays the same.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let mut matrix_game = MatrixGame::from(rewards);
    /// matrix_game.reduce_to_square();
    /// assert_eq!(matrix_game.matrix(), &array![[0., 1., -1.], [1., -1., 0.], [-1., 0., 1.]]);
    /// ```
    ///
    /// A game with a rectangular shape can be reduced.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 2], [2, 0], [-1, -1]];
    /// let mut matrix_game = MatrixGame::from(rewards);
    /// matrix_game.reduce_to_square();
    /// assert_eq!(matrix_game.matrix(), &array![[0., 2.], [2., 0.]]);
    /// ```
    pub fn reduce_to_square(&mut self) {
        while self.actions_row() > self.actions_column() {
            *self = self.reduce_row(self.reduce_row_best());
        }
        while self.actions_row() < self.actions_column() {
            *self = self.reduce_column(self.reduce_column_best());
        }
    }

    /// Returns the indices, together with the corresponding sub-matrix game,
    /// of a square sub-matrix which is *completely-mixed*[^1]
    /// whose value is the same as the original game.
    ///
    /// The first vector of indices corresponds to actions of the row player.
    /// The second vector of indices corresponds to actions of the column player.
    ///
    /// See [is_completely_mixed] method for an explanation of completely-mixed matrix games.
    ///
    /// # Examples
    ///
    /// Rock-paper-scisors is already completely-mixed so its kernel is the whole game.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let matrix_game = MatrixGame::from(rewards.clone());
    /// let (kernel_rows, kernel_columns, kernel_matrix_game) = matrix_game.kernel_completely_mixed();
    /// assert_eq!(kernel_rows, vec![0, 1, 2]);
    /// assert_eq!(kernel_columns, vec![0, 1, 2]);
    /// assert_eq!(kernel_matrix_game, MatrixGame::from(rewards));
    /// ```
    ///
    /// A game with a rectangular shape.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 2], [2, 0], [-1, -1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// let (kernel_rows, kernel_columns, kernel_matrix_game) = matrix_game.kernel_completely_mixed();
    /// assert_eq!(kernel_rows, vec![0, 1]);
    /// assert_eq!(kernel_columns, vec![0, 1]);
    /// assert_eq!(kernel_matrix_game, MatrixGame::from(array![[0, 2], [2, 0]]));
    /// ```
    ///
    /// [is_completely_mixed]: struct.MatrixGame.html#method.is_completely_mixed
    /// [^1]: Kaplansky, I. (1945). 
    ///       [*A Contribution to Von Neumann's Theory of Games*](https://www.jstor.org/stable/1969164). 
    ///       Annals of Mathematics, 46(3), second series, 474-479. 
    ///       doi:10.2307/1969164
    pub fn kernel_completely_mixed(&self) -> (Vec<usize>, Vec<usize>, MatrixGame) {
        // Setting up kernel
        let mut kernel_rows: HashSet<usize> = (0..self.actions_row()).collect();
        let mut kernel_columns: HashSet<usize> = (0..self.actions_column()).collect();
        let mut kernel_matrix_game = self.clone();
        // Iterative reduction
        while !kernel_matrix_game.is_completely_mixed() {
            if kernel_matrix_game.actions_row() > kernel_matrix_game.actions_column() {
                let dropped_row = kernel_matrix_game.reduce_row_best();
                kernel_rows.remove(&dropped_row);
                kernel_matrix_game = kernel_matrix_game.reduce_row(dropped_row);
            } else if kernel_matrix_game.actions_row() < kernel_matrix_game.actions_column() {
                let dropped_column = kernel_matrix_game.reduce_column_best();
                kernel_columns.remove(&dropped_column);
                kernel_matrix_game = kernel_matrix_game.reduce_column(dropped_column);
            }
        }
        // Sorting indices
        let mut kernel_rows: Vec<usize> = kernel_rows.drain().collect();
        kernel_rows.sort();
        let mut kernel_columns: Vec<usize> = kernel_columns.drain().collect();
        kernel_columns.sort();

        (kernel_rows, kernel_columns, kernel_matrix_game)
    }

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
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1], [1, 0], [-1, -1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert_eq!(matrix_game.reduce_row_best(), 2);
    /// ```
    pub fn reduce_row_best(&self) -> usize {
        assert!(!self.is_empty());
        (0..self.actions_row())
            .map(|i| (i, self.reduce_row(i).value()))
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
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[1, 0, -1], [1, -1, 0]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// assert_eq!(matrix_game.reduce_column_best(), 0);
    /// ```
    pub fn reduce_column_best(&self) -> usize {
        assert!(!self.is_empty());
        (0..self.actions_row())
            .map(|i| (i, self.reduce_column(i).value()))
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

    /// Returns a matrix game with one action less for the row player.
    ///
    /// # Examples
    ///
    /// Forgetting about the first action for the row player.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.reduce_row(0);
    /// assert_eq!(sub_matrix_game.matrix(), &array![[1., -1., 0.], [-1., 0., 1.]]);
    /// ```
    ///
    /// Forgetting about the last action for the row player.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.reduce_row(2);
    /// assert_eq!(sub_matrix_game.matrix(), &array![[1., 2., 3.], [4., 5., 6.]]);
    /// ```
    pub fn reduce_row(&self, row: usize) -> MatrixGame {
        let mut sub_rewards: Vec<f64> = Vec::new();
        for i in 0..self.actions_row() {
            if !(i == row) {
                sub_rewards.extend(self.matrix.index_axis(Axis(0), i));
            }
        }
        let sub_reward_matrix =
            Array2::from_shape_vec((self.actions_row() - 1, self.actions_column()), sub_rewards)
                .unwrap();

        MatrixGame::from(sub_reward_matrix)
    }

    /// Returns a matrix game with one action less for the column player.
    ///
    /// # Examples
    ///
    /// Forgetting about the last action for the column player.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[0, 1, -1], [1, -1, 0], [-1, 0, 1]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.reduce_column(2);
    /// assert_eq!(sub_matrix_game.matrix(), &array![[0., 1.], [1., -1.], [-1., 0.]]);
    /// ```
    ///
    /// Forgetting about the first action for the column player.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::MatrixGame;
    /// let rewards = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    /// let matrix_game = MatrixGame::from(rewards);
    /// let sub_matrix_game = matrix_game.reduce_column(2);
    /// assert_eq!(sub_matrix_game.matrix(), &array![[1., 2.], [4., 5.], [7., 8.]]);
    /// ```
    pub fn reduce_column(&self, column: usize) -> MatrixGame {
        let transpose_matrix_game = MatrixGame::from(self.matrix.t().to_owned());
        let transpose_sub_matrix_game = transpose_matrix_game.reduce_row(column);
        MatrixGame::from(transpose_sub_matrix_game.matrix.t().to_owned())
    }

    /// Returns the reward matrix for the row player.
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Returns an optimal strategy for the row player and the value of the game, i.e. the value this player can ensure.
    pub fn solve_row(&self) -> (Vec<f64>, f64) {
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
        self.solve_row().1
    }

    /// Returns a Nash equilibrium and the value of the game.
    pub fn solve(&self) -> (Vec<f64>, Vec<f64>, f64) {
        let column_matrix: Array2<f64> = self.matrix().t().map(|x| -x).to_owned();
        let column_matrix_game = MatrixGame::from(column_matrix);

        // Solve
        let (optimal_row_strategy, value) = self.solve_row();
        let (optimal_column_strategy, _) = column_matrix_game.solve_row();

        (optimal_row_strategy, optimal_column_strategy, value)
    }

    /// Shape of the matrix game.
    ///
    /// First the number of row actions, then the number of column actions.
    pub fn shape(&self) -> [usize; 2] {
        [self.matrix.shape()[0], self.matrix.shape()[1]]
    }

    /// Number of row actions
    pub fn actions_row(&self) -> usize {
        self.matrix.shape()[0]
    }

    /// Number of column actions
    pub fn actions_column(&self) -> usize {
        self.matrix.shape()[1]
    }

    fn input_strategy(&self) -> Array1<f64> {
        println!("Enter you mixed strategy, one probability at a time.");

        let mut weights = Array1::from_elem(self.actions_row(), 0.);
        for i in 0..self.actions_row() {
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
                }
                Err(e) => {
                    eprintln!("{}", e);
                    break;
                }
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
        println!(
            "Welcome! You are playing the following matrix game:\n{}",
            self
        );

        loop {
            let weights = self.input_strategy();

            // Reward
            let reward = weights
                .dot(&self.matrix)
                .iter()
                .cloned()
                .fold(std::f64::NAN, f64::min)
                / weights.sum();
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

impl<T> From<Array2<T>> for MatrixGame
where
    T: Into<f64> + Clone,
{
    fn from(matrix: Array2<T>) -> Self {
        MatrixGame {
            matrix: matrix.mapv(|x: T| -> f64 { x.into() }),
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
    fn checking_dimensions<T>(matrix: Array2<T>, actions_row: usize, actions_column: usize)
    where
        T: Into<f64> + Clone,
    {
        let matrix_game = MatrixGame::from(matrix);
        assert_eq!(actions_row, matrix_game.actions_row());
        assert_eq!(actions_column, matrix_game.actions_column());
    }

    #[test_case( array![[0, 1], [1, 0]],  0.5 ; "positive value")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  0.0 ; "rock-paper-scisors")]
    fn computing_value<T>(matrix: Array2<T>, expected_value: f64)
    where
        T: Into<f64> + Clone,
    {
        let matrix_game = MatrixGame::from(matrix);
        let value = matrix_game.value();
        assert_ulps_eq!(value, expected_value, max_ulps = 1);
    }

    #[test_case( array![[0, 1], [1, 0]],  vec![0.5, 0.5] ; "positive value")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  vec![1./3., 1./3., 1./3.] ; "rock-paper-scisors")]
    fn row_solving<T>(matrix: Array2<T>, expected_strategy: Vec<f64>)
    where
        T: Into<f64> + Clone,
    {
        let matrix_game = MatrixGame::from(matrix);
        let (optimal_row_strategy, _) = matrix_game.solve_row();
        for i in 0..expected_strategy.len() {
            assert_ulps_eq!(optimal_row_strategy[i], expected_strategy[i], max_ulps = 1);
        }
    }

    #[test_case( array![[0, 1], [1, 0]],  (vec![0.5, 0.5], vec![0.5, 0.5], 0.5) ; "positive value")]
    #[test_case( array![[0, 1], [1, 0], [-1, -1]],  (vec![0.5, 0.5, 0.], vec![0.5, 0.5], 0.5) ; "positive value with extra strategy")]
    #[test_case( array![[0, 1, -1], [-1, 0, 1], [1, -1, 0]],  (vec![1./3., 1./3., 1./3.], vec![1./3., 1./3., 1./3.], 0.0) ; "rock-paper-scisors")]
    fn solving<T>(matrix: Array2<T>, expected_solution: (Vec<f64>, Vec<f64>, f64))
    where
        T: Into<f64> + Clone,
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

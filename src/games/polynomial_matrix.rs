/// Helper functions.
mod helper;

use helper::{cofactor, determinant};

use crate::{Certified, MatrixGame};
use itertools::Itertools;
use ndarray::{Array2, Axis};
use num_rational::Ratio;
use polynomials::{poly, Polynomial};
use std::fmt;

/// Polynomial matrix games are [Matrix Games](https://en.wikipedia.org/wiki/Zero-sum_game) whith perturbation terms in terms of polynomials.
///
/// # Examples
///
/// Error term pushes the optimal strategy in a different direction than the error free optimal strategy.
/// ```
/// # use ndarray::array;
/// # use neumann::PolyMatrixGame;
/// let poly_matrix = vec![array![[0, 1], [1, 0]], array![[0, 2], [1, 0]]];
/// PolyMatrixGame::from(poly_matrix);
/// ```
///
/// Error term diminishes the reward for the row player, but does not changes optimal strategies.
/// ```
/// # use ndarray::array;
/// # use neumann::PolyMatrixGame;
/// let poly_matrix = vec![array![[0, 1], [1, 0]], array![[0, -1], [-1, 0]]];
/// PolyMatrixGame::from(poly_matrix);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomailMatrixGame {
    poly_matrix: Vec<Array2<i32>>,
}

impl PolyMatrixGame {
    /// Returns the matrix game that corresponds to evaluate perturbation at `epsilon`.
    pub fn eval(&self, epsilon: f64) -> MatrixGame {
        let mut matrix: Array2<f64> = self.poly_matrix[0].map(|x| -> f64 { f64::from(*x) });
        for i in 1..self.poly_matrix.len() {
            matrix = matrix
                + &self.poly_matrix[i].map(|x| -> f64 { f64::from(*x) }) * epsilon.powi(i as i32);
        }
        MatrixGame::from(matrix)
    }

    pub fn degree(&self) -> usize {
        self.poly_matrix.len() - 1
    }

    fn linear_is_uniform_value_positive(&self) -> bool {
        // Setting
        let dimensions = self.poly_matrix[0].shape();

        // Define LP
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);

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

        // Value constrains: optimal solution of the first matrix
        let error_free_value = self.eval(0.0).value();
        for column_action in 0..dimensions[1] {
            let rewards = self.poly_matrix[0]
                .index_axis(Axis(1), column_action)
                .map(|x| *x as f64);
            let constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, error_free_value);
        }

        // Value constrains: positive in the second matrix
        for column_action in 0..dimensions[1] {
            let rewards = self.poly_matrix[1]
                .index_axis(Axis(1), column_action)
                .map(|x| *x as f64);
            let mut constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            constrain.push((value_variable, -1.0));
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, 0.0);
        }

        // Solve
        match problem.solve() {
            Ok(solution) => solution.objective() >= 0.0,
            Err(_) => false,
        }
    }

    /// Checks exponentially-many LPs to decide uniform value-positivity.
    fn poly_is_uniform_value_positive(&self) -> bool {
        // Setting
        let dimensions = self.poly_matrix[0].shape();

        // Add extra matrix of ones so that we can decide upon strict uniform value-positivity
        let mut poly_matrix = self.poly_matrix.clone();
        poly_matrix.push(Array2::from_elem(self.poly_matrix[0].raw_dim(), 1));
        let augmented_matrix_game = PolyMatrixGame::from(poly_matrix);

        // Define LP basis
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);
        // Add row player strategy
        let mut row_strategy = Vec::with_capacity(dimensions[0]);
        for _ in 0..dimensions[0] {
            row_strategy.push(problem.add_var(0.0, (0.0, 1.0)));
        }
        // Add value variable
        let value_variable = problem.add_var(1.0, (std::f64::NEG_INFINITY, std::f64::INFINITY));

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

        // Value constrains: optimal solution of the first matrix
        let error_free_value = augmented_matrix_game.eval(0.0).value();
        for column_action in 0..dimensions[1] {
            let rewards = augmented_matrix_game.poly_matrix[0]
                .index_axis(Axis(1), column_action)
                .map(|x| *x as f64);
            let constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, error_free_value);
        }

        // Iterate over exponentially-many LPs
        let index_vectors = (0..dimensions[1])
            .map(|_| 1..=augmented_matrix_game.degree())
            .multi_cartesian_product();

        for index_vector in index_vectors {
            // Define the LP
            let mut index_problem = problem.clone();

            // Value constrains: positive before the index_vactor
            for column_action in 0..dimensions[1] {
                for matrix_index in 1..index_vector[column_action] {
                    let rewards = augmented_matrix_game.poly_matrix[matrix_index]
                        .index_axis(Axis(1), column_action)
                        .map(|x| *x as f64);
                    let constrain = row_strategy
                        .clone()
                        .into_iter()
                        .zip(rewards.into_iter().cloned())
                        .collect::<Vec<(minilp::Variable, f64)>>();
                    index_problem.add_constraint(constrain, minilp::ComparisonOp::Eq, 0.0);
                }
            }

            // Value constrains: maximizing the last index
            for column_action in 0..dimensions[1] {
                let matrix_index = index_vector[column_action];
                let rewards = augmented_matrix_game.poly_matrix[matrix_index]
                    .index_axis(Axis(1), column_action)
                    .map(|x| *x as f64);
                let mut constrain = row_strategy
                    .clone()
                    .into_iter()
                    .zip(rewards.into_iter().cloned())
                    .collect::<Vec<(minilp::Variable, f64)>>();
                constrain.push((value_variable, -1.0));
                index_problem.add_constraint(constrain, minilp::ComparisonOp::Ge, 0.0);
            }

            if let Ok(solution) = index_problem.solve() {
                if solution.objective() > std::f64::EPSILON {
                    return true;
                };
            }
        }

        false
    }

    /// Shape of the matrix game.
    ///
    /// First the number of row actions, then the number of column actions.
    ///
    /// # Examples
    ///
    /// Two-actions polynomial matrix games has shape `[2, 2]`.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::PolyMatrixGame;
    /// # use polynomials::poly;
    /// let poly_matrix = vec![array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]], array![[2, 1], [4, 1]]];
    /// let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
    /// assert_eq!(poly_matrix_game.shape(), [2, 2]);
    /// ```
    pub fn shape(&self) -> [usize; 2] {
        [
            self.poly_matrix[0].shape()[0],
            self.poly_matrix[0].shape()[1],
        ]
    }

    /// Number of row actions
    pub fn actions_row(&self) -> usize {
        self.shape()[0]
    }

    /// Number of column actions
    pub fn actions_column(&self) -> usize {
        self.shape()[1]
    }
}

impl crate::value_positivity::ValuePositivity<Vec<f64>, Vec<f64>, ()> for PolyMatrixGame {
    /// Returns a value for the error term `epsilon` such that
    /// kernels of optimal strategies are guaranteed not to change between this value and zero.
    ///
    /// In particular, the value function is a rational function between this value and zero.
    ///
    /// # Examples
    ///
    /// Error is noticed by the row player only if epsilon is greater than one.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::{PolyMatrixGame, value_positivity::ValuePositivity};
    /// let poly_matrix = vec![array![[1, 1], [0, 0]], array![[0, 0], [1, 1]]];
    /// let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
    /// assert!(poly_matrix_game.epsilon_kernel_constant() <= 1.);
    /// ```
    fn epsilon_kernel_constant(&self) -> f64 {
        let m: f64 = *self.poly_matrix[0].shape().iter().max().unwrap() as f64;
        let b: f64 = self
            .poly_matrix
            .iter()
            .map(|matrix| matrix.iter().map(|x| x.abs()).max().unwrap())
            .max()
            .unwrap() as f64;
        let k: f64 = (self.poly_matrix.len() - 1) as f64;
        f64::min(
            1.0,
            (2. * m * k).powf(-m * k)
                * (b * m).powf(m * (1. - 2. * m * k))
                * (m * k + 1.).powf(1. - 2. * m * k),
        )
    }

    /// Checks if the polynomail matrix game has at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a positive threshold for which the value of the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    ///
    /// # Output
    ///
    /// Boolean.
    ///
    /// # Certificate
    ///
    /// If the answer is true, the certificate is a strategy for the row player that ensures a non-negative reward
    /// for the perturbation given by `epsilon_kernel_constant`.
    ///
    /// If the answer is false, the certificate is a strategy for the colmun player that ensures a negative reward
    /// for the column player, for the perturbation given by `epsilon_kernel_constant`.
    ///
    /// Recall that the value function does not change sign from between this perturbation and zero.
    fn is_value_positive(&self) -> Certified<bool, Vec<f64>> {
        let (row_strategy, column_strategy, value) =
            self.eval(self.epsilon_kernel_constant()).solve();
        match value >= self.eval(0.).value() {
            true => Certified::from((true, row_strategy)),
            false => Certified::from((false, column_strategy)),
        }
    }

    /// Checks the rewards given by the strategy.
    fn is_value_positive_checker(&self, certified_output: Certified<bool, Vec<f64>>) -> bool {
        match certified_output.output {
            true => {
                let rewards = ndarray::Array1::from(certified_output.certificate)
                    .dot(self.eval(self.epsilon_kernel_constant()).matrix());
                rewards.iter().all(|&v| v >= 0.0)
            }
            false => {
                let rewards = self
                    .eval(self.epsilon_kernel_constant())
                    .matrix()
                    .dot(&ndarray::Array1::from(certified_output.certificate));
                rewards.iter().all(|&v| v < 0.0)
            }
        }
    }

    /// Checks if the polynomail matrix game has at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a positive threshold for which the value of the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    fn is_value_positive_uncertified(&self) -> bool {
        self.eval(self.epsilon_kernel_constant()).value() >= self.eval(0.).value()
    }

    /// Checks if the polynomail matrix game has a fixed strategy that ensures at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a fixed strategy and a positive threshold for which the reward given by this strategy
    /// in the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    fn is_uniform_value_positive(&self) -> Certified<bool, Vec<f64>> {
        todo!()

        // if self.is_value_positive() {
        //     if self.degree() == 1 {
        //         self.linear_is_uniform_value_positive()
        //     } else {
        //         self.poly_is_uniform_value_positive()
        //     }
        // } else {
        //     false
        // }
    }

    ///
    fn is_uniform_value_positive_checker(&self, _: Certified<bool, Vec<f64>>) -> bool {
        todo!()
    }

    /// Returns the value function close to zero.
    ///
    /// The value function close to zero is a rational function. At zero and far from zero,
    /// the value function might correspond to another rational function. In general,
    /// the value function of a polynomial matrix game is a piecewise rational function.
    /// See [epsilon_kernel_constant] to have a bound on the interval in which this rational function
    /// is indeed the value function.
    ///
    /// # Remarks
    ///
    /// The `Ratio` returned is not simplified, i.e. there might be a polynomial factor in common
    /// between the numerator and denominator.
    ///
    /// # Examples
    ///
    /// Two-actions linear matrix games can lead to quadratic numerator in the value function.
    /// ```
    /// # use ndarray::array;
    /// # use neumann::{PolyMatrixGame, value_positivity::ValuePositivity};
    /// # use polynomials::poly;
    /// let poly_matrix = vec![array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]]];
    /// let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
    /// let value_function = poly_matrix_game.functional_form_value();
    /// assert_eq!(value_function.numer().degree(), 2);
    /// assert_eq!(value_function.numer(), &poly![0, 0, 2]);
    /// assert_eq!(value_function.denom(), &poly![4, 6]);
    /// ```
    ///
    /// [epsilon_kernel_constant]: struct.PolyMatrixGame.html#method.epsilon_kernel_constant
    fn functional_form(&self) -> Certified<num_rational::Ratio<polynomials::Polynomial<i32>>, ()> {
        todo!()

        // let matrix_game = self.eval(self.epsilon_kernel_constant());
        // let (kernel_rows, kernel_columns, _) = matrix_game.kernel_completely_mixed();
        // let poly_array: Array2<Polynomial<i32>> = self.clone().into();
        // let mut kernel_poly_array =
        //     Array2::from_elem((kernel_rows.len(), kernel_columns.len()), poly![0]);
        // for i in 0..kernel_rows.len() {
        //     for j in 0..kernel_columns.len() {
        //         kernel_poly_array[[i, j]] = poly_array[[kernel_rows[i], kernel_columns[j]]].clone();
        //     }
        // }
        // let numer: Polynomial<i32> = determinant(&kernel_poly_array);
        // let denom: Polynomial<i32> = (0..kernel_rows.len())
        //     .cartesian_product(0..kernel_columns.len())
        //     .map(|(i, j)| cofactor(&kernel_poly_array, i, j))
        //     .fold(poly![0], |acc, p| acc + p);

        // Ratio::new_raw(numer, denom)
    }

    ///
    fn functional_form_checker(
        &self,
        _: Certified<num_rational::Ratio<polynomials::Polynomial<i32>>, ()>,
    ) -> bool {
        todo!()
    }
}

impl<T> From<Vec<Array2<T>>> for PolyMatrixGame
where
    T: Into<i32> + Clone,
{
    /// # Panics
    ///
    /// If an empty vector is given or the shape of the matrices do not coincide.
    fn from(poly_matrix: Vec<Array2<T>>) -> Self {
        assert!(!poly_matrix.is_empty());
        assert!(poly_matrix.windows(2).all(|w| w[0].shape() == w[1].shape()));

        let mut converted_poly_matrix = Vec::new();
        for matrix in poly_matrix {
            let converted_matrix = matrix.mapv(|x: T| -> i32 { x.into() });
            converted_poly_matrix.push(converted_matrix);
        }

        PolyMatrixGame {
            poly_matrix: converted_poly_matrix,
        }
    }
}

impl fmt::Display for PolyMatrixGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();
        for i in 0..self.poly_matrix.len() {
            string += &format!("{}", self.poly_matrix[i]);
            if i == 1 {
                string += "eps";
            }
            if i > 1 {
                string += &format!(" eps^{}", i);
            }
            if i < self.poly_matrix.len() - 1 {
                string += "\n+\n"
            }
        }
        write!(f, "{}", string)
    }
}

impl Into<Vec<Array2<i32>>> for PolyMatrixGame {
    fn into(self) -> Vec<Array2<i32>> {
        self.poly_matrix
    }
}

impl Into<Array2<Polynomial<i32>>> for PolyMatrixGame {
    /// Performs the conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ndarray::{Array2, array};
    /// # use polynomials::{Polynomial, poly};
    /// # use neumann::PolyMatrixGame;
    /// let poly_matrix = vec![array![[0, 1], [1, 0]], array![[0, 2], [3, 0]]];
    /// let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
    /// let poly_array: Array2<Polynomial<i32>> = poly_matrix_game.into();
    /// assert_eq!(poly_array[[0, 0]], poly![0]);
    /// assert_eq!(poly_array[[0, 1]], poly![1, 2]);
    /// assert_eq!(poly_array[[1, 0]], poly![1, 3]);
    /// assert_eq!(poly_array[[1, 1]], poly![0]);
    /// ```
    fn into(self) -> Array2<Polynomial<i32>> {
        let mut poly_array = Array2::from_elem(self.shape(), poly![0]);
        // Change the error-free term
        for i in 0..self.actions_row() {
            for j in 0..self.actions_column() {
                poly_array[[i, j]] = poly![self.poly_matrix[0][[i, j]]]
            }
        }
        // Add the error terms
        for k in 1..=self.degree() {
            for i in 0..self.actions_row() {
                for j in 0..self.actions_column() {
                    poly_array[[i, j]].push(self.poly_matrix[k][[i, j]]);
                }
            }
        }

        poly_array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_positivity::ValuePositivity;
    use ndarray::array;
    use test_case::test_case;
    // use approx::assert_ulps_eq;

    #[test_case( vec![array![[0, 1], [1, 0]], array![[0, 1], [1, 0]]] ; "valid construction")]
    #[test_case( vec![array![[0, 1], [1, 0]], array![[0, 1]]] => panics "" ; "invalid construction: different shapes")]
    #[test_case( vec![] => panics "" ; "invalid construction: empty vector")]
    fn construction(poly_matrix: Vec<Array2<i32>>) {
        PolyMatrixGame::from(poly_matrix);
    }

    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, 1], [1, 0]] ],  true ; "value-positive easy")]
    #[test_case( vec![ array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]] ],  true ; "value-positive medium")]
    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, -1], [-1, 0]] ],  false ; "not value-positive")]
    fn computing_value_positivity(poly_matrix: Vec<Array2<i32>>, expected_value: bool) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        assert_eq!(poly_matrix_game.is_value_positive().output, expected_value);
        assert_eq!(
            poly_matrix_game.is_value_positive_checker(poly_matrix_game.is_value_positive()),
            expected_value
        );
    }

    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, 1], [1, 0]] ],  true ; "uniform value-positive easy")]
    #[test_case( vec![ array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]] ],  false ; "not uniform value-positive medium")]
    #[test_case( vec![ array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]] ],  false ; "not uniform value-positive hard")]
    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, -1], [-1, 0]] ],  false ; "not uniform value-positive easy")]
    #[test_case( vec![ array![[0]], array![[0]], array![[0]] ], true ; "uniform value-positive easy polynomial")]
    #[test_case( vec![ array![[1, -1], [-1, 1]], array![[2, -2], [-2, 2]], array![[3, -3], [-3, 3]] ], true ; "uniform value-positive medium polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1]], array![[2, -1], [-1, 2]], array![[2, -1], [-1, 2]] ], true ; "uniform value-positive medium-hard polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1], [1, 1]], array![[0, 0], [0, 0], [0, 1]], array![[1, -1], [1, -1], [1, -1]] ], true ; "uniform value-positive hard polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1]], array![[1, -1], [-1, 1]], array![[-2, -1], [-1, -2]] ], false ; "not uniform value-positive medium polynomial")]
    fn computing_uniform_value_positivity(poly_matrix: Vec<Array2<i32>>, expected_value: bool) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        assert_eq!(
            poly_matrix_game.is_uniform_value_positive().output,
            expected_value
        );
        assert_eq!(
            poly_matrix_game
                .is_uniform_value_positive_checker(poly_matrix_game.is_uniform_value_positive()),
            expected_value
        );
    }

    #[test_case( vec![ array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]] ], Ratio::new_raw(poly![0, 0, 2], poly![4, 6]) ; "quadratic")]
    #[test_case( vec![ array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]] ], Ratio::new_raw(poly![0, 1], poly![2, 3]) ; "linear")]
    fn computing_functional_form_value(
        poly_matrix: Vec<Array2<i32>>,
        expected_rational: Ratio<Polynomial<i32>>,
    ) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        let functional_form_value = poly_matrix_game.functional_form().output;
        println!("{}", poly_matrix_game);
        println!("{:?}", functional_form_value);
        assert_eq!(functional_form_value.numer(), expected_rational.numer());
        assert_eq!(functional_form_value.denom(), expected_rational.denom());
        assert!(poly_matrix_game.functional_form_checker(poly_matrix_game.functional_form()));
    }
}

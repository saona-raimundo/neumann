use crate::MatrixGame;
use ndarray::{Axis, Array2};
use std::fmt;

/// Polynomial matrix games are [Matrix Games](https://en.wikipedia.org/wiki/Zero-sum_game) whith  
/// perturbation terms in terms of polynomials.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyMatrixGame {
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

    /// Checks if the polynomail matrix game has at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a positive threshold for which the value of the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    pub fn is_value_positive(&self) -> bool {
        // Computing epsilon_0
        let m: f64 = *self.poly_matrix[0].shape().iter().max().unwrap() as f64;
        let b: f64 = self
            .poly_matrix
            .iter()
            .map(|matrix| matrix.iter().map(|x| x.abs()).max().unwrap())
            .max()
            .unwrap() as f64;
        let k: f64 = (self.poly_matrix.len() - 1) as f64;
        let epsilon_0 = (2. * m * k).powf(-m * k)
            * (b * m).powf(m * (1. - 2. * m * k))
            * (m * k + 1.).powf(1. - 2. * m * k);

        self.eval(epsilon_0).value() >= self.eval(0.).value()
    }

    /// Checks if the polynomail matrix game has a fixed strategy that ensures at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a fixed strategy and a positive threshold for which the reward given by this strategy
    /// in the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    pub fn is_uniform_value_positive(&self) -> bool {
        // Computing epsilon_0
        let m: f64 = *self.poly_matrix[0].shape().iter().max().unwrap() as f64;
        let b: f64 = self
            .poly_matrix
            .iter()
            .map(|matrix| matrix.iter().map(|x| x.abs()).max().unwrap())
            .max()
            .unwrap() as f64;
        let k: f64 = (self.poly_matrix.len() - 1) as f64;
        let epsilon_1 = 1. / (2. * m * k * b);
        let lower_bound = 1. / (b.powf(m) * m.powf(m / 2.));

        let mut null_actions = Vec::new(); // I_0
        let mut positive_actions = Vec::new(); // I_1
        let error_free_value = self.eval(0.0).value();
        for i in 0..(self.poly_matrix[0].shape()[0]) { // All actions of the row player
        	let mut extended_positive_actions = positive_actions.clone();
        	extended_positive_actions.extend(0..(self.poly_matrix[0].shape()[0]));
        	let problem = self.restricted_lp(epsilon_1, lower_bound, &null_actions, &extended_positive_actions);
        	let solution = problem.solve().unwrap();
        	if solution.objective() >= error_free_value {
        		println!("The polynomial matrix game:\n{}", self);
        		println!("is uniform value-positive by: {:?}", solution.iter().map(|(_, p)| *p).collect::<Vec<f64>>());
        		return true
        	}
        	null_actions.push(i);
        	let problem = self.restricted_lp(epsilon_1, lower_bound, &null_actions, &positive_actions);
        	if !(problem.solve().unwrap().objective() >= error_free_value) {
        		null_actions.pop();
        		positive_actions.push(i);
        	}
        }

        false
    }

    /// Returns an LP where possible actions are restricted.
    fn restricted_lp(&self, epsilon: f64, lower_bound: f64, null_actions: &Vec<usize>, positive_actions: &Vec<usize>) -> minilp::Problem {
        // Define LP
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);

        // Setting
        let matrix = self.eval(epsilon).matrix().clone();
        let dimensions = matrix.shape();

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
            let rewards = matrix.index_axis(Axis(1), column_action);
            let mut constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            constrain.push((value_variable, -1.0));
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, 0.0);
        }

        // Null restrictions
        for i in null_actions {
        	problem.add_constraint(&[(row_strategy[*i], 1.0)], minilp::ComparisonOp::Eq, 0.0);
        }

        // Positive restrictions
        for i in positive_actions {
        	problem.add_constraint(&[(row_strategy[*i], 1.0)], minilp::ComparisonOp::Ge, lower_bound);
        }

        problem
    }
}

impl<T> From<Vec<Array2<T>>> for PolyMatrixGame 
where
    i32: From<T>,
    T: Copy,
{
    /// # Panics
    ///
    /// If an empty vector is given.
    fn from(poly_matrix: Vec<Array2<T>>) -> Self {
        assert!(!poly_matrix.is_empty());

        let mut converted_poly_matrix = Vec::new();
        for matrix in poly_matrix {
            let converted_matrix = matrix.map(|x| i32::from(*x));
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
            if i < self.poly_matrix.len() - 1 {
                string += "\n+\n"
            }
            if i > 0 {
                string += &format!("eps^{}", i);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use test_case::test_case;
    // use approx::assert_ulps_eq;

    #[test]
    fn construction() {
        let poly_matrix = vec![array![[0, 1], [1, 0]], array![[0, 1], [1, 0]]];
        PolyMatrixGame::from(poly_matrix);
    }

    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, 1], [1, 0]] ],  true ; "value-positive easy")]
    #[test_case( vec![ array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]] ],  true ; "value-positive medium")]
    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, -1], [-1, 0]] ],  false ; "not value-positive")]
    fn computing_value_positivity(poly_matrix: Vec<Array2<i32>>, expected_value: bool) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        assert_eq!(poly_matrix_game.is_value_positive(), expected_value);
    }

    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, 1], [1, 0]] ],  true ; "uniform value-positive easy")]
    #[test_case( vec![ array![[0, 0], [-1, 1]], array![[2, -1], [0, 0]] ],  false ; "not uniform value-positive medium")]
    #[test_case( vec![ array![[1, -1], [-1, 1]], array![[1, -3], [0, 2]] ],  false ; "not uniform value-positive hard")]
    #[test_case( vec![ array![[0, 1], [1, 0]], array![[0, -1], [-1, 0]] ],  false ; "not uniform value-positive easy")]
    fn computing_uniform_value_positivity(poly_matrix: Vec<Array2<i32>>, expected_value: bool) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        assert_eq!(poly_matrix_game.is_uniform_value_positive(), expected_value);
    }
}
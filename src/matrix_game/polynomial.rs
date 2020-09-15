use crate::MatrixGame;
use ndarray::{Axis, Array2};
use std::fmt;
use itertools::Itertools;

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
        let epsilon_0 = f64::min(1.0, (2. * m * k).powf(-m * k)
            * (b * m).powf(m * (1. - 2. * m * k))
            * (m * k + 1.).powf(1. - 2. * m * k));

        self.eval(epsilon_0).value() >= self.eval(0.).value()
    }

    pub fn degree(&self) -> usize {
    	self.poly_matrix.len() - 1
    }

    /// Checks if the polynomail matrix game has a fixed strategy that ensures at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a fixed strategy and a positive threshold for which the reward given by this strategy
    /// in the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    pub fn is_uniform_value_positive(&self) -> bool {
    	if self.is_value_positive() {
        	if self.degree() == 1 {
	    		self.linear_is_uniform_value_positive()
	    	} else {
	    		self.poly_is_uniform_value_positive()
	    	}		
    	} else {
    		false
    	}

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
            let rewards = self.poly_matrix[0].index_axis(Axis(1), column_action).map(|x| *x as f64);
            let constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, error_free_value);
        }

        // Value constrains: positive in the second matrix
        for column_action in 0..dimensions[1] {
            let rewards = self.poly_matrix[1].index_axis(Axis(1), column_action).map(|x| *x as f64);
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
            let rewards = augmented_matrix_game.poly_matrix[0].index_axis(Axis(1), column_action).map(|x| *x as f64);
            let constrain = row_strategy
                .clone()
                .into_iter()
                .zip(rewards.into_iter().cloned())
                .collect::<Vec<(minilp::Variable, f64)>>();
            problem.add_constraint(constrain, minilp::ComparisonOp::Ge, error_free_value);
        }

        // Iterate over exponentially-many LPs 
        let index_vectors = (0..dimensions[1]).map(|_| 1..=augmented_matrix_game.degree())
        	.multi_cartesian_product();

        for index_vector in index_vectors {

        	println!("{:?}", index_vector);
        	// Define the LP
        	let mut index_problem = problem.clone();
        	
	        // Value constrains: positive before the index_vactor
	        for column_action in 0..dimensions[1] {
		        for matrix_index in 1..index_vector[column_action] {
		            let rewards = augmented_matrix_game.poly_matrix[matrix_index].index_axis(Axis(1), column_action).map(|x| *x as f64);
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
	            let rewards = augmented_matrix_game.poly_matrix[matrix_index].index_axis(Axis(1), column_action).map(|x| *x as f64);
	            let mut constrain = row_strategy
	                .clone()
	                .into_iter()
	                .zip(rewards.into_iter().cloned())
	                .collect::<Vec<(minilp::Variable, f64)>>();
	            constrain.push((value_variable, -1.0));
	            index_problem.add_constraint(constrain, minilp::ComparisonOp::Ge, 0.0);
	        }

        	if let Ok(solution) = index_problem.solve() {
        		println!("{:?}", solution);
        		println!("{}", solution.objective());
        		if solution.objective() > std::f64::EPSILON {
        			return true
        		};
        	}
        }

        false

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
            if i == 1 {
            	string += "eps";
            }
            if i > 1 {
                string += &format!("eps^{}", i);
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
    #[test_case( vec![ array![[0]], array![[0]], array![[0]] ], true ; "uniform value-positive easy polynomial")]
    #[test_case( vec![ array![[1, -1], [-1, 1]], array![[2, -2], [-2, 2]], array![[3, -3], [-3, 3]] ], true ; "uniform value-positive medium polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1]], array![[2, -1], [-1, 2]], array![[2, -1], [-1, 2]] ], true ; "uniform value-positive medium-hard polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1], [1, 1]], array![[0, 0], [0, 0], [0, 1]], array![[1, -1], [1, -1], [1, -1]] ], true ; "uniform value-positive hard polynomial")]
    #[test_case( vec![ array![[1, 1], [1, 1]], array![[1, -1], [-1, 1]], array![[-2, -1], [-1, -2]] ], false ; "not uniform value-positive medium polynomial")]
    fn computing_uniform_value_positivity(poly_matrix: Vec<Array2<i32>>, expected_value: bool) {
        let poly_matrix_game = PolyMatrixGame::from(poly_matrix);
        assert_eq!(poly_matrix_game.is_uniform_value_positive(), expected_value);
    }
}

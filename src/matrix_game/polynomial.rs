
use ndarray::{Array2};
use crate::MatrixGame;

/// Polynomial matrix games are [Matrix Games](https://en.wikipedia.org/wiki/Zero-sum_game) whith  
/// perturbation terms in terms of polynomials.
#[derive(Debug, Clone)]
pub struct PolyMatrixGame {
    poly_matrix: Vec<Array2<f64>>,
}

impl PolyMatrixGame {
	/// # Panics
	/// 
	/// If an empty vector is given.
    pub fn new<T>(poly_matrix: Vec<Array2<T>>) -> Self
	where
		T: Into<f64>,
    {
    	assert!(poly_matrix.len() > 0);

    	let mut converted_poly_matrix = Vec::new();
    	for matrix in poly_matrix {
            let converted_matrix = Array2::from_shape_vec(
	            (matrix.shape()[0], matrix.shape()[1]),
	        	matrix.into_raw_vec()
	                .into_iter()
	        		.map(|x: T| -> f64 {x.into()})
	        		.collect::<Vec<f64>>()
	        	).unwrap();
	        converted_poly_matrix.push(converted_matrix);		
    	}

	    PolyMatrixGame { poly_matrix: converted_poly_matrix }
    }

    pub fn eval(&self, epsilon: f64) -> MatrixGame {
    	let mut matrix: Array2<f64> = self.poly_matrix[0].clone();
    	for i in 1..self.poly_matrix.len() {
    		matrix = matrix + &self.poly_matrix[i] * epsilon.powi(i as i32);
    	}
    	MatrixGame::new(matrix)
    }

    /// Checks if the polynomail matrix game has at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a positive threshold for which the value of the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    pub fn is_value_positive(&self) -> bool {
    	unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    // use test_case::test_case;
    // use approx::assert_ulps_eq;

    #[test]
    fn construction() {
        let poly_matrix = vec![
        	array![[0, 1], [1, 0]],
        	array![[0, 1], [1, 0]]
        	];
        PolyMatrixGame::new(poly_matrix);
    }
}
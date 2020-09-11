use crate::MatrixGame;
use ndarray::Array2;
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
}

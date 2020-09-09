use crate::traits::Game;
use ndarray::{Array1, Array2};

/// [Matrix games](https://en.wikipedia.org/wiki/Zero-sum_game) are finite zero-sum two-player games.
#[derive(Debug)]
pub struct MatrixGame {
    matrix: Array2<f64>,
}

impl MatrixGame {
    pub fn new(matrix: Array2<f64>) -> Self
// where
	// 	T: Into<f64>,
    {
        // let matrix = Array2::from(
        // 	matrix.to_vec()
        // 		.iter()
        // 		.map(|x| -> f64 {x.into()})
        // 		.collect::<Vec<f64>>()
        // 	);
        MatrixGame { matrix }
    }
}

impl Game<(Array1<f64>, Array1<f64>, f64)> for MatrixGame {
    fn value(&self) -> Option<f64> {
        Some(self.solve().unwrap().2)
    }

    fn solve(&self) -> Option<(Array1<f64>, Array1<f64>, f64)> {
        // Define LP
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Maximize);

        // Setting
        let dimensions = self.matrix.shape();

        // Add row player strategy
        let mut row_strategy = Vec::new();
        for _ in 0..dimensions[0] {
            row_strategy.push(problem.add_var(0.0, (0.0, 1.0)));
        }

        // Add row player strategy
        let mut column_strategy = Vec::new();
        for _ in 0..dimensions[0] {
            column_strategy.push(problem.add_var(0.0, (0.0, 1.0)));
        }

        // Probabiltiy constrains
        let mut ones = Vec::new();
        for _ in 0..row_strategy.len() {
            ones.push(1.0);
        }
        problem.add_constraint(
            row_strategy.clone().into_iter().zip(ones),
            minilp::ComparisonOp::Le,
            1.0,
        );
        let mut ones = Vec::new();
        for _ in 0..row_strategy.len() {
            ones.push(1.0);
        }
        problem.add_constraint(
            row_strategy.clone().into_iter().zip(ones),
            minilp::ComparisonOp::Le,
            1.0,
        );

        // Matrix primal-dual constrain

        // Solve
        let solution = problem.solve().unwrap();

        // Retrieve the solution
        let value = solution.objective();
        let mut optimal_row_strategy = Vec::new();
        for var in row_strategy {
            optimal_row_strategy.push(solution[var]);
        }
        let mut optimal_column_strategy = Vec::new();
        for var in column_strategy {
            optimal_column_strategy.push(solution[var]);
        }

        Some((
            Array1::from(optimal_row_strategy),
            Array1::from(optimal_column_strategy),
            value,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn construction() {
        let matrix = array![[0., 1.], [1., 0.],];
        MatrixGame::new(matrix);
    }

    #[test]
    fn solving() {
        let matrix = array![[0., 1.], [1., 0.],];
        let matrix_game = MatrixGame::new(matrix);
        let solution = matrix_game.solve().unwrap();
        assert_eq!(0.0, solution.2);
    }
}

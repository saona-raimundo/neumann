use nalgebra::{DMatrix, DVector};
use num_rational::Ratio;
use polynomials::Polynomial;

use crate::{
    traits::{value_positivity::ValuePositivity, Solvable},
    DPolynomialMatrixGame,
};

// mod helper;

impl ValuePositivity for DPolynomialMatrixGame<i32> {
    type Perturbation = f64;
    /// Strategy for a player.
    type ValuePositivityCertificate = Vec<f64>;
    /// Strategy for a player.
    type UniformValuePositivityCertificate = Vec<f64>;
    type FunctionalFormCertificate = ();
    type Value = f64;
    /// Rational function.
    ///
    /// # Remarks
    ///
    /// The coefficients are integers, but represented as `f64` for efficient routins.
    type FunctionalForm = Ratio<Polynomial<f64>>;

    /// Returns a value for the perturbation such that
    /// kernels of optimal strategies are guaranteed not to change between this value and zero.
    ///
    /// In particular, the value function is a rational function between this value and zero.
    ///
    /// # Examples
    ///
    /// Error is noticed by the row player only if epsilon is greater than one.
    /// ```
    /// # use neumann::{PolynomialMatrixGame, traits::value_positivity::ValuePositivity};
    /// let poly_matrix_game = PolynomialMatrixGame::from([[[1, 1], [0, 0]], [[0, 0], [1, 1]]]).into_dynamic();
    /// assert!(poly_matrix_game.epsilon_kernel_constant() <= 1.);
    /// ```
    fn epsilon_kernel_constant(&self) -> <Self as ValuePositivity>::Perturbation {
        let m: f64 = self.nrows().max(self.ncols()) as f64;
        let b: f64 = self
            .matrices
            .iter()
            .map(|matrix_game| matrix_game.matrix.iter().map(|x| x.abs()).max().unwrap())
            .max()
            .unwrap() as f64;
        let k: f64 = (self.matrices.len() - 1) as f64;
        f64::min(
            1.0,
            (2. * m * k).powf(-m * k)
                * (b * m).powf(m * (1. - 2. * m * k))
                * (m * k + 1.).powf(1. - 2. * m * k),
        )
    }

    /// Returns `true` if the perturbed game has at least the value of the error-free game
    /// at a right neigborhood of zero.
    ///
    /// # Examples
    ///
    /// Positive error perturbations make it value positive.
    /// ```
    /// # use neumann::{PolynomialMatrixGame, traits::value_positivity::ValuePositivity};
    /// let p = PolynomialMatrixGame::from([ [[0, 1], [1, 0]], [[0, 1], [1, 0]] ]).into_dynamic();
    /// assert!(p.is_value_positive());
    /// ```
    fn is_value_positive(&self) -> bool {
        self.is_value_positive_checker(self.is_value_positive_certifying())
    }

    /// Checks if the polynomail matrix game has at least the value of the error-free game at a right neigborhood of zero.
    ///
    /// That is, there exists a positive threshold for which the value of the perturbed matrix game is at least as much as
    /// the value of the matrix game corresponding to evaluating the polynomial matrix game at zero.
    ///
    /// # Certificate
    ///
    /// If the answer is `true`, the certificate is a strategy for the row player that ensures a non-negative reward
    /// for the perturbation given by `epsilon_kernel_constant`.
    ///
    /// If the answer is `false`, the certificate is a strategy for the colmun player that ensures a negative reward
    /// for the column player, for the perturbation given by `epsilon_kernel_constant`.
    ///
    /// Recall that the value function does not change sign from between this perturbation and zero.
    fn is_value_positive_certifying(
        &self,
    ) -> (bool, <Self as ValuePositivity>::ValuePositivityCertificate) {
        let polynomial_matrix_game = self.clone().map(|x| -> f64 { x.into() });
        let (row_strategy, column_strategy, value) = {
            let matrix_game = polynomial_matrix_game.eval(&self.epsilon_kernel_constant());
            matrix_game.some_solution().unwrap() // Never fails by Nash theorem
        };
        match value >= polynomial_matrix_game.eval(&0.).value().unwrap() {
            // Never fails by Nash theorem
            true => (true, row_strategy),
            false => (false, column_strategy),
        }
    }

    /// Returns `true` if `certifying_output` is correct.
    ///
    /// Computes the value given by the certificate at `epsilon_kernel_constant`.
    fn is_value_positive_checker(
        &self,
        certifying_output: (bool, <Self as ValuePositivity>::ValuePositivityCertificate),
    ) -> bool {
        let polynomial_matrix_game = self.clone().map(|x| -> f64 { x.into() });
        let matrix = polynomial_matrix_game
            .eval(&self.epsilon_kernel_constant())
            .matrix;
        let proposal = certifying_output.1;
        if certifying_output.0 {
            // Checking that the strategy guarantees at least zero for the row player
            let reward = {
                let strategy = DMatrix::from_vec(1, proposal.len(), proposal); // PR for RowDVector::from(proposal.0);
                (strategy * matrix.clone()).min()
            };
            reward >= 0.0
        } else {
            // Checking that the strategy guarantees at least zero for the column player
            let reward = {
                let strategy = DVector::from(proposal);
                (matrix * strategy).min()
            };
            reward <= 0.0
        }
    }

    fn is_uniform_value_positive_certifying(
        &self,
    ) -> (
        bool,
        <Self as ValuePositivity>::UniformValuePositivityCertificate,
    ) {
        todo!()
    }
    fn is_uniform_value_positive_checker(
        &self,
        _: (
            bool,
            <Self as ValuePositivity>::UniformValuePositivityCertificate,
        ),
    ) -> bool {
        todo!()
    }

    fn functional_form_certifying(
        &self,
    ) -> (
        <Self as ValuePositivity>::FunctionalForm,
        <Self as ValuePositivity>::FunctionalFormCertificate,
    ) {
        // let kernel_matrix_of_polynomials = {
        //     let (nrows, ncols) = self.shape();
        //     let polynomial_matrix_game = self.clone().map(|x| -> f64 { x.into() });
        //     let matrix_game = polynomial_matrix_game.eval(&self.epsilon_kernel_constant());
        //     let (indixes_row, indixes_column, _sub_matrix_game) =
        //         matrix_game.kernel_completely_mixed();
        //     let remove_rows: Vec<usize> = (0..nrows)
        //         .filter(|row| !indixes_row.contains(row))
        //         .collect();
        //     let remove_columns: Vec<usize> = (0..ncols)
        //         .filter(|col| !indixes_column.contains(col))
        //         .collect();
        //     let matrix_of_polynomials: DMatrix<Polynomial<i32>> = self.clone().into();
        //     matrix_of_polynomials
        //         .remove_rows_at(&remove_rows)
        //         .remove_columns_at(&remove_columns)
        // };

        let kernel_polynomial_matrix_game = {
            let (nrows, ncols) = self.shape();
            let polynomial_matrix_game = self.clone().map(|x| -> f64 { x.into() });
            let matrix_game = polynomial_matrix_game.eval(&self.epsilon_kernel_constant());
            let (indixes_row, indixes_column, _sub_matrix_game) =
                matrix_game.kernel_completely_mixed();
            let remove_rows: Vec<usize> = (0..nrows)
                .filter(|row| !indixes_row.contains(row))
                .collect();
            let remove_columns: Vec<usize> = (0..ncols)
                .filter(|col| !indixes_column.contains(col))
                .collect();
            let mut polynomial_matrix_game = self.clone();
            for row in remove_rows {
                polynomial_matrix_game = polynomial_matrix_game.remove_row(row);
            }
            for col in remove_columns {
                polynomial_matrix_game = polynomial_matrix_game.remove_column(col);
            }
            polynomial_matrix_game
        };
        // Solving a completely mixed game has a formula
        let polynomial_matrix_game = kernel_polynomial_matrix_game.map(|x| -> f64 { x.into() });
        let numer = {
            let determinants: Vec<_> = polynomial_matrix_game
                .matrices
                .iter()
                .map(|matrix_game| matrix_game.matrix.determinant()) // helper::determinant(&matrix_game.matrix))
                .collect();
            Polynomial::from(determinants)
        };
        let denom = {
            let (nrows, ncols) = self.shape();
            let values: Vec<_> = polynomial_matrix_game
                .matrices
                .iter()
                .map(|matrix_game| -> f64 {
                    let mut sum = 0.0;
                    for row in 0..nrows {
                        for column in 0..ncols {
                            // cofactor
                            sum += matrix_game
                                .matrix
                                .clone()
                                .remove_row(row)
                                .remove_column(column)
                                .determinant();
                        }
                    }
                    sum
                })
                .collect();
            Polynomial::from(values)
        };

        (Ratio::new_raw(numer, denom), ())
    }

    /// Compute the value under `` and compare.
    ///
    /// # Remarks
    ///
    /// Allow accuracy erros, since we are working with `f64`.
    fn functional_form_checker(
        &self,
        _: (
            <Self as ValuePositivity>::FunctionalForm,
            <Self as ValuePositivity>::FunctionalFormCertificate,
        ),
    ) -> bool {
        // let test_points = {
        //     self.epsilon_kernel_constant()
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polynomials::poly;
    use test_case::test_case;

    use crate::PolynomialMatrixGame;

    #[test_case([ [[1]] ], true; "error-free")]
    #[test_case([ [[1]], [[-1]] ], false; "negative linear perturbation")]
    #[test_case( [ [[0, 1], [1, 0]], [[0, 1], [1, 0]] ],  true ; "value-positive easy")]
    #[test_case( [ [[0, 0], [-1, 1]], [[2, -1], [0, 0]] ],  true ; "value-positive medium")]
    #[test_case( [ [[0, 1], [1, 0]], [[0, -1], [-1, 0]] ],  false ; "not value-positive")]
    fn is_value_positive<const R: usize, const C: usize, const K: usize>(
        array: [[[i32; C]; R]; K],
        expected: bool,
    ) {
        let polynomial_matrix_game = PolynomialMatrixGame::from(array).into_dynamic();
        assert_eq!(polynomial_matrix_game.is_value_positive(), expected);
    }

    #[test_case( [ [[1, -1], [-1, 1]], [[1, -3], [0, 2]] ], Ratio::new_raw(poly![0., 0., 2.], poly![4., 6.]) ; "quadratic")]
    #[test_case( [ [[0, 0], [-1, 1]], [[2, -1], [0, 0]] ], Ratio::new_raw(poly![0., 1.], poly![2., 3.]) ; "linear")]
    fn functional_form<const R: usize, const C: usize, const K: usize>(
        array: [[[i32; C]; R]; K],
        expected: Ratio<Polynomial<f64>>,
    ) {
        let polynomial_matrix_game = PolynomialMatrixGame::from(array).into_dynamic();

        let functional_form = polynomial_matrix_game.functional_form();
        println!("{}", polynomial_matrix_game);
        println!("computed: {:?}", functional_form);
        println!("expected: {:?}", functional_form);

        panic!()
        // assert_eq!(functional_form.numer(), expected.numer());
        // assert_eq!(functional_form.denom(), expected.denom());
    }
}

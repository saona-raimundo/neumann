use core::ops::Neg;
use nalgebra::{DMatrix, Scalar};
use num_traits::{One, Zero};

/// Returns the determinant applying a recursive formula.
///
/// # Complexity
///
/// It involves computing `n!` products of polynomials.
pub fn determinant<T>(matrix: &DMatrix<T>) -> T
where
    T: Scalar + One + Zero + Neg<Output = T>,
{
    assert!(
        matrix.is_square(),
        "Unable to compute the determinant of a non-square matrix."
    );
    let (nrows, _ncols) = matrix.shape();
    if nrows == 1 {
        matrix[(0, 0)].clone()
    } else {
        (0..nrows)
            .map(|j| {
                if (j % 2) == 0 {
                    let sub_matrix = matrix.clone().remove_row(j).remove_column(0);
                    matrix[(j, 0)].clone() * determinant(&sub_matrix)
                } else {
                    let sub_matrix = matrix.clone().remove_row(j).remove_column(0);
                    matrix[(j, 0)].clone() * (-T::one()) * determinant(&sub_matrix)
                }
            })
            .fold(T::zero(), |acc, p| acc + p)
    }
}

/// Returns the determinant of the sub-matrix without a row and a column,
/// with the corresponding sign.
///
/// If `row + column` is odd, then it returns minus the determinant,
/// if it is even, the determinant.
pub fn cofactor<T>(matrix: &DMatrix<T>, row: usize, column: usize) -> T
where
    T: Scalar + One + Zero + Neg<Output = T>,
{
    let sub_matrix = matrix.clone().remove_row(row).remove_column(column);
    if (row + column) % 2 == 0 {
        determinant(&sub_matrix)
    } else {
        -determinant(&sub_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use test_case::test_case;

    #[test_case(dmatrix![1, 2; 3, 4], -2; "square")]
    #[test_case(dmatrix![1], 1; "unit")]
    fn computing_determinant<T>(matrix: DMatrix<T>, expected: T)
    where
        T: Scalar + One + Zero + Neg<Output = T> + core::fmt::Display,
    {
        println!("{}", matrix);
        assert_eq!(determinant(&matrix), expected);
    }

    #[test_case(dmatrix![1, 2; 3, 4], 0, 0, 4; "direct")]
    #[test_case(dmatrix![0, 0, 0; 0, 1, 2; 0, 3, 4], 0, 0, -2; "square")]
    // #[test_case(dmatrix![1], 1; "unit")]
    fn computing_cofactor<T>(matrix: DMatrix<T>, row: usize, column: usize, expected: T)
    where
        T: Scalar + One + Zero + Neg<Output = T> + core::fmt::Display,
    {
        println!("{}", matrix);
        assert_eq!(cofactor(&matrix, row, column), expected);
    }
}

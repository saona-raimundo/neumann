//! This module is mainly for internal usage.

use ndarray::Array2;
use polynomials::{poly, Polynomial};

/// Returns the determinant as a polynomial.
///
/// # Complexity
///
/// It involves computing `n!` products of polynomials.
pub fn determinant(poly_array: &Array2<Polynomial<i32>>) -> Polynomial<i32> {
    if poly_array.shape() == &[1, 1] {
        poly_array[[0, 0]].clone()
    } else {
        (0..poly_array.shape()[1])
            .map(|j| {
                if (j % 2) == 0 {
                    poly_array[[0, j]].clone() * determinant(&reduce_coordinate(poly_array, 0, j))
                } else {
                    poly_array[[0, j]].clone()
                        * (-1)
                        * determinant(&reduce_coordinate(poly_array, 0, j))
                }
            })
            .fold(poly![0], |acc, p| acc + p)
    }
}

/// Returns the determinant of the sub-matrix without a row and a column, with the corresponding sign.
///
/// If `row + column` is odd, then it returns minus the determinant, if it is even, the determinant.
pub fn cofactor(
    poly_array: &Array2<Polynomial<i32>>,
    row: usize,
    column: usize,
) -> Polynomial<i32> {
    if (row + column) % 2 == 0 {
        determinant(&reduce_coordinate(poly_array, row, column))
    } else {
        determinant(&reduce_coordinate(poly_array, row, column)) * (-1)
    }
}

/// Returns a sub-array by eliminating the corresponding row and column.
fn reduce_coordinate(
    poly_array: &Array2<Polynomial<i32>>,
    row: usize,
    column: usize,
) -> Array2<Polynomial<i32>> {
    let original_shape = [poly_array.shape()[0], poly_array.shape()[1]];
    let mut sub_poly_array = Vec::new();
    for i in 0..original_shape[0] {
        for j in 0..original_shape[1] {
            if !((i == row) || (j == column)) {
                sub_poly_array.push(poly_array[[i, j]].clone())
            }
        }
    }
    Array2::from_shape_vec(
        (original_shape[0] - 1, original_shape[1] - 1),
        sub_poly_array,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use polynomials::poly;
    use test_case::test_case;

    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], poly![1, -1] ; "linear")]
    #[test_case(array![[poly![1, 1], poly![0, 1]], [poly![1, 0], poly![0, 1]]], poly![0, 0, 1] ; "quadratic")]
    fn computing_determinant(poly_array: Array2<Polynomial<i32>>, expected: Polynomial<i32>) {
        assert_eq!(determinant(&poly_array), expected);
    }

    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 0, 0, poly![1, 1] ; "Two by two linear polynomials [0, 0]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 0, 1, poly![-1, 0] ; "Two by two linear polynomials [0, 1]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 1, 0, poly![0, -2] ; "Two by two linear polynomials [1, 0]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 1, 1, poly![1, 0] ; "Two by two linear polynomials [1, 1]")]
    fn computing_cofactor(
        poly_array: Array2<Polynomial<i32>>,
        row: usize,
        column: usize,
        expected: Polynomial<i32>,
    ) {
        assert_eq!(cofactor(&poly_array, row, column), expected);
    }

    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 0, 0, array![[poly![1, 1]]] ; "Two by two linear polynomials [0, 0]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 0, 1, array![[poly![1, 0]]] ; "Two by two linear polynomials [0, 1]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 1, 0, array![[poly![0, 2]]] ; "Two by two linear polynomials [1, 0]")]
    #[test_case(array![[poly![1, 0], poly![0, 2]], [poly![1, 0], poly![1, 1]]], 1, 1, array![[poly![1, 0]]] ; "Two by two linear polynomials [1, 1]")]
    fn reducing_coordinate(
        poly_array: Array2<Polynomial<i32>>,
        row: usize,
        column: usize,
        expected: Array2<Polynomial<i32>>,
    ) {
        assert_eq!(reduce_coordinate(&poly_array, row, column), expected);
    }
}

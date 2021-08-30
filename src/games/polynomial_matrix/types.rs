use nalgebra::{ArrayStorage, Const, Dynamic, VecStorage};

use crate::PolynomialMatrixGame;

pub type SPolynomialMatrixGame<T, const R: usize, const C: usize> =
    PolynomialMatrixGame<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
pub type DPolynomialMatrixGame<T> =
    PolynomialMatrixGame<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>;

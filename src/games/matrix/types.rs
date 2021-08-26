use nalgebra::{ArrayStorage, Const, Dynamic, VecStorage};

use crate::MatrixGame;

pub type SMatrixGame<T, const R: usize, const C: usize> =
    MatrixGame<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>;
pub type DMatrixGame<T> = MatrixGame<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>;

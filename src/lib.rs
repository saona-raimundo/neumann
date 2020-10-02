//! Game Theory algorithms in Rust.
//!
//! This library aims to be a good place to aid theoretical research in Game Theory.
//! Therefore, correctness is the main focus.
//!

pub use matrix_game::{MatrixGame, PolyMatrixGame};
pub use traits::Playable;

mod matrix_game;
mod traits;

// #[cfg(test)]
// mod tests {
//     // #[test]
//     // fn it_works() {
//     //     assert_eq!(2 + 2, 4);
//     // }
// }

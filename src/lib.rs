//! Game Theory algorithms in Rust.
//!
//! This library aims to be a good place to aid theoretical research in Game Theory.
//! Therefore, correctness is the main focus.
//!
//! # Structure
//!
//! Each research-specific properties and algorithms have a dedicated module and a main trait therein, 
//! while general games and their usual properties are directly in the main module.
//! 
//! As an example, consider polynomially-perturbed matrix games. They are represented by [PolyMatrixGame]
//! and implement the trait [ValuePositivity].
//!
//! [PolyMatrixGame]: struct.PolyMatrixGame.htm
//! [ValuePositivity]: value_positivity/trait.ValuePositivity.html

// Main crate
pub use matrix_game::{MatrixGame}; // , PolyMatrixGame};
pub use traits::Playable;
pub use certifying::Certified;
pub use stochastic_game::StochasticGame;

mod stochastic_game;
mod matrix_game;
mod traits;
mod certifying;

// Research specific
pub mod value_positivity;

// #[cfg(test)]
// mod tests {
//     // #[test]
//     // fn it_works() {
//     //     assert_eq!(2 + 2, 4);
//     // }
// }

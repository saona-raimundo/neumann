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

// Re-export all games
// pub use certifying::Certified;
pub use games::{
    matrix::{
        types::{DMatrixGame, SMatrixGame},
        MatrixGame,
    },
    polynomial_matrix::{
        types::{DPolynomialMatrixGame, SPolynomialMatrixGame},
        PolynomialMatrixGame,
    },
};

// mod certifying;

/// Equilibrium concepts.
pub mod equilibria;
/// All games supported.
pub mod games;
/// Traits of this crate.
pub mod traits;

pub mod prelude {
    pub use crate::games::matrix::MatrixGame;
    #[cfg(feature = "play")]
    pub use crate::traits::CliPlayable;
    pub use crate::traits::Solvable;
}

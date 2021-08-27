use nalgebra::{
    allocator::Allocator,
    constraint::{AreMultipliable, ShapeConstraint},
    DefaultAllocator, Dim, Dynamic, Storage,
};
use rand::thread_rng;
use rand_distr::{Distribution, WeightedError, WeightedIndex};
use thiserror::Error;

use crate::{
    traits::{CliPlayable, Solvable},
    MatrixGame,
};

#[derive(Debug, Error)]
pub enum CliError {
    #[error("Failed to construct a strategy.")]
    Weighted(#[from] WeightedError),
    #[error("Failed to get input.")]
    Processing(#[from] asking::error::Processing),
}

/// Play once applying mixed strategies!
// # Note
// Only f64 implements `Solvable`
impl<R, C, S> CliPlayable for MatrixGame<f64, R, C, S>
where
    S: Storage<f64, R, C> + Clone + std::fmt::Debug,
    R: Dim,
    C: Dim,
    DefaultAllocator:
        Allocator<f64, R, C> + Allocator<f64, C, R> + Allocator<f64, R> + Allocator<f64, C>,
    ShapeConstraint: AreMultipliable<R, C, Dynamic, Dynamic>
        + AreMultipliable<C, R, Dynamic, Dynamic>
        + AreMultipliable<Dynamic, Dynamic, R, C>
        + AreMultipliable<Dynamic, Dynamic, C, R>,
{
    type Outcome = f64;
    type Error = CliError;
    /// Starts a REPL to play the game.
    ///
    /// The user is asked to input a strategy, one probability at a time.
    /// For robustness, inputs are read as weights: a renormalization is performed to obtain the mixed strategy.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neumann::prelude::*;
    /// // With `play` feature enabled
    /// let matrix_game = MatrixGame::from([[0., 0.], [-1., 1.]]);
    /// matrix_game.play();
    /// ```
    fn play(&self) -> Result<f64, CliError> {
        println!(
            "Welcome! You two are playing the following matrix game:\n{}",
            self
        );

        println!(
            "You are the row player.\n\
            What mixed strategy are you going to use?\n\
            Enter a weight for each action, your strategy will be the corresponding mixed strategy."
        );

        let action_row = {
            let mixed_strategy = WeightedIndex::new(input_strategy(self.nrows())?)?;
            mixed_strategy.sample(&mut thread_rng())
        };

        println!(
            "You are the column player now.\n\
            What mixed strategy are you going to use?\n\
            Enter a weight for each action, your strategy will be the corresponding mixed strategy."
        );

        let action_column = {
            let mixed_strategy = WeightedIndex::new(input_strategy(self.nrows())?)?;
            mixed_strategy.sample(&mut thread_rng())
        };

        let result = self.matrix[(action_row, action_column)];
        println!("The result of this game is {}", result);

        Ok(result)
    }

    /// Starts a REPL to play the game as the row player against the computer.
    ///
    /// The computer will play an optimal mixed strategy according to the Mixed Nash equilibrium.
    /// The user is asked to input a strategy, one probability at a time in the form of weights.
    fn play_solo(&self) -> Result<f64, CliError> {
        println!(
            "Welcome! You are playing the following matrix game:\n{}",
            self
        );

        println!(
            "You are the raw player.\n\
            What mixed strategy are you going to use?\n\
            Enter a weight for each action, your strategy will be the corresponding mixed strategy."
        );

        let action_row = {
            let mixed_strategy_row = WeightedIndex::new(input_strategy(self.nrows())?)?;
            mixed_strategy_row.sample(&mut thread_rng())
        };

        let action_column = {
            let mixed_strategy_column = self.some_solution_for_player(1).unwrap(); // Never fails
            let mixed_strategy_column = WeightedIndex::new(mixed_strategy_column)?;
            mixed_strategy_column.sample(&mut thread_rng())
        };

        let result = self.matrix[(action_row, action_column)];
        println!("The result of this game is {}", result);

        Ok(result)
    }
}

fn input_strategy(num_actions: usize) -> Result<Vec<usize>, asking::error::Processing> {
    let mut weights = Vec::with_capacity(num_actions);
    for _ in 1..num_actions {
        let weight = asking::question()
            .default_value(0_usize)
            .repeat_help("Please enter a number.\n")
            .ask_and_wait()?;
        weights.push(weight)
    }
    let partial_sum: usize = weights.iter().sum();
    let weight = asking::question()
        .default_value(0_usize)
        .test_with_msg(
            move |v| v + partial_sum > 0,
            "You must give weight to at least one action!\n",
        )
        .repeat_help("Please enter a number.\n")
        .ask_and_wait()?;
    weights.push(weight);
    Ok(weights)
}

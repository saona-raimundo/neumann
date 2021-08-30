//! All traits of the crate.

/// Research specific traits.
pub mod value_positivity;

/// Games that can be played in an interactive command-line fashion.
#[cfg(feature = "play")]
pub trait CliPlayable {
    type Outcome;
    type Error: std::error::Error;

    /// Starts an opinionated version of the game as a cli application.
    fn play(&self) -> Result<Self::Outcome, Self::Error>;
    /// Starts an opinionated version of the game as a cli application,
    /// where the actions of other players are simulated.
    fn play_solo(&self) -> Result<Self::Outcome, Self::Error>;
}

/// [Normal] form (also called strategic) games
///
/// [Normal]: https://en.wikipedia.org/wiki/Normal-form_game
pub trait Normal<const N: usize> {
    /// Type representation of a real number.
    type Real;
    /// Returns the outcome of the game when each player plays according to `strategy_profile`.
    ///
    /// # Panics
    ///
    /// When the strategy profile is not valid.
    fn outcome(&self, strategy_profile: [usize; N]) -> [Self::Real; N];
    /// Returns the number of pure strategies players have.
    fn num_pure_strategies(&self) -> [usize; N];
}

/// Marker for equilibrium concepts.
pub trait Equilibrium {}

/// Games that can be solved in terms of the given equilibrium concept.
pub trait Solvable<E: Equilibrium> {
    type Value;
    type PlayerStrategy;
    type Solution;
    type SolutionIter: Iterator<Item = Self::Solution>;
    /// Returns all possible solutions.
    ///
    /// # Remarks
    ///
    /// Usually, there are infinitely many solutions.
    /// The representation of this infinite set should be documented by the implementation.
    fn all_solutions(&self) -> Self::SolutionIter;
    /// Checks whether the game has a solution.
    fn is_solvable(&self) -> bool;
    /// Checks whether `proposal` is a solution.
    fn is_solution(&self, proposal: Self::Solution) -> bool;
    /// Returns a possible solution.
    ///
    /// # Remarks
    ///
    /// If the game is solvable, this function never returns `None`.
    fn some_solution(&self) -> Option<Self::Solution>;
    /// Returns a possible solution.
    ///
    /// # Remarks
    ///
    /// If the game is solvable, this function never returns `None`.
    fn some_solution_for_player(&self, player: usize) -> Option<Self::PlayerStrategy>;
    /// Returns the value of the game, if it is solvable.
    fn value(&self) -> Option<Self::Value>;
}

/// Games that can be played in an interactive command-line fashion.
pub trait CliPlayable {
    /// Starts an opinionated version of the game as a cli application.
    fn play(&self);
    /// Starts an opinionated version of the game as a cli application,
    /// where the actions of other players are simulated.
    fn play_solo(&self);
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
    type PlayerStrategy;
    type Solution;
    /// Checks whether the game has a solution.
    fn is_solvable(&self) -> bool;
    /// Returns a possible solution.
    ///
    /// # Remarks
    ///
    /// If the game is solvable, this function never returns `None`.
    fn some_solution(&self) -> Option<Self::Solution>;
    /// Checks whether `proposal` is a solution.
    fn is_solution(&self, proposal: Self::Solution) -> bool;
    /// Returns a possible solution.
    ///
    /// # Remarks
    ///
    /// If the game is solvable, this function never returns `None`.
    fn some_solution_for_player(&self, player: usize) -> Option<Self::PlayerStrategy>;
    /// Returns all possible solutions.
    ///
    /// # Remarks
    ///
    /// Usually, there are infinitely many solutions.
    /// The representation of this infinite set should be documented by the implementation.
    fn all_solutions(&self) -> dyn Iterator<Item = Self::Solution>;
}

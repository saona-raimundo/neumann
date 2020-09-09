/// Based on the work of Nash
pub trait Game<S> {
    /// Returns a Nash equilibrium and the value of the game, if they exist.
    fn solve(&self) -> Option<S>;
    /// Returns the value of the game, if it exists.
    fn value(&self) -> Option<f64>;
}

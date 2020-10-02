/// Games that can be played in an interactive manner.
pub trait Playable {
    /// Starts a REPL to play the game.
    fn play(&self);
}

use crate::traits::CliPlayable;
use crate::MatrixGame;

// impl<T, R, C, S> CliPlayable for MatrixGame<T, R, C, S> {
//     fn input_strategy(&self) -> [f64] {
//         println!("Enter you mixed strategy, one probability at a time.");

//         let mut weights = Array1::from_elem(self.actions_row(), 0.);
//         for i in 0..self.actions_row() {
//             let mut weight = String::new();

//             std::io::stdin()
//                 .read_line(&mut weight)
//                 .expect("Failed to read line");

//             let mut ns = fasteval::EmptyNamespace;
//             match fasteval::ez_eval(&weight, &mut ns) {
//                 Ok(val) => {
//                     if val.is_sign_positive() {
//                         weights[i] = val
//                     } else {
//                         eprintln!("Probabilities must be greater or equal to zero");
//                         break;
//                     }
//                 }
//                 Err(e) => {
//                     eprintln!("{}", e);
//                     break;
//                 }
//             }
//         }

//         weights
//     }
// }

impl<T, R, C, S> CliPlayable for MatrixGame<T, R, C, S> {
    /// Starts a REPL to play the game.
    ///
    /// The user is asked to input a strategy, one probability at a time.
    /// For robustness, inputs are read as weights: a renormalization is performed to obtain the mixed strategy.
    ///
    /// # Remarks
    ///
    /// Values are parsed using the [fasteval] crate, accepting a big range of inputs.
    ///
    /// [fasteval]: https://crates.io/crates/fasteval
    fn play(&self) {
        todo!()
        // println!(
        //     "Welcome! You are playing the following matrix game:\n{}",
        //     self
        // );

        // loop {
        //     let weights = self.input_strategy();

        //     // Reward
        //     let reward = weights
        //         .dot(&self.matrix)
        //         .iter()
        //         .cloned()
        //         .fold(std::f64::NAN, f64::min)
        //         / weights.sum();
        //     println!("You obtained: {}\n", reward);

        //     // Repeating?
        //     println!("Keep playing?(y/n)");
        //     let mut repeat = String::new();
        //     std::io::stdin()
        //         .read_line(&mut repeat)
        //         .expect("Failed to read line");
        //     if !(repeat.trim() == "y") {
        //         println!("Thank you for playing!");
        //         break;
        //     }
        // }
    }
    fn play_solo(&self) {
        todo!()
    }
}

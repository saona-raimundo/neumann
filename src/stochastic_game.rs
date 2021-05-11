use std::collections::HashMap;

/// Objective of the players
#[derive(Debug, Clone, PartialEq, Copy)]
enum Objective {
    /// Normalized discounted mean
    ///
    /// \lambda \sum_{i >= 0} (1 - \lambda)^i <reward_at_time_i>
    Discounted(f64),
    // Total(usize),
    // Average(usize),
}

/// Follows [Shapley53].
///
/// [Shapley53]: https://doi.org/10.1073/pnas.39.10.1095
pub struct StochasticGame<const STATES: usize> {
    actions_one: [usize; STATES], // Actions available per state
    actions_two: [usize; STATES], // Actions available per state
    transition: HashMap<([usize; STATES], [usize; STATES]), [[f64; STATES]; STATES]>, // Transition matrix
    reward: HashMap<([usize; STATES], [usize; STATES]), [f64; STATES]>, // Rewards
    current_state: usize,
    objective: Objective,
}

impl<const STATES: usize> StochasticGame<STATES> {
    // pub new(
    //     actions_one: [usize; STATES],
    //     actions_two: [usize; STATES],
    //     transition: Fn: ([usize; STATES], [usize; STATES]) -> [[f64; STATES]; STATES], // Transition matrix
    //     reward: Fn: ([usize; STATES], [usize; STATES]) -> [R; STATES],
    //     initial_state: usize,
    //     objective: Objective,
    // ) -> Self {}
    // pub fn rewards(action_profile: ([usize; STATES], [usize; STATES])) -> Result<[R; STATES]> {}
    // pub fn transition_matrix(action_profile: ([usize; STATES], [usize; STATES])) -> Result<[[f64; STATES]; STATES]> {}
    // pub fn action_profiles(&self) -> Vec<([usize; STATES], [usize; STATES])>
    // pub fn fix_stationary_strategy(&self, stationary_strategy: [Vec<f64>; STATES]) -> MarkovDecissionProcess {}
}
// impl StochasticGame {
//     pub fn aux_matrix_game(state: usize, z: f64) -> MatrixGame {}
//     fn null_determinant(action_profile: ([usize; STATES], [usize; STATES])) -> f64 {
//         // Construct ID - (1 - \lambda) Q
//         // Compute determinant
//     }
//     fn state_determinant(state: usize, action_profile: ([usize; STATES], [usize; STATES])) -> f64 {
//         // Construct ID - (1 - \lambda) Q
//         // Replace the state-column
//         // Compute determinant
//     }    
// }
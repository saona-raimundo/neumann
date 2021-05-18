use std::collections::HashMap;
use neumann::{stochastic_game::Objective, StochasticGame};
use pyo3::Python;

fn main() {
    // Parameters
    let actions_one = [1, 1];
    let actions_two = [1, 1];
    let mut transition = HashMap::new();
    transition.insert(([1, 1], [1, 1]), [[1., 0.], [0., 1.]]);
    let mut reward = HashMap::new();
    reward.insert(([1, 1], [1, 1]), [0., 1.]);
    let initial_state = 0;
    let objective = Objective::Discounted(0.5);
    // Construction
    let stoc_game = StochasticGame::new(
        actions_one,
        actions_two,
        |x, y| transition[&(x, y)],
        |x, y| reward[&(x, y)],
        initial_state,
        objective,
    );

	let gil = Python::acquire_gil();
    let py = gil.python();
    let state = 0;
    let aux_matrix_game = stoc_game.sym_aux_matrix_game(state, py);
    println!("{:?}", aux_matrix_game);


}

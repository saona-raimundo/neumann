use neumann::{stochastic_game::Objective, StochasticGame};
use pyo3::Python;

fn main() {
    // Parameters
    let actions_one = [1, 2, 2, 1];
    let actions_two = [1, 1, 1, 1];
    #[rustfmt::skip]
    let transition = vec![
    	( (0, (0, 0)), [1., 0., 0., 0.] ),
    	( (1, (0, 0)), [0., 0., 0., 1.] ),
    	( (1, (1, 0)), [0., 0., 1., 0.] ),
    	( (2, (0, 0)), [0., 0., 0., 1.] ),
    	( (2, (1, 0)), [0.5, 0.5, 0., 0.] ),
    	( (3, (0, 0)), [0., 0., 0., 1.] ),
    	].into_iter().collect();
    let reward = vec![
        ((0, (0, 0)), 0.),
        ((1, (0, 0)), 0.),
        ((1, (1, 0)), 0.),
        ((2, (0, 0)), 0.),
        ((2, (1, 0)), 0.),
        ((3, (0, 0)), 1.),
    ]
    .into_iter()
    .collect();
    let initial_state = 0;
    let objective = Objective::Discounted(0.5);
    // Construction
    let stoc_game = StochasticGame::new(
        actions_one,
        actions_two,
        transition,
        reward,
        initial_state,
        objective,
    )
    .unwrap();

    // Auxiliary matrix game
    let gil = Python::acquire_gil();
    let py = gil.python();
    let state = 1;

    let aux_matrix_game = stoc_game.sym_aux_matrix_game(state, py).unwrap();
    println!("{:#?}", aux_matrix_game);
}

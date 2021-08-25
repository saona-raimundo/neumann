use neumann::{stochastic_game::Objective, StochasticGame};
use pyo3::Python;

fn main() {
    if true {
        // Parameters
        let actions_one = [1, 2, 2, 1];
        let actions_two = [1, 1, 1, 1];
        let transition = vec![
            ((0, (0, 0)), [1., 0., 0., 0.]),
            ((1, (0, 0)), [0., 0., 0., 1.]),
            ((1, (1, 0)), [0., 0., 1., 0.]),
            ((2, (0, 0)), [0., 0., 0., 1.]),
            ((2, (1, 0)), [0.5, 0.5, 0., 0.]),
            ((3, (0, 0)), [0., 0., 0., 1.]),
        ]
        .into_iter()
        .collect();
        let reward = vec![
            ((0, (0, 0)), 0.),
            ((1, (0, 0)), 0.),
            ((1, (1, 0)), 0.),
            ((2, (0, 0)), 1.),
            ((2, (1, 0)), 0.),
            ((3, (0, 0)), 1.),
        ]
        .into_iter()
        .collect();
        let initial_state = 0;
        let objective = Objective::Discounted(1. / 3.);
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
        println!("lambda: {:?}", objective);
        let z = 1. / 2.;
        let aux_matrix_game = stoc_game.aux_matrix_game(1, z).unwrap();
        println!("For z = {}", z);
        println!("{}", aux_matrix_game);

        // Symbolic
        let gil = Python::acquire_gil();
        let py = gil.python();
        let state = 1;

        let aux_matrix_game = stoc_game.sym_aux_matrix_game(state, py).unwrap();
        println!("{:#?}", aux_matrix_game);
    }

    if false {
        // Parameters
        let actions_one = [1, 2, 2, 1];
        let actions_two = [1, 2, 1, 1];
        let transition = vec![
            ((0, (0, 0)), [1., 0., 0., 0.]),
            ((1, (0, 0)), [0., 0., 0., 1.]),
            ((1, (0, 1)), [0., 0., 1., 0.]),
            ((1, (1, 0)), [0., 0., 1., 0.]),
            ((1, (1, 1)), [0., 0., 0., 1.]),
            ((2, (0, 0)), [0., 0., 0., 1.]),
            // ((2, (0, 1)), [0.5, 0.5, 0., 0.]),
            ((2, (1, 0)), [0.5, 0.5, 0., 0.]),
            // ((2, (1, 1)), [0., 0., 0., 1.]),
            ((3, (0, 0)), [0., 0., 0., 1.]),
        ]
        .into_iter()
        .collect();
        let reward = vec![
            ((0, (0, 0)), 0.),
            ((1, (0, 0)), 0.),
            ((1, (0, 1)), 0.),
            ((1, (1, 0)), 0.),
            ((1, (1, 1)), 0.),
            ((2, (0, 0)), 0.),
            // ((2, (0, 1)), 0.),
            ((2, (1, 0)), 0.),
            // ((2, (1, 1)), 0.),
            ((3, (0, 0)), 1.),
        ]
        .into_iter()
        .collect();
        let initial_state = 0;
        let objective = Objective::Discounted(1. / 3.);
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

        println!("lambda: {:?}", objective);
        let z = 10. / 18.;
        let aux_matrix_game = stoc_game.aux_matrix_game(state, z).unwrap();
        println!("For z = {}", z);
        println!("{}", aux_matrix_game);

        let error = 1e-20;
        let approx_value = stoc_game.approx_value(state, error);
        println!("lambda: {:?}\nvalue: {:?}", objective, approx_value);
    }

    if false {
        // Parameters
        let actions_one = [2, 2];
        let actions_two = [1, 2];
        let transition = vec![
            ((0, (0, 0)), [0.5, 0.5]),
            ((0, (1, 0)), [0.5, 0.5]),
            ((1, (0, 0)), [0.5, 0.5]),
            ((1, (1, 0)), [0.5, 0.5]),
            ((1, (0, 1)), [0.5, 0.5]),
            ((1, (1, 1)), [0.5, 0.5]),
        ]
        .into_iter()
        .collect();
        let reward = vec![
            ((0, (0, 0)), 1.),
            ((0, (1, 0)), 0.),
            ((1, (0, 0)), 1.),
            ((1, (1, 0)), 0.),
            ((1, (0, 1)), 0.),
            ((1, (1, 1)), 1.),
        ]
        .into_iter()
        .collect();
        let initial_state = 0;
        let objective = Objective::Discounted(1. / 3000.);
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
        println!("{:?}", stoc_game.action_profiles().collect::<Vec<_>>());
        println!("{:#?}", aux_matrix_game);

        println!("lambda: {:?}", objective);
        let z = 0.7;
        let aux_matrix_game = stoc_game.aux_matrix_game(state, z).unwrap();
        println!("For z = {}", z);
        println!("{}", aux_matrix_game);

        let error = 1e-20;
        let approx_value = stoc_game.approx_value(state, error);
        println!("lambda: {:?}\nvalue: {:?}", objective, approx_value);
    }

    if false {
        // Parameters
        let actions_one = [1, 2, 2, 1];
        let actions_two = [1, 3, 2, 1];
        let transition = vec![
            ((0, (0, 0)), [1., 0., 0., 0.]),
            ((1, (0, 0)), [0., 0., 0., 1.]),
            ((1, (0, 1)), [0., 0., 1., 0.]),
            ((1, (0, 2)), [0., 0.5, 0.5, 0.]),
            ((1, (1, 0)), [0., 0., 1., 0.]),
            ((1, (1, 1)), [0., 0., 0., 1.]),
            ((1, (1, 2)), [0., 0.5, 0.5, 0.]),
            ((2, (0, 0)), [0., 0., 0., 1.]),
            ((2, (0, 1)), [0., 0.5, 0.5, 0.]),
            ((2, (1, 0)), [0.5, 0.5, 0., 0.]),
            ((2, (1, 1)), [0., 0.5, 0.5, 0.]),
            ((3, (0, 0)), [0., 0., 0., 1.]),
        ]
        .into_iter()
        .collect();
        let reward = vec![
            ((0, (0, 0)), 0.),
            ((1, (0, 0)), 0.),
            ((1, (0, 1)), 0.),
            ((1, (0, 2)), 1.),
            ((1, (1, 0)), 0.),
            ((1, (1, 1)), 0.),
            ((1, (1, 2)), 0.),
            ((2, (0, 0)), 1.),
            ((2, (0, 1)), 0.),
            ((2, (1, 0)), 0.),
            ((2, (1, 1)), 1.),
            ((3, (0, 0)), 1.),
        ]
        .into_iter()
        .collect();
        let initial_state = 0;
        let objective = Objective::Discounted(1. / 3000000.);
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

        println!("lambda: {:?}", objective);
        let z = 10. / 18.;
        let aux_matrix_game = stoc_game.aux_matrix_game(state, z).unwrap();
        println!("For z = {}", z);
        println!("{}", aux_matrix_game);
        println!("{}", aux_matrix_game.matrix() * 10000000000000.);

        let error = 1e-20;
        let approx_value = stoc_game.approx_value(state, error);
        println!("lambda: {:?}\nvalue: {:?}", objective, approx_value);
    }

    if true {
        // Parameters
        let actions_one = [1, 2, 2, 1];
        let actions_two = [1, 2, 2, 1];
        let transition = vec![
            ((0, (0, 0)), [1., 0., 0., 0.]),
            ((1, (0, 0)), [0., 0., 0., 1.]),
            ((1, (0, 1)), [0., 0., 1., 0.]),
            ((1, (1, 0)), [0., 0., 1., 0.]),
            ((1, (1, 1)), [0., 0., 0., 1.]),
            ((2, (0, 0)), [0., 0., 0., 1.]),
            ((2, (0, 1)), [0.5, 0.5, 0., 0.]),
            ((2, (1, 0)), [0.5, 0.5, 0., 0.]),
            ((2, (1, 1)), [0., 0., 0., 1.]),
            ((3, (0, 0)), [0., 0., 0., 1.]),
        ]
        .into_iter()
        .collect();
        let reward = vec![
            ((0, (0, 0)), 0.),
            ((1, (0, 0)), 0.),
            ((1, (0, 1)), 0.),
            ((1, (1, 0)), 0.),
            ((1, (1, 1)), 0.),
            ((2, (0, 0)), 1.),
            ((2, (0, 1)), 0.),
            ((2, (1, 0)), 0.),
            ((2, (1, 1)), 1.),
            ((3, (0, 0)), 1.),
        ]
        .into_iter()
        .collect();
        let initial_state = 0;
        let objective = Objective::Discounted(1. / 30000000.);
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

        println!("lambda: {:?}", objective);
        let z = 0.75; // 10. / 18.;
        let aux_matrix_game = stoc_game.aux_matrix_game(state, z).unwrap();
        println!("For z = {}", z);
        println!("{}", aux_matrix_game);

        let error = 1e-20;
        let approx_value = stoc_game.approx_value(state, error);
        println!("lambda: {:?}\nvalue: {:?}", objective, approx_value);
        let z = 0.99;
        println!("For z = {}", z);
        println!("{:?}", stoc_game.aux_matrix_game(1, z).unwrap().solve());
        use neumann::Playable;
        stoc_game.aux_matrix_game(1, z).unwrap().play();
    }
}

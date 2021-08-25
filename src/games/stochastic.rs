use crate::MatrixGame;
use action_profiles::ActionProfiles;
use std::collections::HashMap;

mod action_profiles;

/// Objective of the players
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Objective {
    /// Normalized discounted mean
    ///
    /// \lambda \sum_{i >= 0} (1 - \lambda)^i <reward_at_time_i>
    Discounted(f64),
    /// Mean of the first stages
    ///
    /// \frac{1}{n} \sum_{i=0}^{n-1} <reward_at_time_i>
    Average(usize),
}

/// Follows [[Shapley53]].
///
/// [Shapley53]: https://doi.org/10.1073/pnas.39.10.1095
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticGame<const STATES: usize> {
    actions_one: [usize; STATES], // Actions available per state
    actions_two: [usize; STATES], // Actions available per state
    transition: HashMap<(usize, (usize, usize)), [f64; STATES]>, // Local transition rule
    reward: HashMap<(usize, (usize, usize)), f64>, // Local rewards
    current_state: usize,
    objective: Objective,
}

impl<const STATES: usize> StochasticGame<STATES> {
    /// Creates a new `StochasticGame`.
    ///
    /// # Errors
    ///
    /// If the keys in `local_transition` or `local_reward` are incomplete.
    pub fn new(
        actions_one: [usize; STATES],
        actions_two: [usize; STATES],
        local_transition: HashMap<(usize, (usize, usize)), [f64; STATES]>, // Transition matrix
        local_reward: HashMap<(usize, (usize, usize)), f64>,
        initial_state: usize,
        objective: Objective,
    ) -> anyhow::Result<Self> {
        let transition_keys = {
            let mut vec = local_transition
                .keys()
                .collect::<Vec<&(usize, (usize, usize))>>();
            vec.sort();
            vec
        };
        let reward_keys = {
            let mut vec = local_reward
                .keys()
                .collect::<Vec<&(usize, (usize, usize))>>();
            vec.sort();
            vec
        };
        anyhow::ensure!(
            transition_keys.len() == reward_keys.len(),
            "local transition and rewards must have the same number of keys. \
            There are {} transition keys, \
            while there are {} reward keys.",
            transition_keys.len(),
            reward_keys.len(),
        );
        for i in 0..transition_keys.len() {
            anyhow::ensure!(
                transition_keys[i] == reward_keys[i],
                "local transition and rewards must be defined with the same keys. \
                Check transition key {:?}, or reward key {:?}",
                transition_keys[i],
                reward_keys[i],
            );
        }

        let mut transition_keys_iter = transition_keys.into_iter();
        for state in 0..STATES {
            for action_one in 0..actions_one[state] {
                for action_two in 0..actions_two[state] {
                    let next = transition_keys_iter.next();
                    if let Some((s, (a1, a2))) = next {
                        if a1 > &(actions_one[state] - 1) {
                            anyhow::bail!(
                                "Incongruent actions for player one. \
                                There are only {} actions for player one at state {}, \
                                but the key {:?} was used.",
                                actions_one[state],
                                state,
                                (s, (a1, a2)),
                            );
                        } else if a2 > &(actions_two[state] - 1) {
                            anyhow::bail!(
                                "Incongruent actions for player two. \
                                There are only {} actions for player two at state {}, \
                                but the key {:?} was used.",
                                actions_two[state],
                                state,
                                (s, (a1, a2)),
                            );
                        } else if s > &STATES {
                            anyhow::bail!(
                                "Incongruent number of states. \
                                There are only {} states in the game, \
                                but the key {:?} was used.",
                                STATES,
                                (s, (a1, a2)),
                            );
                        } else {
                            anyhow::ensure!(
                                (s, (a1, a2)) == (&state, (&action_one, &action_two)),
                                "number of actions and keys do not match. \
                                They key {:?} was expected, \
                                while the key {:?} was given.",
                                (state, (action_one, action_two)),
                                (s, (a1, a2))
                            );
                        }
                    } else {
                        anyhow::bail!(
                            "Insufficient keys.\
                            There key {:?} is a missing.",
                            (state, (action_one, action_two))
                        );
                    }
                }
            }
        }

        Ok(StochasticGame {
            actions_one,
            actions_two,
            transition: local_transition,
            reward: local_reward,
            current_state: initial_state,
            objective,
        })
    }

    /// Creates a new `StochasticGame`.
    ///
    /// # Remarks
    ///
    /// Functions are called once per input convination.
    pub fn from_fn(
        actions_one: [usize; STATES],
        actions_two: [usize; STATES],
        mut local_transition: impl FnMut(usize, (usize, usize)) -> [f64; STATES], // Transition matrix
        mut local_reward: impl FnMut(usize, (usize, usize)) -> f64,
        initial_state: usize,
        objective: Objective,
    ) -> Self {
        let mut transition = HashMap::new();
        for state in 0..STATES {
            for action_one in 0..actions_one[state] {
                for action_two in 0..actions_two[state] {
                    transition.insert(
                        (state, (action_one, action_two)),
                        local_transition(state, (action_one, action_two)),
                    );
                }
            }
        }
        let mut reward = HashMap::new();
        for state in 0..STATES {
            for action_one in 0..actions_one[state] {
                for action_two in 0..actions_two[state] {
                    reward.insert(
                        (state, (action_one, action_two)),
                        local_reward(state, (action_one, action_two)),
                    );
                }
            }
        }

        StochasticGame {
            actions_one,
            actions_two,
            transition,
            reward,
            current_state: initial_state,
            objective,
        }
    }
    /// Returns the rewards at each state, for the given action profile.
    pub fn rewards(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
    ) -> anyhow::Result<[&f64; STATES]> {
        let mut result = [&0.; STATES];
        for state in 0..STATES {
            result[state] =
                self.reward_at_state(state, (action_profile.0[state], action_profile.1[state]))?
        }
        Ok(result)
    }
    /// Returns the rewards at the given state, for the given local actions.
    pub fn reward_at_state(
        &self,
        state: usize,
        local_actions: (usize, usize),
    ) -> anyhow::Result<&f64> {
        self.reward
            .get(&(state, local_actions))
            .ok_or(anyhow::anyhow!("Invalid local actions"))
    }
    /// Returns the transition between states, for the given action profile.
    pub fn transition_matrix(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
    ) -> anyhow::Result<[&[f64; STATES]; STATES]> {
        let mut result = [&[0.; STATES]; STATES];
        for state in 0..STATES {
            result[state] =
                self.transition_at_state(state, (action_profile.0[state], action_profile.1[state]))?
        }
        Ok(result)
    }
    /// Returns the transition between states, for the given action profile.
    pub fn transition_at_state(
        &self,
        state: usize,
        local_actions: (usize, usize),
    ) -> anyhow::Result<&[f64; STATES]> {
        self.transition
            .get(&(state, local_actions))
            .ok_or(anyhow::anyhow!("Invalid local actions"))
    }
    /// Returns an iterator over all action profiles.
    pub fn action_profiles(&self) -> ActionProfiles<STATES> {
        ActionProfiles::new(self.actions_one, self.actions_two)
    }
    /// Returns `true` if the action profile is allowed in the game.
    ///
    /// # Remarks
    ///
    /// Recall that actions are indexed starting from zero.
    pub fn has_action_profile(&self, action_profile: ([usize; STATES], [usize; STATES])) -> bool {
        (0..STATES).all(|state| {
            (action_profile.0[state] < self.actions_one[state])
                && (action_profile.1[state] < self.actions_two[state])
        })
    }
    // PROJECT
    // pub fn fix_stationary_strategy(&self, stationary_strategy: [Vec<f64>; STATES]) -> MarkovDecissionProcess {}
    // REQUIRES
    // struct MarkovDecissionProcess
}

/// Numerical implementation
impl<const STATES: usize> StochasticGame<STATES>
where
    nalgebra::Const<STATES>:
        nalgebra::DimMin<nalgebra::Const<STATES>, Output = nalgebra::Const<STATES>>,
{
    pub fn approx_value(&self, state: usize, error: f64) -> anyhow::Result<f64> {
        assert!(error > 0.);
        use itertools::Itertools;
        let (mut lower_bound, mut upper_bound): (f64, f64) = self
            .action_profiles()
            .map(|x| self.rewards(x).unwrap().iter().cloned().collect::<Vec<_>>())
            .flatten()
            .cloned()
            .minmax()
            .into_option()
            .unwrap();
        while (upper_bound - lower_bound).abs() > error {
            let z = (lower_bound + upper_bound) / 2.;
            anyhow::ensure!(
                upper_bound > z && z > lower_bound,
                "Machine precision reached!\nThe best approximation is {}",
                z
            );
            // println!("{:?}", z);
            let value_z = self.aux_matrix_game(state, z).unwrap().value();
            if value_z > 0. {
                lower_bound = z;
            } else {
                upper_bound = z;
            }
        }
        Ok((lower_bound + upper_bound) / 2.)
    }
    pub fn aux_matrix_game(&self, state: usize, z: f64) -> anyhow::Result<MatrixGame> {
        let dimension_one = self.actions_one.iter().product();
        let dimension_two: usize = self.actions_two.iter().product();
        let mut aux_matrix = ndarray::Array2::<f64>::zeros((dimension_one, dimension_two));
        for (counter, action_profile) in self.action_profiles().enumerate() {
            // Define the corresponding index
            let i = counter / dimension_two;
            let j = counter % dimension_two;
            // Compute entries
            let d0 = self.null_determinant(action_profile)?;
            let dk = self.state_determinant(state, action_profile)?;
            // update entry
            aux_matrix[[i, j]] = dk - z * d0;
        }
        Ok(MatrixGame::from(aux_matrix))
    }

    fn null_determinant(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
    ) -> anyhow::Result<f64> {
        let lambda;
        if let Objective::Discounted(param) = self.objective {
            lambda = param;
        } else {
            return Err(anyhow::anyhow!("The objective must be discounted"));
        }
        // Construct ID - (1 - \lambda) Q
        let transition_matrix = self.transition_matrix(action_profile).unwrap();
        let mut new_matrix = nalgebra::SMatrix::<f64, STATES, STATES>::identity();
        for k1 in 0..STATES {
            for k2 in 0..STATES {
                new_matrix[(k1, k2)] += (lambda - 1.) * transition_matrix[k1][k2];
            }
        }
        // Compute determinant
        Ok(new_matrix.determinant())
    }
    /// Determinant of `ID - (1 - \lambda) Q` change the state column for the reward vector
    fn state_determinant(
        &self,
        state: usize,
        action_profile: ([usize; STATES], [usize; STATES]),
    ) -> anyhow::Result<f64> {
        assert!(state < STATES);
        let lambda;
        if let Objective::Discounted(param) = self.objective {
            lambda = param;
        } else {
            return Err(anyhow::anyhow!("The objective must be discounted"));
        }
        // Construct ID - (1 - \lambda) Q
        let transition_matrix = self.transition_matrix(action_profile)?;
        let rewards = self.rewards(action_profile)?;
        let mut new_matrix = nalgebra::SMatrix::<f64, STATES, STATES>::identity();
        for k1 in 0..STATES {
            for k2 in 0..STATES {
                if k2 == state {
                    new_matrix[(k1, k2)] = lambda * *rewards[k1];
                } else {
                    new_matrix[(k1, k2)] += (lambda - 1.) * transition_matrix[k1][k2];
                }
            }
        }
        // Compute determinant
        Ok(new_matrix.determinant())
    }
}

/// Symbolic computations
impl<const STATES: usize> StochasticGame<STATES>
where
    nalgebra::Const<STATES>:
        nalgebra::DimMin<nalgebra::Const<STATES>, Output = nalgebra::Const<STATES>>,
{
    pub fn sym_aux_matrix_game<'a>(
        &self,
        state: usize,
        py: pyo3::Python<'a>,
    ) -> anyhow::Result<Vec<Vec<String>>> {
        let dimension_one = self.actions_one.iter().product();
        let dimension_two: usize = self.actions_two.iter().product();
        let mut aux_matrix = vec![vec![String::from(""); dimension_two]; dimension_one];
        for (counter, action_profile) in self.action_profiles().enumerate() {
            // Define the corresponding index
            let i = counter / dimension_two;
            let j = counter % dimension_two;
            // Compute entries
            let d0 = self.sym_null_determinant(action_profile, py)?;
            let dk = self.sym_state_determinant(state, action_profile, py)?;
            // update entry
            aux_matrix[i][j] = format!("{} - z ({})", dk, d0);
        }
        Ok(aux_matrix)
    }

    fn sym_null_determinant<'a>(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
        py: pyo3::Python<'a>,
    ) -> anyhow::Result<&'a pyo3::PyAny> {
        // Matrix
        let lambda_matrix = self.sym_lambda_matrix(action_profile, py)?;
        // Computation
        let sympy = pyo3::prelude::PyModule::import(py, "sympy")?;
        let result = sympy.call1("det", (lambda_matrix,))?;
        Ok(result)
    }
    /// Determinant of `ID - (1 - \lambda) Q` changing the state column for the reward vector
    fn sym_state_determinant<'a>(
        &self,
        state: usize,
        action_profile: ([usize; STATES], [usize; STATES]),
        py: pyo3::Python<'a>,
    ) -> anyhow::Result<&'a pyo3::PyAny> {
        assert!(state < STATES);
        // Matrix
        // Transition matrix
        let changed_lambda_matrix = {
            let transition_array = self.transition_matrix(action_profile)?;
            let rewards = self.rewards(action_profile)?;
            let locals = pyo3::types::PyDict::new(py);
            let code = format!(
                r#"
import sympy
lam = sympy.Symbol("lam")
lambda_matrix = sympy.eye({}) - (1 - lam) * sympy.Matrix({:?})
rewards = lam * sympy.Matrix({:?})
lambda_matrix.col_del({})
ret = lambda_matrix.col_insert({}, rewards)
            "#,
                STATES, transition_array, rewards, state, state,
            );
            py.run(&code, None, Some(locals))?;
            locals
                .get_item("ret")
                .expect("Could not transform into sympy Matrix")
        };

        // Computation
        let sympy = pyo3::prelude::PyModule::import(py, "sympy")?;
        let result = sympy.call1("det", (changed_lambda_matrix,))?;
        Ok(result)
    }

    /// Symbolic matrix `Id - (1 - \lambda) Q`
    fn sym_lambda_matrix<'a>(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
        py: pyo3::Python<'a>,
    ) -> anyhow::Result<&'a pyo3::PyAny> {
        let sympy = pyo3::prelude::PyModule::import(py, "sympy")?;
        // Transition matrix
        let q_matrix = {
            let transition_array = self.transition_matrix(action_profile)?;
            let locals = pyo3::types::PyDict::new(py);
            let code = format!(
                r#"
import sympy
ret = sympy.Matrix({:?})
            "#,
                transition_array
            );
            py.run(&code, None, Some(locals))?;
            locals
                .get_item("ret")
                .expect("Could not transform into sympy Matrix")
        };
        // Lambda
        let lam_symbol = sympy.call1("Symbol", ("lam",))?;
        // Identity
        let identity = {
            let locals = pyo3::types::PyDict::new(py);
            let code = format!(
                r#"
import sympy
ret = sympy.eye({:?})
            "#,
                STATES
            );
            py.run(&code, None, Some(locals))?;
            locals
                .get_item("ret")
                .expect("Could not transform into sympy Matrix")
        };
        // Dictionary
        let sympy_dict = pyo3::types::IntoPyDict::into_py_dict(
            vec![
                ("q_matrix", q_matrix),
                ("lam", lam_symbol),
                ("identity", identity),
            ],
            py,
        );
        // Computation
        let result = py.eval("identity - (1 - lam) * q_matrix", None, Some(sympy_dict))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // use ndarray::array;
    // use test_case::test_case;

    #[test]
    fn construction() {
        // Parameters
        let actions_one = [1, 1];
        let actions_two = [1, 1];
        let mut transition = HashMap::new();
        transition.insert((0, (0, 0)), [1., 0.]);
        transition.insert((1, (0, 0)), [0., 1.]);
        let mut reward = HashMap::new();
        reward.insert((0, (0, 0)), 0.);
        reward.insert((1, (0, 0)), 1.);
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

        assert_eq!(stoc_game.rewards(([0, 0], [0, 0])).unwrap(), [&0., &1.]);
        assert_eq!(
            stoc_game.transition_matrix(([0, 0], [0, 0])).unwrap(),
            [&[1., 0.], &[0., 1.]]
        );
        assert!(stoc_game.has_action_profile(([0, 0], [0, 0])));
        assert!(!stoc_game.has_action_profile(([0, 0], [0, 1])));
    }

    #[test]
    fn approx_value_fixed() {
        // Parameters
        let actions_one = [1];
        let actions_two = [1];
        let mut transition = HashMap::new();
        transition.insert((0, (0, 0)), [1.]);
        let mut reward = HashMap::new();
        reward.insert((0, (0, 0)), 0.5);
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

        let approx_value = stoc_game.approx_value(0, 1e-7).unwrap();
        let expected = 0.5;
        assert_eq!(approx_value, expected);
    }

    #[test]
    fn approx_value() {
        // Parameters
        let actions_one = [1, 1];
        let actions_two = [1, 1];
        let mut transition = HashMap::new();
        transition.insert((0, (0, 0)), [1., 0.]);
        transition.insert((1, (0, 0)), [0., 1.]);
        let mut reward = HashMap::new();
        reward.insert((0, (0, 0)), 0.);
        reward.insert((1, (0, 0)), 1.);
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

        assert_abs_diff_eq!(stoc_game.approx_value(0, 1e-7).unwrap(), 0., epsilon = 1e-7);
        assert_abs_diff_eq!(stoc_game.approx_value(1, 1e-7).unwrap(), 1., epsilon = 1e-7);
    }

    #[test]
    fn symbolic_computations() {
        // Parameters
        let actions_one = [1, 1];
        let actions_two = [1, 1];
        let mut transition = HashMap::new();
        transition.insert((0, (0, 0)), [1., 0.]);
        transition.insert((1, (0, 0)), [0., 1.]);
        let mut reward = HashMap::new();
        reward.insert((0, (0, 0)), 0.5);
        reward.insert((1, (0, 0)), 1.);
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

        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();

        // lambda matrix
        let result = stoc_game
            .sym_lambda_matrix(([0, 0], [0, 0]), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([\n[1.0*lam,       0],\n[      0, 1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        // Determinant
        let result = stoc_game
            .sym_null_determinant(([0, 0], [0, 0]), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        // State determinant
        let result = stoc_game
            .sym_state_determinant(0, ([0, 0], [0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0.5*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        // Auxiliary matrix
        let result = stoc_game
            .sym_aux_matrix_game(0, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"0.5*lam**2 - z (1.0*lam**2)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
    }

    #[test]
    fn symbolic_computations_2() {
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

        // Basic checks
        assert_eq!(
            stoc_game
                .transition_matrix(([0, 0, 0, 0], [0, 0, 0, 0]))
                .unwrap(),
            [
                &[1., 0., 0., 0.],
                &[0., 0., 0., 1.],
                &[0., 0., 0., 1.],
                &[0., 0., 0., 1.],
            ]
        );
        assert_eq!(
            stoc_game
                .transition_matrix(([0, 0, 1, 0], [0, 0, 0, 0]))
                .unwrap(),
            [
                &[1., 0., 0., 0.],
                &[0., 0., 0., 1.],
                &[0.5, 0.5, 0., 0.],
                &[0., 0., 0., 1.],
            ]
        );
        assert_eq!(
            stoc_game
                .transition_matrix(([0, 1, 0, 0], [0, 0, 0, 0]))
                .unwrap(),
            [
                &[1., 0., 0., 0.],
                &[0., 0., 1., 0.],
                &[0., 0., 0., 1.],
                &[0., 0., 0., 1.],
            ]
        );
        assert_eq!(
            stoc_game
                .transition_matrix(([0, 1, 1, 0], [0, 0, 0, 0]))
                .unwrap(),
            [
                &[1., 0., 0., 0.],
                &[0., 0., 1., 0.],
                &[0.5, 0.5, 0., 0.],
                &[0., 0., 0., 1.],
            ]
        );

        // Symbolic
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();

        // lambda matrix
        let result = stoc_game
            .sym_lambda_matrix(([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([
[1.0*lam, 0, 0,             0],
[      0, 1, 0, 1.0*lam - 1.0],
[      0, 0, 1, 1.0*lam - 1.0],
[      0, 0, 0,       1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_lambda_matrix(([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([
[      1.0*lam,             0, 0,             0],
[            0,             1, 0, 1.0*lam - 1.0],
[0.5*lam - 0.5, 0.5*lam - 0.5, 1,             0],
[            0,             0, 0,       1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_lambda_matrix(([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([
[1.0*lam, 0,             0,             0],
[      0, 1, 1.0*lam - 1.0,             0],
[      0, 0,             1, 1.0*lam - 1.0],
[      0, 0,             0,       1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_lambda_matrix(([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([
[      1.0*lam,             0,             0,       0],
[            0,             1, 1.0*lam - 1.0,       0],
[0.5*lam - 0.5, 0.5*lam - 0.5,             1,       0],
[            0,             0,             0, 1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        // Determinant
        let result = stoc_game
            .sym_null_determinant(([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_null_determinant(([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_null_determinant(([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        let result = stoc_game
            .sym_null_determinant(([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        // State determinant
        // State 0
        let state = 0;
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);

        // State 1
        let state = 1;
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "-1.0*lam**3 + 1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "-1.0*lam**3 + 1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "1.0*lam**4 - 2.0*lam**3 + 1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);

        // State 2
        let state = 2;
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "-1.0*lam**3 + 1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0.5*lam**4 - 1.0*lam**3 + 0.5*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "-1.0*lam**3 + 1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "0";
        assert_eq!(&format!("{:?}", result), expected);

        // State 3
        let state = 3;
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 0, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 0, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
        let result = stoc_game
            .sym_state_determinant(state, ([0, 1, 1, 0], [0, 0, 0, 0]), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2";
        assert_eq!(&format!("{:?}", result), expected);

        // Auxiliary matrix
        let state = 0;
        let result = stoc_game
            .sym_aux_matrix_game(state, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"0 - z (1.0*lam**2)\"], [\"0 - z (1.0*lam**2)\"], [\"0 - z (1.0*lam**2)\"], [\"0 - z (-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
        let state = 1;
        let result = stoc_game
            .sym_aux_matrix_game(state, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"-1.0*lam**3 + 1.0*lam**2 - z (1.0*lam**2)\"], [\"-1.0*lam**3 + 1.0*lam**2 - z (1.0*lam**2)\"], [\"1.0*lam**4 - 2.0*lam**3 + 1.0*lam**2 - z (1.0*lam**2)\"], [\"0 - z (-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
        let state = 2;
        let result = stoc_game
            .sym_aux_matrix_game(state, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"-1.0*lam**3 + 1.0*lam**2 - z (1.0*lam**2)\"], [\"0.5*lam**4 - 1.0*lam**3 + 0.5*lam**2 - z (1.0*lam**2)\"], [\"-1.0*lam**3 + 1.0*lam**2 - z (1.0*lam**2)\"], [\"0 - z (-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
        let state = 3;
        let result = stoc_game
            .sym_aux_matrix_game(state, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"1.0*lam**2 - z (1.0*lam**2)\"], [\"1.0*lam**2 - z (1.0*lam**2)\"], [\"1.0*lam**2 - z (1.0*lam**2)\"], [\"-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2 - z (-0.5*lam**4 + 1.0*lam**3 + 0.5*lam**2)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
    }
}

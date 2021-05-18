use crate::MatrixGame;
use std::collections::HashMap;

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

/// Iterator over all (pure stationary) action profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionProfiles<const STATES: usize> {
    actions_one: [usize; STATES],
    actions_two: [usize; STATES],
    next: Option<([usize; STATES], [usize; STATES])>,
}

impl<const STATES: usize> ActionProfiles<STATES> {
    /// Creates a new `ActionProfiles` for the specified number of actions
    ///  in each state for both players.
    pub fn new(actions_one: [usize; STATES], actions_two: [usize; STATES]) -> Self {
        let next = if actions_one.iter().all(|&i| i > 0) && actions_two.iter().all(|&i| i > 0) {
            Some(([1; STATES], [1; STATES]))
        } else {
            None
        };
        ActionProfiles {
            actions_one,
            actions_two,
            next,
        }
    }
}

impl<const STATES: usize> Iterator for ActionProfiles<STATES> {
    type Item = ([usize; STATES], [usize; STATES]);
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.clone();
        if self.next.is_some() {
            // Player two
            for j in 1..=STATES {
                if self.next.unwrap().1[STATES - j] == self.actions_two[STATES - j] {
                    self.next = self.next.map(|mut v| {
                        v.1[STATES - j] = 1;
                        v
                    });
                } else {
                    self.next = self.next.map(|mut v| {
                        v.1[STATES - j] += 1;
                        v
                    });
                    return next;
                }
            }
            // Player one
            for i in 1..=STATES {
                if self.next.unwrap().0[STATES - i] == self.actions_one[STATES - i] {
                    self.next = self.next.map(|mut v| {
                        v.0[STATES - i] = 1;
                        v
                    });
                } else {
                    self.next = self.next.map(|mut v| {
                        v.0[STATES - i] += 1;
                        v
                    });
                    return next;
                }
            }
        }
        self.next = None;
        return next;
    }
}

/// Follows [[Shapley53]].
///
/// [Shapley53]: https://doi.org/10.1073/pnas.39.10.1095
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticGame<const STATES: usize> {
    actions_one: [usize; STATES], // Actions available per state
    actions_two: [usize; STATES], // Actions available per state
    transition: HashMap<([usize; STATES], [usize; STATES]), [[f64; STATES]; STATES]>, // Transition matrix
    reward: HashMap<([usize; STATES], [usize; STATES]), [f64; STATES]>,               // Rewards
    current_state: usize,
    objective: Objective,
}

impl<const STATES: usize> StochasticGame<STATES> {
    /// Creates a new `StochasticGame`.
    pub fn new(
        actions_one: [usize; STATES],
        actions_two: [usize; STATES],
        transition: impl Fn([usize; STATES], [usize; STATES]) -> [[f64; STATES]; STATES], // Transition matrix
        reward: impl Fn([usize; STATES], [usize; STATES]) -> [f64; STATES],
        initial_state: usize,
        objective: Objective,
    ) -> Self {
        let transition = ActionProfiles::new(actions_one, actions_two)
            .map(|x| (x, transition(x.0, x.1)))
            .collect();
        let reward = ActionProfiles::new(actions_one, actions_two)
            .map(|x| (x.clone(), reward(x.0, x.1)))
            .collect();

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
    ) -> anyhow::Result<&[f64; STATES]> {
        self.reward
            .get(&action_profile)
            .ok_or(anyhow::anyhow!("Invalid action profile"))
    }
    /// Returns the rewards at the given state, for the given local actions.
    pub fn reward_at_state(
        &self,
        state: usize,
        local_actions: (usize, usize),
    ) -> anyhow::Result<f64> {
        todo!("")
        // self.reward
        //     .get(&action_profile)
        //     .ok_or(anyhow::anyhow!("Invalid action profile"))
        //     .map(|v| v[state])
    }
    /// Returns the transition between states, for the given action profile.
    pub fn transition_matrix(
        &self,
        action_profile: ([usize; STATES], [usize; STATES]),
    ) -> anyhow::Result<&[[f64; STATES]; STATES]> {
        self.transition
            .get(&action_profile)
            .ok_or(anyhow::anyhow!("Invalid action profile"))
    }
    /// Returns an iterator over all action profiles.
    pub fn action_profiles(&self) -> ActionProfiles<STATES> {
        ActionProfiles::new(self.actions_one, self.actions_two)
    }
    /// Returns `true` if the action profile is allowed in the game.
    pub fn has_action_profile(&self, action_profile: ([usize; STATES], [usize; STATES])) -> bool {
        self.reward.keys().any(|&i| i == action_profile)
    }
    // pub fn fix_stationary_strategy(&self, stationary_strategy: [Vec<f64>; STATES]) -> MarkovDecissionProcess {}
}

/// Numerical implementation
impl<const STATES: usize> StochasticGame<STATES>
where
    nalgebra::Const<STATES>:
        nalgebra::DimMin<nalgebra::Const<STATES>, Output = nalgebra::Const<STATES>>,
{
    pub fn approx_value(&self, state: usize, error: f64) -> f64 {
        assert!(error > 0.);
        use itertools::Itertools;
        let (mut lower_bound, mut upper_bound): (f64, f64) = self
            .action_profiles()
            .map(|x| self.rewards(x).unwrap())
            .flatten()
            .cloned()
            .minmax()
            .into_option()
            .unwrap();
        while (upper_bound - lower_bound).abs() > error {
            let z = (lower_bound + upper_bound) / 2.;
            let value_z = self.aux_matrix_game(state, z).unwrap().value();
            if value_z > 0. {
                lower_bound = z;
            } else {
                upper_bound = z;
            }
        }
        (lower_bound + upper_bound) / 2.
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
        // ndarray::Array2::<f64>::zeros((STATES, STATES));
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
                    new_matrix[(k1, k2)] = rewards[k1];
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
    pub fn sym_aux_matrix_game<'a>(&self, state: usize, py: pyo3::Python<'a>) -> anyhow::Result<Vec<Vec<String>>> {
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
            aux_matrix[j][i] = format!("{} - z ({})", d0, dk);
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
rewards = sympy.Matrix({:?})
lambda_matrix.col_del({})
ret = lambda_matrix.col_insert({}, rewards)
            "#,
                STATES,
                transition_array,
                rewards,
                state,
                state,
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

        assert_eq!(stoc_game.rewards(([1, 1], [1, 1])).unwrap(), &[0., 1.]);
        assert_eq!(stoc_game.transition_matrix(([1, 1], [1, 1])).unwrap(), &[[1., 0.], [0., 1.]]);
        assert!(stoc_game.has_action_profile(([1, 1], [1, 1])));
        
    }

    #[test]
    fn approx_value_fixed() {
        // Parameters
        let actions_one = [1];
        let actions_two = [1];
        let mut transition = HashMap::new();
        transition.insert(([1], [1]), [[1.]]);
        let mut reward = HashMap::new();
        reward.insert(([1], [1]), [0.5]);
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

        let approx_value = stoc_game.approx_value(0, 1e-7);
        let expected = 0.5;
        assert_eq!(approx_value, expected);
    }

    #[test]
    fn approx_value() {
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

        assert_abs_diff_eq!(stoc_game.approx_value(0, 1e-7), 0., epsilon = 1e-7);
        assert_abs_diff_eq!(stoc_game.approx_value(1, 1e-7), 1., epsilon = 1e-7);
    }

    #[test]
    fn symbolic_computations() {
        // Parameters
        let actions_one = [1, 1];
        let actions_two = [1, 1];
        let transition_matrix = [[1., 0.], [0., 1.]];
        let mut transition = HashMap::new();
        transition.insert(([1, 1], [1, 1]), transition_matrix);
        let mut reward = HashMap::new();
        reward.insert(([1, 1], [1, 1]), [1., 1.]);
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

        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();

        // lambda matrix
        let result = stoc_game
            .sym_lambda_matrix((actions_one, actions_two), py)
            .expect("Could not compute the symbolic matrix Id - (1 - lambda) Q");
        let expected = "Matrix([\n[1.0*lam,       0],\n[      0, 1.0*lam]])";
        assert_eq!(&format!("{:?}", result), expected);

        // Determinant
        let result = stoc_game
            .sym_null_determinant((actions_one, actions_two), py)
            .expect("Could not compute the determinant of the symbolic matrix!");
        let expected = "1.0*lam**2";
        assert_eq!(&format!("{:?}", result), expected);
    
        // State determinant
        let result = stoc_game
            .sym_state_determinant(0, (actions_one, actions_two), py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "1.0*lam";
        assert_eq!(&format!("{:?}", result), expected);

        // Auxiliary matrix
        let result = stoc_game
            .sym_aux_matrix_game(0, py)
            .expect("Could not compute the state determinant of the symbolic matrix!");
        let expected = "[[\"1.0*lam**2 - z (1.0*lam)\"]]";
        assert_eq!(&format!("{:?}", result), expected);
    }
}

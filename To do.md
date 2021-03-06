# Computing tools

## LP solver

Change to certifying LP solvers!

- [A review of computation of mathematically rigorous bounds on optima of linear programs](https://link.springer.com/article/10.1007/s10898-016-0489-2)
- [Certifying feasibility and objective value of linear programs](https://www.sciencedirect.com/science/article/pii/S0167637712000272)
  - [Complete thesis](https://domino.mpi-inf.mpg.de/imprs/imprspubl.nsf/0/B1A302821896EA31C1257E4C002468F6/$file/dumitriu_phdthesis.pdf)
- Karmarkar algorithm
  - Exact for rational inputs (approximate for real data)
  - First efficient polynomial time algorithm for LP
  - Interior point method



# Traits

## Game forms
### Strategic or Normal
Static
``` rust
/// [Strategic] or [Normal] form games 
///
/// [Strategic]: https://en.wikipedia.org/w/index.php?title=Strategic_form&redirect=no
/// [Normal]: https://en.wikipedia.org/wiki/Normal-form_game
trait Strategic<Utility: Num> // We want to allow different implementation for different numerical types. 
// Would one need to annotate which utility type is using if there were many implamantations??
const N: usize; // num_players
type Action = usize;
type ActionSet = Vec<Action>;

// Another option is to simply ask for the payoff matrix!
fn payoff_matrix(&self) -> ArrayBase<[Utility; N]>;

fn strategies(&self) -> [ActionSet; N];
fn payoffs(&self, strategy_profile: [Action; N]) -> Utility;

fn action_set(&self, player: usize) -> ActionSet {
  self.strategies[player]
}
fn num_players(&self) -> usize {
  N
}
fn min_social_payoff(&self) -> Utility {
  todo!()
}
fn player_payoffs(&self) -> Iterator<Item = Utility> {
  todo!()
}
fn social_payoffs(&self) -> Iterator<Item = Utility> {
  todo!()
}
```
Mutable
``` rust
trait StrategicMut<Utility>: Strategic<Utility>
fn remove_action(&mut self, player: usize, action: usize) -> Self;
```

Should there be a representation of a Strategic game from a `Array<[Utility; N]>`?

### Extensive

## Solutions
### Equilibria

A number of desirable methods is given in [[GZ89]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.588.7844&rep=rep1&type=pdf), including

- max_payoff
- uniqueness
- is_subset_of
- contains_the_subset
- support_at_least
- support_at_most

### Strategies
- is_pure
- is_mixed
- sample
- support
## Playable
It should create a yew app to play the game :) 

# Equilibria

## Pure Nash

## (Mixed) Nash

``` rust
trait Nash<Utility>
const N: usize; // num_players
type Action;
type Solution = ([Action; N], [Utility; N]);

fn all_solutions(&self) -> impl Iterator<Item=Solution>;
fn is_equilibrium(&self, strategy_profile: [Action; N]) -> bool;
fn one_solution(&self) -> Option<Solution> {
    self.all_solutions().next()
}
fn one_solution_for_player(&self, player: usize) -> Option<(Action, Utility)> {
    self.one_solution().map(|(strategy_profile, utilities)| (strategy_profile[player], utilities[player]))
}
fn has_equilibrium(&self) -> maybe_bool {
    self.one_solution().is_some()
}
```

[Chapter 14, Section 7, Handbook of Game Theory, Vol. 4 (2015)]

## Correlated

## Evolutionary

## Policy

## Algorithms

- [Homotopy methods to compute equilibria in game theory](https://link.springer.com/article/10.1007/s00199-009-0441-5)
- Lemke-Howson algorithm
  - [Paper](https://web.stanford.edu/~saberi/lecture4.pdf)
  - [Video](https://www.youtube.com/watch?v=-OnHWm_Wycw)
  - its worst running time is exponential in the number of pure strategies of the players (Savani and von Stengel 2004).  
- [Solving systems of polynomial equations](https://math.berkeley.edu/~bernd/cbms.pdf) (See Chapter 6)
  - [PHCpack: a general-purpose solver for polynomial systems by homotopy continuation](http://homepages.math.uic.edu/~jan/)


# Games
## Bimatrix (nonzero-sum two player games)

- Implement
  - one_solution()
    - Lemke-Howson?

## Graphical games

References

- [Graphical Models for Game Theory](https://arxiv.org/abs/1301.2281)
- [Multi-Agent Influence Diagrams for Representing and Solving Games](http://people.csail.mit.edu/milch/papers/geb-maid.pdf)

## Other tractable Nash equilibria cases

See Algorithmic Game Theory, chapter 7, pages 159–18.

## Stochastic game

```rust
struct StochasticGame {
    states: [Game; N],
    transition: Fn: (game, action_profile) -> [Probability; N],
    current: usize,
}
```

## Vectorial payoffs

Approachable and excludable sets computation? 

- An analog of the minimax theorem for vector payoffs. Blackwell (1956)
  Presentation of matrix games with vector payoffs.
- Approachable sets of vector payoffs in stochastic games. Milman (2006)
  Online learning and Blackwell approachability in quitting games. Flesch et al. (2016) 
  Partial results.

# Certifying algorithms

- PolyMatrixGame
  - [x] is_value_positive
  - is_uniform_value_positive
    - Documentation
    - Check out is_value_positive and treat each case
  - functional_form
    The kernel is still missing, see MatrixGame
    - [x] value
    - [ ] strategy
    - [ ] Documentation
- MatrixGame methods
  - is_completely_mixed
    - Can you certify it?
  - kernel_completely_mixed
    - Does it give a correct kernel? Test it!
  - solve
    - Improve documentation
    - from (certified) value?
  - solve_row
  - solve_column?
  - value
    - value_uncertified?

# LPs with errors

- Reduction to PolyMatrixGame
- is_weakly_robust
- is_strongly_robust
- functional_form

# Polynomial Matrix Games

- [x] functional_form_value() -> Ratio<Polynomial<i32>>
- functional_form_strategy_row() -> Vec<Ratio<Polynomial<i32>>>
- Given the kernel, use the formula for completely mixed games
- functional_form() -> (Ratio<Polynomial<i32>>, Ratio<Polynomial<i32>>)
  - Need a Kernel that maintain optimal strategies!!
    
    - Kaplansky 1945 (doi: 10.2307/1969164) gives only the value
- is_uniform_value_positive
  - Add a preliminary test (some “interesting” sufficient conditions) 


# POMDPs

- Implement from the basics
- play()

# Strategies

## Regret

### External regret

#### Cost for all actions

We assume we have access to all costs of the possible actions (not only the one taken)

- Exponential weights for one player

  ```rust
struct ExponentialWeights {
      counter: usize,
      weights: [f64; N],
      dist: rand_distr::WeightedIndex,
      rng: R,
  }
  impl ExponentialWeightsTrait {
      fn losts(&self, time: usize) -> [f64; N] {
          assert!(0 <= losts <= 1);
      }
  }
  impl Iterator for ExponentialWeights {
      type Item = Action
      fn next(&mut self) -> Option<Action> {
          let next_action = self.dist.sample(&mut self.rng);
          self.counter += 1;
          let losts = self.losts(self.counter);
          for i in 0..N {
              self.weights[i] *= (1. - self.eps).powf(losts[i]);
          }
          let update_weights = (0..N).filter(|i| costs[i]).map(|i| (i, self.weights[i])).collect();
          self.dist.update_weights(update_weights)
          	.expect("There was a problem when updating weights!");
          Some(new_action)
      }
  }
  ```
  
  
  
  - Binary costs?
  - define $\varepsilon \in (0, 1)$ as input?
  - Weights to update and sample **once**
    - [rand_distr](https://docs.rs/rand_distr/0.4.0/rand_distr/index.html)::[WeightedIndex](https://docs.rs/rand_distr/0.4.0/rand_distr/struct.WeightedIndex.html) 
    - `update_weights` 
  - action -> mut weight

#### Cost for action taken

We assume we have access only to the cost of the action taken

- Bandits
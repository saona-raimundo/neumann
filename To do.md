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
### Equilibriums
Enum
- Nash
- Correlated
- Evolutionary
- Policy
### Strategies
Enum
- Pure
- Mixed
### Nash
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

## Playable
It should create a yew app to play the game :) 


# Games
## Bimatrix

- Implement
  - one_solution()
    - Lemke-Howson algorithm
    - [Paper](https://web.stanford.edu/~saberi/lecture4.pdf)
    - [Video](https://www.youtube.com/watch?v=-OnHWm_Wycw)

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


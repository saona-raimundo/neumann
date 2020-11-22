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


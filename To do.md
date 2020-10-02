# Matrix Games

- [x] is_square() -> bool
- [x] best_row_reduction -> usize
  - [x] is_completely_mixed() -> bool
  - [x] best_column_reduction -> usize
    - [x] reduce_to_square(&mut self)
    - [x] completely_mixed_kernel() -> (Vec<usize>, Vec<usize>)
      Maybe use std::collections::HashSet for the iterative procedure of reduction

# Polynomial Matrix Games

- functional_form() -> ?? (Fn? (Vec, Vec)? (Poly, Poly)? Rational_function?)
  - Need a Kernel!!
    
    - Look at Kaplansky 1945 (doi: 10.2307/1969164)
  - Helper crates

    - [polynomials](https://crates.io/crates/polynomials)
      - Suggestions: 

        - Implement traits

          - std::fmt::Display


# POMDPs

- Implement from the basics
- play()

# Next release

## Documentation

- Add reference for kernel_completely_mixed


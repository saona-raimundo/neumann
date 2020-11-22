
pub use traits::*;

mod traits {

	use crate::Certified;

	/// Stability concept for polynomially perturbed games[^1].
	///
	/// [^1] To appear.
	pub trait ValuePositivity<C1, C2, C3> {

		/// Returns a bound on the size of perturbation that gives an explicit right neighborhood of zero. 
		fn epsilon_kernel_constant(&self) -> f64;

	    /// Checks if the perturbed matrix game has at least the value of the unperturbed game at a right neigborhood of zero.
		fn is_value_positive(&self) -> Certified<bool, C1>;

		/// Checks the answer.
		fn is_value_positive_checker(&self, certified_output: Certified<bool, C1>) -> bool;

		/// Checks if the perturbed matrix game has at least the value of the unperturbed game at a right neigborhood of zero.
		fn is_value_positive_uncertified(&self) -> bool {
			self.is_value_positive().output
		}

		/// Checks if the main player of a game has a fixed strategy that ensures at least the value of the error-free game at a right neigborhood of zero.
	    ///
	    /// That is, there exists a fixed strategy and a positive threshold for which the reward given by this strategy
	    /// in the perturbed game is at least as much as
	    /// the value of the game with no perturbations.
		fn is_uniform_value_positive(&self) -> Certified<bool, C2>;

		/// Checks the answer.
		fn is_uniform_value_positive_checker(&self, certified_output: Certified<bool, C2>) -> bool;

		/// Checks if the main player of a game has a fixed strategy that ensures at least the value of the error-free game at a right neigborhood of zero.
		fn is_uniform_value_positive_uncertified(&self) -> bool {
			self.is_uniform_value_positive().output
		}

		/// Returns the value function at a right neigborhood of zero.
		fn functional_form(&self) -> Certified<num_rational::Ratio<polynomials::Polynomial<i32>>, C3>;

		/// Returns the value function at a right neigborhood of zero.
		fn functional_form_checker(&self, certified_output: Certified<num_rational::Ratio<polynomials::Polynomial<i32>>, C3>) -> bool;

		fn functional_form_uncertified(&self) -> num_rational::Ratio<polynomials::Polynomial<i32>> {
			self.functional_form().output
		}


	}
}
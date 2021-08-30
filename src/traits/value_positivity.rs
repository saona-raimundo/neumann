/// Stability concept for polynomially perturbed games[^1].
///
/// [^1] To appear.
pub trait ValuePositivity {
    /// Type representation of perturbation.
    type Perturbation;
    /// Certificate of value positivity.
    type ValuePositivityCertificate;
    /// Certificate of uniform value positivity.
    type UniformValuePositivityCertificate;
    /// Certificate for functional form.
    type FunctionalFormCertificate;
    /// Value of the game at a given perturbation.
    type Value;
    /// Function representation.
    ///
    /// It is expected to implement some sort of `Fn(Self::Perturbation) -> Self::Value`.
    type FunctionalForm;

    /// Returns a value for the error term `epsilon` such that
    /// kernels of optimal strategies are guaranteed not to change between this value and zero.
    fn epsilon_kernel_constant(&self) -> Self::Perturbation;

    /// Returns `true` if the perturbed game has at least the value of the error-free game
    /// at a right neigborhood of zero.
    fn is_value_positive(&self) -> bool {
        self.is_value_positive_checker(self.is_value_positive_certifying())
    }

    /// Returns `true` if the perturbed game has at least the value of the error-free game
    /// at a right neigborhood of zero, and gives a certificate of this answer.
    fn is_value_positive_certifying(&self) -> (bool, Self::ValuePositivityCertificate);

    /// Returns `true` if `certifying_output` is correct.
    fn is_value_positive_checker(
        &self,
        certifying_output: (bool, Self::ValuePositivityCertificate),
    ) -> bool;

    /// Returns `true` if the perturbed game has a fixed strategy that gives
    /// at least the value of the error-free game at a right neigborhood of zero.
    fn is_uniform_value_positive(&self) -> bool {
        self.is_uniform_value_positive_checker(self.is_uniform_value_positive_certifying())
    }

    /// Returns `true` if the perturbed game has a fixed strategy that gives
    /// at least the value of the error-free game at a right neigborhood of zero,
    /// and gives a certificate of this answer.
    fn is_uniform_value_positive_certifying(
        &self,
    ) -> (bool, Self::UniformValuePositivityCertificate);

    /// Returns `true` if `certifying_output` is correct.
    fn is_uniform_value_positive_checker(
        &self,
        certifying_output: (bool, Self::UniformValuePositivityCertificate),
    ) -> bool;

    /// Returns the value function at a right neigborhood of zero.
    fn functional_form(&self) -> Self::FunctionalForm {
        self.functional_form_certifying().0
    }

    /// Returns the value function at a right neigborhood of zero,
    /// and gives a certificate of this answer.
    fn functional_form_certifying(&self)
        -> (Self::FunctionalForm, Self::FunctionalFormCertificate);

    /// Returns `true` if `certifying_output` is correct.
    fn functional_form_checker(
        &self,
        certifying_output: (Self::FunctionalForm, Self::FunctionalFormCertificate),
    ) -> bool;
}

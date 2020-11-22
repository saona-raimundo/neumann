/// Certified answers.
///
/// Certifying algorithms[^1] produce, with each output, a certificate or witness (easy-to-verify proof)
/// that the particular output has not been compromised by a bug. Therefore, one does not need to depend 
/// on the correctness of the algorithm, by verifying peer-instance answers.
///
/// [^1]: McConnell, R.M.; Mehlhorn, K.; Näher, S.; Schweitzer, P. (May 2011), 
/// *"Certifying algorithms"*, Computer Science Review, 5 (2): 119–161, 
/// doi: [10.1016/j.cosrev.2010.09.009](https://doi.org/10.1016%2Fj.cosrev.2010.09.009)
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct Certified<T, C> {
	/// Otput or answer.
	pub output: T,
	/// Certificate or witness.
	pub certificate: C,
}


impl<T, C> From<(T, C)> for Certified<T, C> {
    /// Copies `T` as output and `C` as certificate.
    ///
    /// # Examples
    ///
    /// ```
    /// let o: Option<u8> = Option::from(67);
    ///
    /// assert_eq!(Some(67), o);
    /// ```
    fn from(certified_output: (T, C)) -> Certified<T, C> {
        Certified{
        	output: certified_output.0, 
        	certificate: certified_output.1
        }
    }
}

// /// Certified boolean answer with different certificate types.
// #[derive(Debug)]
// pub enum CertifiedBool<C1, C2> {
//     True(C1),
//     False(C2),
// }
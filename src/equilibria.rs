use crate::traits::Equilibrium;
use core::marker::PhantomData;

/// Pure (or deterministic) Nash equilibrium.
#[derive(Debug)]
pub struct PureNash;
impl Equilibrium for PureNash {}

/// Mixed Nash equilibrium.
#[derive(Debug)]
pub struct MixedNash<T> {
    _phantom_data: PhantomData<T>,
}
impl<T> Equilibrium for MixedNash<T> {}

// /// Correlated equilibrium.
// #[derive(Debug)]
// pub struct Correlated;
// impl Equilibrium for Correlated {}

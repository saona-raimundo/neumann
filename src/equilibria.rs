use crate::traits::Equilibrium;
use core::marker::PhantomData;

#[derive(Debug)]
pub struct PureNash;
impl Equilibrium for PureNash {}

#[derive(Debug)]
pub struct MixedNash<T> {
    _phantom_data: PhantomData<T>,
}
impl<T> Equilibrium for MixedNash<T> {}

#[derive(Debug)]
pub struct Correlated;
impl Equilibrium for Correlated {}

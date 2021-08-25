use crate::traits::Equilibrium;

#[derive(Debug)]
pub struct PureNash;

impl Equilibrium for PureNash {}

#[derive(Debug)]
pub struct MixedNash;

impl Equilibrium for MixedNash {}

#[derive(Debug)]
pub struct Correlated;

impl Equilibrium for Correlated {}

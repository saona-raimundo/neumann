/// Iterator over all (pure stationary) action profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionProfiles<const STATES: usize> {
    actions_one: [usize; STATES],
    actions_two: [usize; STATES],
    next: Option<([usize; STATES], [usize; STATES])>,
}

impl<const STATES: usize> ActionProfiles<STATES> {
    /// Creates a new `ActionProfiles` for the specified number of actions
    ///  in each state for both players.
    pub fn new(actions_one: [usize; STATES], actions_two: [usize; STATES]) -> Self {
        let next = if actions_one.iter().all(|&i| i > 0) && actions_two.iter().all(|&i| i > 0) {
            Some(([0; STATES], [0; STATES]))
        } else {
            None
        };
        ActionProfiles {
            actions_one,
            actions_two,
            next,
        }
    }
}

impl<const STATES: usize> Iterator for ActionProfiles<STATES> {
    type Item = ([usize; STATES], [usize; STATES]);
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.clone();
        if self.next.is_some() {
            // Player two
            for j in 1..=STATES {
                if self.next.unwrap().1[STATES - j] == self.actions_two[STATES - j] - 1 {
                    self.next = self.next.map(|mut v| {
                        v.1[STATES - j] = 0;
                        v
                    });
                } else {
                    self.next = self.next.map(|mut v| {
                        v.1[STATES - j] += 1;
                        v
                    });
                    return next;
                }
            }
            // Player one
            for i in 1..=STATES {
                if self.next.unwrap().0[STATES - i] == self.actions_one[STATES - i] - 1 {
                    self.next = self.next.map(|mut v| {
                        v.0[STATES - i] = 0;
                        v
                    });
                } else {
                    self.next = self.next.map(|mut v| {
                        v.0[STATES - i] += 1;
                        v
                    });
                    return next;
                }
            }
        }
        self.next = None;
        return next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let actions_one = [1, 2];
        let actions_two = [2, 1];
        let mut action_profiles = ActionProfiles::new(actions_one, actions_two);

        assert_eq!(action_profiles.next(), Some(([0, 0], [0, 0])));
        assert_eq!(action_profiles.next(), Some(([0, 0], [1, 0])));
        assert_eq!(action_profiles.next(), Some(([0, 1], [0, 0])));
        assert_eq!(action_profiles.next(), Some(([0, 1], [1, 0])));
        assert_eq!(action_profiles.next(), None);
    }
}

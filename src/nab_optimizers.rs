use std::ops::Add;

use crate::nab_array::NDArray;

pub trait Optimizer {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray);
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Option<NDArray>,  // First moment
    v: Option<NDArray>,  // Second moment
    t: usize,           // Timestep
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            t: 0,
        }
    }

    pub fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Optimizer for Adam {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray) {
        self.t += 1;

        // Initialize momentum and velocity if not exists
        if self.m.is_none() {
            self.m = Some(NDArray::zeros(weights.shape().to_vec()));
            self.v = Some(NDArray::zeros(weights.shape().to_vec()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment estimate
        *m = m.multiply_scalar(self.beta1)
            .add(&gradients.multiply_scalar(1.0 - self.beta1));

        // Update biased second raw moment estimate
        *v = v.multiply_scalar(self.beta2)
            .add(&gradients.multiply(&gradients).multiply_scalar(1.0 - self.beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = m.divide_scalar(1.0 - self.beta1.powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = v.divide_scalar(1.0 - self.beta2.powi(self.t as i32));

        // Update parameters
        let update = m_hat.divide(&v_hat.sqrt().add_scalar(self.epsilon))
            .multiply_scalar(self.learning_rate);
        
        *weights = weights.subtract(&update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::default();
        
        // Create test weights and gradients
        let mut weights = NDArray::from_matrix(vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ]);
        let gradients = NDArray::from_matrix(vec![
            vec![0.01, 0.02],
            vec![0.03, 0.04],
        ]);

        // Initial weights
        let initial_weights = weights.clone();

        // Update weights
        optimizer.update(&mut weights, &gradients);

        // Verify that weights have been updated
        assert!(weights.data() != initial_weights.data(), 
            "Weights should be updated after optimization step");

        // Verify shape is maintained
        assert_eq!(weights.shape(), initial_weights.shape(),
            "Weight shape should remain unchanged");
    }

    #[test]
    fn test_adam_multiple_updates() {
        let mut optimizer = Adam::default();
        let mut weights = NDArray::randn_2d(2, 2);
        let gradients = NDArray::randn_2d(2, 2);

        // Multiple updates should work without errors
        for _ in 0..5 {
            optimizer.update(&mut weights, &gradients);
        }

        assert_eq!(optimizer.t, 5, "Timestep should be incremented correctly");
    }
} 
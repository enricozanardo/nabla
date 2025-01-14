use crate::nab_array::NDArray;
use std::collections::HashMap;

pub trait Optimizer {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray);
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    moments: HashMap<String, (NDArray, NDArray)>,  // Store m and v for each layer
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            moments: HashMap::new(),
            t: 1,
        }
    }

    pub fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }

    fn get_or_init_moments(&mut self, layer_name: &str, shape: &[usize]) -> &mut (NDArray, NDArray) {
        if !self.moments.contains_key(layer_name) {
            let m = NDArray::zeros(shape.to_vec());
            let v = NDArray::zeros(shape.to_vec());
            self.moments.insert(layer_name.to_string(), (m, v));
        }
        self.moments.get_mut(layer_name).unwrap()
    }
}

impl Optimizer for Adam {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray) {
        let layer_name = format!("{:?}", weights.shape());
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let t = self.t;
        let learning_rate = self.learning_rate;
        
        let (m, v) = self.get_or_init_moments(&layer_name, weights.shape());
        
        *m = m.multiply_scalar(beta1)
            .add(&gradients.multiply_scalar(1.0 - beta1));
        
        *v = v.multiply_scalar(beta2)
            .add(&gradients.multiply(gradients).multiply_scalar(1.0 - beta2));
        
        let m_hat = m.multiply_scalar(1.0 / (1.0 - beta1.powi(t as i32)));
        let v_hat = v.multiply_scalar(1.0 / (1.0 - beta2.powi(t as i32)));
        
        let update = m_hat.divide(&v_hat.sqrt().add_scalar(epsilon))
            .multiply_scalar(learning_rate);
            
        *weights = weights.subtract(&update);
        
        self.t += 1;
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

        // We start at t=1 and increment 5 times, so we expect t=6
        assert_eq!(optimizer.t, 6, "Timestep should be incremented correctly");
    }
} 
use crate::nab_array::NDArray;

pub trait Optimizer {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray);
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: NDArray,  // First moment
    v: NDArray,  // Second moment
    t: usize,           // Timestep
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 1,
            m: NDArray::zeros(vec![1, 1]),  // Will be resized on first update
            v: NDArray::zeros(vec![1, 1]),  // Will be resized on first update
        }
    }

    pub fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }

    fn ensure_moment_shapes(&mut self, shape: &[usize]) {
        if self.m.shape() != shape {
            println!("Initializing moment vectors with shape: {:?}", shape);
            self.m = NDArray::zeros(shape.to_vec());
            self.v = NDArray::zeros(shape.to_vec());
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, weights: &mut NDArray, gradients: &NDArray) -> () {
        println!("Adam update:");
        println!("  Weights shape: {:?}", weights.shape());
        println!("  Gradients shape: {:?}", gradients.shape());
        
        // Ensure moment vectors have correct shape
        self.ensure_moment_shapes(weights.shape());

        // Update biased first moment estimate
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * gradients;

        // Update biased second raw moment estimate
        let squared_gradients = gradients.multiply(gradients);
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * &squared_gradients;

        // Compute bias-corrected first moment estimate
        let m_hat = self.m.multiply_scalar(1.0 / (1.0 - self.beta1.powi(self.t as i32)));

        // Compute bias-corrected second raw moment estimate
        let v_hat = self.v.multiply_scalar(1.0 / (1.0 - self.beta2.powi(self.t as i32)));

        // Update parameters
        let update = m_hat.divide(&(v_hat.sqrt() + self.epsilon));
        *weights = weights.subtract(&update.multiply_scalar(self.learning_rate));

        // Increment timestep after all computations
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
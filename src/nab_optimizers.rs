use crate::nab_array::NDArray;

pub struct NablaOptimizer;


impl NablaOptimizer {

    /// Performs Stochastic Gradient Descent (SGD) update
    /// 
    /// w = w - learning_rate * gradient
    ///
    /// # Arguments
    ///
    /// * `weights` - NDArray of current weights to update
    /// * `gradient` - NDArray of gradients for the weights
    /// * `learning_rate` - Learning rate for the update
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_optimizers::NablaOptimizer;
    ///
    /// let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
    /// let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
    /// let learning_rate = 0.1;
    ///
    /// NablaOptimizer::sgd_update(&mut weights, &gradients, learning_rate);
    /// ```
    pub fn sgd_update(weights: &mut NDArray, gradient: &NDArray, learning_rate: f64) {
        let update = gradient.multiply_scalar(learning_rate);
        *weights = weights.subtract(&update);
    }

    /// Performs SGD update with momentum
    /// 
    /// v = momentum * v - learning_rate * gradient
    /// w = w + v
    ///
    /// # Arguments
    ///
    /// * `weights` - NDArray of current weights to update
    /// * `gradient` - NDArray of gradients for the weights
    /// * `velocity` - Mutable reference to momentum velocity
    /// * `learning_rate` - Learning rate for the update
    /// * `momentum` - Momentum coefficient (default: 0.9)
    pub fn sgd_momentum_update(
        weights: &mut NDArray,
        gradient: &NDArray,
        velocity: &mut NDArray,
        learning_rate: f64,
        momentum: f64,
    ) {
        // Update velocity
        *velocity = velocity.multiply_scalar(momentum)
            .subtract(&gradient.multiply_scalar(learning_rate));
        
        // Update weights using velocity
        *weights = weights.clone().add(velocity);
    }

    /// Performs RMSprop update
    /// 
    /// cache = decay_rate * cache + (1 - decay_rate) * gradient^2
    /// w = w - learning_rate * gradient / (sqrt(cache) + epsilon)
    ///
    /// # Arguments
    ///
    /// * `weights` - NDArray of current weights to update
    /// * `gradient` - NDArray of gradients for the weights
    /// * `cache` - Running average of squared gradients
    /// * `learning_rate` - Learning rate for the update
    /// * `decay_rate` - Decay rate for running average (default: 0.9)
    /// * `epsilon` - Small value for numerical stability (default: 1e-8)
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_optimizers::NablaOptimizer;
    ///
    /// let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
    /// let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
    /// let mut cache = NDArray::zeros(vec![3]);
    /// let learning_rate = 0.01;
    /// let decay_rate = 0.9;
    /// let epsilon = 1e-8;
    ///
    /// NablaOptimizer::rmsprop_update(
    ///     &mut weights, 
    ///     &gradients, 
    ///     &mut cache,
    ///     learning_rate,
    ///     decay_rate,
    ///     epsilon
    /// );
    /// ```
    pub fn rmsprop_update(
        weights: &mut NDArray,
        gradient: &NDArray,
        cache: &mut NDArray,
        learning_rate: f64,
        decay_rate: f64,
        epsilon: f64,
    ) {
        // Update cache
        *cache = cache.multiply_scalar(decay_rate)
            .add(&gradient.multiply(gradient).multiply_scalar(1.0 - decay_rate));
        
        // Compute update
        let update = gradient.divide(
            &cache.sqrt().add_scalar(epsilon)
        ).multiply_scalar(learning_rate);
        
        // Update weights
        *weights = weights.subtract(&update);
    }

    /// Performs Adam (Adaptive Moment Estimation) update
    /// 
    /// m = beta1 * m + (1 - beta1) * gradient           // Update first moment
    /// v = beta2 * v + (1 - beta2) * gradient^2         // Update second moment
    /// m_hat = m / (1 - beta1^t)                        // Bias correction
    /// v_hat = v / (1 - beta2^t)                        // Bias correction
    /// w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    ///
    /// # Arguments
    ///
    /// * `weights` - NDArray of current weights to update
    /// * `gradient` - NDArray of gradients for the weights
    /// * `m` - First moment vector (momentum)
    /// * `v` - Second moment vector (uncentered variance)
    /// * `t` - Current timestep (starting from 1)
    /// * `learning_rate` - Learning rate for the update
    /// * `beta1` - Exponential decay rate for first moment (default: 0.9)
    /// * `beta2` - Exponential decay rate for second moment (default: 0.999)
    /// * `epsilon` - Small value for numerical stability (default: 1e-8)
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_optimizers::NablaOptimizer;
    ///
    /// let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
    /// let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
    /// let mut m = NDArray::zeros(vec![3]);
    /// let mut v = NDArray::zeros(vec![3]);
    /// let t = 1;
    /// let learning_rate = 0.001;
    /// let beta1 = 0.9;
    /// let beta2 = 0.999;
    /// let epsilon = 1e-8;
    ///
    /// NablaOptimizer::adam_update(
    ///     &mut weights,
    ///     &gradients,
    ///     &mut m,
    ///     &mut v,
    ///     t,
    ///     learning_rate,
    ///     beta1,
    ///     beta2,
    ///     epsilon
    /// );
    /// ```
    pub fn adam_update(
        weights: &mut NDArray,
        gradient: &NDArray,
        m: &mut NDArray,
        v: &mut NDArray,
        t: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) {
        // Update biased first moment estimate
        *m = m.multiply_scalar(beta1)
            .add(&gradient.multiply_scalar(1.0 - beta1));
        
        // Update biased second raw moment estimate
        *v = v.multiply_scalar(beta2)
            .add(&gradient.multiply(gradient).multiply_scalar(1.0 - beta2));
        
        // Compute bias-corrected first moment estimate
        let m_hat = m.multiply_scalar(1.0 / (1.0 - beta1.powi(t as i32)));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v.multiply_scalar(1.0 / (1.0 - beta2.powi(t as i32)));
        
        // Compute the update
        let update = m_hat.divide(&v_hat.sqrt().add_scalar(epsilon))
            .multiply_scalar(learning_rate);
        
        // Apply update to weights
        *weights = weights.subtract(&update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        // Initialize test data
        let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
        let learning_rate = 0.1;

        // Store initial weights
        let initial_weights = weights.clone();

        // Perform update
        NablaOptimizer::sgd_update(&mut weights, &gradients, learning_rate);

        // Verify weights were updated correctly
        for i in 0..weights.data().len() {
            let expected = initial_weights.data()[i] - learning_rate * gradients.data()[i];
            assert!((weights.data()[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sgd_momentum() {
        // Initialize test data
        let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
        let mut velocity = NDArray::zeros(vec![3]);
        let learning_rate = 0.1;
        let momentum = 0.9;

        // Store initial weights
        let initial_weights = weights.clone();

        // Perform update
        NablaOptimizer::sgd_momentum_update(
            &mut weights,
            &gradients,
            &mut velocity,
            learning_rate,
            momentum
        );

        // Verify weights changed
        assert!(weights.data() != initial_weights.data());

        // Verify velocity is non-zero
        assert!(velocity.data().iter().any(|&x| x != 0.0));

        // Verify momentum effect (velocity should be -learning_rate * gradients)
        for i in 0..velocity.data().len() {
            assert!((velocity.data()[i] + learning_rate * gradients.data()[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rmsprop_update() {
        // Initialize test data
        let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
        let mut cache = NDArray::zeros(vec![3]);
        let learning_rate = 0.01;
        let decay_rate = 0.9;
        let epsilon = 1e-8;

        // Store initial values
        let initial_weights = weights.clone();
        let initial_cache = cache.clone();

        // Perform update
        NablaOptimizer::rmsprop_update(
            &mut weights,
            &gradients,
            &mut cache,
            learning_rate,
            decay_rate,
            epsilon
        );

        // Verify weights changed
        assert!(weights.data() != initial_weights.data(),
            "Weights should be updated");

        // Verify cache was updated
        assert!(cache.data() != initial_cache.data(),
            "Cache should be updated");

        // Verify cache contains squared gradient information
        for i in 0..cache.data().len() {
            let expected_cache = (1.0 - decay_rate) * gradients.data()[i].powi(2);
            assert!((cache.data()[i] - expected_cache).abs() < 1e-6,
                "Cache should contain squared gradient information");
        }

        // Test multiple updates to verify cache accumulation
        let prev_cache = cache.clone();
        NablaOptimizer::rmsprop_update(
            &mut weights,
            &gradients,
            &mut cache,
            learning_rate,
            decay_rate,
            epsilon
        );

        // Verify cache decay
        for i in 0..cache.data().len() {
            assert!(cache.data()[i] > prev_cache.data()[i],
                "Cache should accumulate gradient information");
        }
    }

    #[test]
    fn test_adam_update() {
        // Initialize test data
        let mut weights = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = NDArray::from_vec(vec![0.1, 0.2, 0.3]);
        let mut m = NDArray::zeros(vec![3]);
        let mut v = NDArray::zeros(vec![3]);
        let t = 1;
        let learning_rate = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        // Store initial values
        let initial_weights = weights.clone();
        let initial_m = m.clone();
        let initial_v = v.clone();

        // Perform update
        NablaOptimizer::adam_update(
            &mut weights,
            &gradients,
            &mut m,
            &mut v,
            t,
            learning_rate,
            beta1,
            beta2,
            epsilon
        );

        // Verify weights changed
        assert!(weights.data() != initial_weights.data(),
            "Weights should be updated");

        // Verify moment estimates changed
        assert!(m.data() != initial_m.data(),
            "First moment should be updated");
        assert!(v.data() != initial_v.data(),
            "Second moment should be updated");

        // Verify first moment update
        for i in 0..m.data().len() {
            let expected_m = (1.0 - beta1) * gradients.data()[i];
            assert!((m.data()[i] - expected_m).abs() < 1e-6,
                "First moment should be correctly updated");
        }

        // Verify second moment update
        for i in 0..v.data().len() {
            let expected_v = (1.0 - beta2) * gradients.data()[i].powi(2);
            assert!((v.data()[i] - expected_v).abs() < 1e-6,
                "Second moment should be correctly updated");
        }

        // Test multiple updates
        let prev_m = m.clone();
        let prev_v = v.clone();
        
        NablaOptimizer::adam_update(
            &mut weights,
            &gradients,
            &mut m,
            &mut v,
            t + 1,
            learning_rate,
            beta1,
            beta2,
            epsilon
        );

        // Verify moment accumulation
        assert!(m.data().iter().zip(prev_m.data().iter())
            .all(|(&new, &old)| new != old),
            "First moment should accumulate");
        assert!(v.data().iter().zip(prev_v.data().iter())
            .all(|(&new, &old)| new != old),
            "Second moment should accumulate");
    }
} 
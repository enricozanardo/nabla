use crate::nab_array::NDArray;
use crate::nab_math::NabMath;

pub struct NablaActivation;

impl NablaActivation {
    /// Applies the Rectified Linear Unit (ReLU) activation function in forward pass
    /// 
    /// ReLU(x) = max(0, x)
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with ReLU activation applied element-wise
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_activations::NablaActivation;
    ///
    /// let x = NDArray::from_vec(vec![-1.0, 0.0, 2.0]);
    /// let output = NablaActivation::relu_forward(&x);
    /// assert_eq!(output.data(), &[0.0, 0.0, 2.0]);
    /// ```
    pub fn relu_forward(x: &NDArray) -> NDArray {
        NabMath::relu(x)
    }

    /// Computes the gradient for ReLU activation in backward pass
    /// 
    /// ReLU'(x) = 1 if x > 0, else 0
    ///
    /// # Arguments
    ///
    /// * `gradient` - Gradient from the next layer
    /// * `x` - Original input to the ReLU function
    ///
    /// # Returns
    ///
    /// NDArray containing the gradients for backpropagation
    pub fn relu_backward(gradient: &NDArray, x: &NDArray) -> NDArray {
        // ReLU derivative: 1 if x > 0, 0 otherwise
        let dx = x.map(|val| if val > 0.0 { 1.0 } else { 0.0 });
        gradient * &dx
    }

    /// Applies the Softmax activation function in forward pass
    /// 
    /// Softmax(x)_i = exp(x_i) / sum(exp(x_j))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `axis` - Optional axis along which to apply softmax
    ///
    /// # Returns
    ///
    /// NDArray with softmax probabilities that sum to 1
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_activations::NablaActivation;
    ///
    /// let x = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
    /// let output = NablaActivation::softmax_forward(&x, None);
    /// let sum: f64 = output.data().iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn softmax_forward(x: &NDArray, axis: Option<usize>) -> NDArray {
        NabMath::softmax(x, axis)
    }

    /// Computes the gradient for Softmax activation in backward pass
    /// 
    /// Note: For numerical stability, the actual softmax gradient computation
    /// is typically combined with the loss function gradient.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Gradient from the loss function
    /// * `output` - Output from the softmax forward pass
    ///
    /// # Returns
    ///
    /// NDArray containing the gradients for backpropagation
    pub fn softmax_backward(gradient: &NDArray, _output: &NDArray) -> NDArray {
        // Softmax derivative is handled in loss function for numerical stability
        gradient.clone()
    }

    /// Applies the Sigmoid activation function in forward pass
    /// 
    /// sigmoid(x) = 1 / (1 + exp(-x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with values squashed between 0 and 1
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_activations::NablaActivation;
    ///
    /// let x = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
    /// let output = NablaActivation::sigmoid_forward(&x);
    /// // Values should be between 0 and 1
    /// for &val in output.data() {
    ///     assert!(val > 0.0 && val < 1.0);
    /// }
    /// ```
    pub fn sigmoid_forward(x: &NDArray) -> NDArray {
        NabMath::sigmoid(x)
    }

    /// Computes the gradient for Sigmoid activation in backward pass
    /// 
    /// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    ///
    /// # Arguments
    ///
    /// * `gradient` - Gradient from the next layer
    /// * `output` - Output from the sigmoid forward pass
    ///
    /// # Returns
    ///
    /// NDArray containing the gradients for backpropagation
    pub fn sigmoid_backward(gradient: &NDArray, output: &NDArray) -> NDArray {
        let sigmoid_derivative = output * &(output.scalar_sub(1.0).multiply_scalar(-1.0));
        gradient * &sigmoid_derivative
    }

    /// Applies the Leaky ReLU activation function in forward pass
    /// 
    /// leaky_relu(x) = x if x > 0, else alpha * x
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `alpha` - Slope for negative values (default: 0.01)
    ///
    /// # Returns
    ///
    /// NDArray with Leaky ReLU activation applied element-wise
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_activations::NablaActivation;
    ///
    /// let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
    /// let output = NablaActivation::leaky_relu_forward(&x, Some(0.1));
    /// // Negative values are scaled by alpha
    /// assert_eq!(output.data()[0], -0.2);
    /// // Positive values remain unchanged
    /// assert_eq!(output.data()[2], 2.0);
    /// ```
    pub fn leaky_relu_forward(x: &NDArray, alpha: Option<f64>) -> NDArray {
        NabMath::leaky_relu(x, alpha)
    }

    /// Computes the gradient for Leaky ReLU activation in backward pass
    /// 
    /// leaky_relu'(x) = 1 if x > 0, else alpha
    ///
    /// # Arguments
    ///
    /// * `gradient` - Gradient from the next layer
    /// * `x` - Original input to the Leaky ReLU function
    /// * `alpha` - Slope for negative values (default: 0.01)
    ///
    /// # Returns
    ///
    /// NDArray containing the gradients for backpropagation
    pub fn leaky_relu_backward(gradient: &NDArray, x: &NDArray, alpha: Option<f64>) -> NDArray {
        let alpha = alpha.unwrap_or(0.01);
        let dx = x.map(|val| if val >= 0.0 { 1.0 } else { alpha });
        gradient * &dx
    }

    /// Applies the Hyperbolic Tangent (tanh) activation function in forward pass
    /// 
    /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with values squashed between -1 and 1
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_array::NDArray;
    /// use nabla_ml::nab_activations::NablaActivation;
    ///
    /// let x = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
    /// let output = NablaActivation::tanh_forward(&x);
    /// // Values should be between -1 and 1
    /// for &val in output.data() {
    ///     assert!(val >= -1.0 && val <= 1.0);
    /// }
    /// ```
    pub fn tanh_forward(x: &NDArray) -> NDArray {
        NabMath::tanh(x)
    }

    /// Computes the gradient for tanh activation in backward pass
    /// 
    /// tanh'(x) = 1 - tanh²(x)
    ///
    /// # Arguments
    ///
    /// * `gradient` - Gradient from the next layer
    /// * `output` - Output from the tanh forward pass
    ///
    /// # Returns
    ///
    /// NDArray containing the gradients for backpropagation
    pub fn tanh_backward(gradient: &NDArray, output: &NDArray) -> NDArray {
        let tanh_derivative = output.multiply(output)  // tanh²(x)
            .scalar_sub(1.0)                          // -1 + tanh²(x)
            .multiply_scalar(-1.0);                   // 1 - tanh²(x)
        gradient * &tanh_derivative
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward_backward() {
        // Test forward pass with mixed positive/negative values
        let x = NDArray::from_vec(vec![-1.0, 0.0, 2.0]);
        let forward = NablaActivation::relu_forward(&x);
        // Verify ReLU zeros out negative values and keeps positive values
        assert_eq!(forward.data(), &[0.0, 0.0, 2.0]);

        // Test backward pass with uniform gradient
        let gradient = NDArray::from_vec(vec![1.0, 1.0, 1.0]);
        let backward = NablaActivation::relu_backward(&gradient, &x);
        // Verify gradient is zero for negative inputs and unchanged for positive inputs
        assert_eq!(backward.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_forward_backward() {
        // Test forward pass with increasing values
        let x = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let forward = NablaActivation::softmax_forward(&x, None);
        
        // Verify softmax output sums to 1 (probability distribution)
        let sum: f64 = forward.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Verify softmax maintains relative ordering (monotonicity)
        let mut prev = 0.0;
        for &val in forward.data() {
            assert!(val >= prev);
            prev = val;
        }
    }

    #[test]
    fn test_sigmoid_forward_backward() {
        // Test forward pass with various inputs
        let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
        let forward = NablaActivation::sigmoid_forward(&x);
        
        // Verify sigmoid output is between 0 and 1
        for &val in forward.data() {
            assert!(val > 0.0 && val < 1.0);
        }

        // Verify sigmoid(0) ≈ 0.5
        assert!((forward.data()[1] - 0.5).abs() < 1e-6);

        // Test backward pass
        let gradient = NDArray::from_vec(vec![1.0, 1.0, 1.0]);
        let backward = NablaActivation::sigmoid_backward(&gradient, &forward);
        
        // Verify gradient shape matches input
        assert_eq!(backward.shape(), x.shape());
        
        // Verify gradient is maximum at x = 0 (where sigmoid'(0) = 0.25)
        assert!(backward.data()[1] > backward.data()[0]);
        assert!(backward.data()[1] > backward.data()[2]);
    }

    #[test]
    fn test_leaky_relu_forward_backward() {
        // Test forward pass with default alpha
        let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
        let forward = NablaActivation::leaky_relu_forward(&x, None);
        
        // Verify positive values remain unchanged
        assert_eq!(forward.data()[2], 2.0);
        // Verify negative values are scaled by default alpha (0.01)
        assert_eq!(forward.data()[0], -0.02);
        // Verify zero remains unchanged
        assert_eq!(forward.data()[1], 0.0);

        // Test forward pass with custom alpha
        let forward_custom = NablaActivation::leaky_relu_forward(&x, Some(0.1));
        // Verify negative values are scaled by custom alpha
        assert_eq!(forward_custom.data()[0], -0.2);

        // Test backward pass
        let gradient = NDArray::from_vec(vec![1.0, 1.0, 1.0]);
        let backward = NablaActivation::leaky_relu_backward(&gradient, &x, Some(0.1));
        
        // Verify gradient for positive values is unchanged
        assert_eq!(backward.data()[2], 1.0);
        // Verify gradient for negative values is scaled by alpha
        assert_eq!(backward.data()[0], 0.1);
        // Verify gradient at zero is 1 (positive side of derivative)
        assert_eq!(backward.data()[1], 1.0);
    }

    #[test]
    fn test_tanh_forward_backward() {
        // Test forward pass with various inputs
        let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
        let forward = NablaActivation::tanh_forward(&x);
        
        // Verify tanh output is between -1 and 1
        for &val in forward.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }

        // Verify tanh(0) = 0
        assert!(forward.data()[1].abs() < 1e-6);

        // Test backward pass
        let gradient = NDArray::from_vec(vec![1.0, 1.0, 1.0]);
        let backward = NablaActivation::tanh_backward(&gradient, &forward);
        
        // Verify gradient shape matches input
        assert_eq!(backward.shape(), x.shape());
        
        // Verify gradient is maximum at x = 0 (where tanh'(0) = 1)
        assert!(backward.data()[1] > backward.data()[0]);
        assert!(backward.data()[1] > backward.data()[2]);

        // Verify gradient at x = 0 is close to 1
        assert!((backward.data()[1] - 1.0).abs() < 1e-6);
    }
}
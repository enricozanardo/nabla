//! Mathematical functions for NDArray operations
//! 
//! This module provides mathematical operations commonly found in NumPy,
//! implemented for the NDArray struct.

use crate::nab_array::NDArray;

/// Mathematical functions for NDArray
pub struct NabMath;

impl NDArray {
    /// Calculates the square root of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the square root of each element.
    #[allow(dead_code)]
    pub fn sqrt(&self) -> Self {
        let data = self.data().iter().map(|x| x.sqrt()).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Calculates the exponential (e^x) of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the exponential of each element.
    #[allow(dead_code)]
    pub fn exp(&self) -> Self {
        let data = self.data().iter().map(|x| x.exp()).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Calculates the sine of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the sine of each element.
    #[allow(dead_code)]
    pub fn sin(&self) -> Self {
        let data: Vec<f64> = self.data().iter().map(|&x| x.sin()).collect();
        Self::new(data, self.shape().to_vec())
    }

    /// Calculates the cosine of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the cosine of each element.
    #[allow(dead_code)]
    pub fn cos(&self) -> Self {
        let data: Vec<f64> = self.data().iter().map(|&x| x.cos()).collect();
        Self::new(data, self.shape().to_vec())
    }

    /// Calculates the natural logarithm of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the natural logarithm of each element.
    #[allow(dead_code)]
    pub fn ln(&self) -> Self {
        let data: Vec<f64> = self.data().iter().map(|&x| x.ln()).collect();
        Self::new(data, self.shape().to_vec())
    }

}

impl NabMath {
    /// Computes the sigmoid function element-wise
    /// 
    /// sigmoid(x) = 1 / (1 + exp(-x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with sigmoid applied element-wise
    pub fn sigmoid(x: &NDArray) -> NDArray {
        x.map(|val| 1.0 / (1.0 + (-val).exp()))
    }

    /// Computes the derivative of sigmoid function element-wise
    ///
    /// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with sigmoid derivative applied element-wise
    pub fn sigmoid_derivative(x: &NDArray) -> NDArray {
        let sigmoid_x = Self::sigmoid(x);
        sigmoid_x.map(|val| val * (1.0 - val))
    }

    /// Computes the hyperbolic tangent function element-wise
    ///
    /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with tanh applied element-wise
    pub fn tanh(x: &NDArray) -> NDArray {
        x.map(|val| val.tanh())
    }

    /// Computes the derivative of tanh function element-wise
    ///
    /// tanh'(x) = 1 - tanh²(x)
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with tanh derivative applied element-wise
    pub fn tanh_derivative(x: &NDArray) -> NDArray {
        let tanh_x = Self::tanh(x);
        tanh_x.map(|val| 1.0 - val * val)
    }

    /// Computes the ReLU function element-wise
    ///
    /// ReLU(x) = max(0, x)
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with ReLU applied element-wise
    pub fn relu(x: &NDArray) -> NDArray {
        x.map(|val| val.max(0.0))
    }

    /// Computes the derivative of ReLU function element-wise
    ///
    /// ReLU'(x) = 1 if x > 0, 0 otherwise
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    ///
    /// # Returns
    ///
    /// NDArray with ReLU derivative applied element-wise
    pub fn relu_derivative(x: &NDArray) -> NDArray {
        x.map(|val| if val > 0.0 { 1.0 } else { 0.0 })
    }

    /// Computes the softmax function along the specified axis
    ///
    /// softmax(x) = exp(x) / sum(exp(x))
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `axis` - Axis along which to compute softmax (default: -1 for last axis)
    ///
    /// # Returns
    ///
    /// NDArray with softmax applied along specified axis
    pub fn softmax(x: &NDArray, _axis: Option<usize>) -> NDArray {
        assert!(x.ndim() == 1 || x.ndim() == 2, "Softmax is only defined for 1D or 2D arrays");

        let exp = x.map(|val| val.exp());
        
        if x.ndim() == 1 {
            // For 1D arrays
            let sum = exp.sum();
            exp.map(|val| val / sum)
        } else {
            // For 2D arrays, always compute along rows (axis=1)
            let (rows, cols) = (x.shape()[0], x.shape()[1]);
            let sum = exp.sum_axis(1);  // Shape: [rows, 1]
            
            // Create broadcasted sum array
            let mut result_data = Vec::with_capacity(rows * cols);
            for i in 0..rows {
                for j in 0..cols {
                    // Use sum[i] for each row instead of sum[0]
                    result_data.push(exp.data()[i * cols + j] / sum.data()[i]);
                }
            }
            
            NDArray::new(result_data, x.shape().to_vec())
        }
    }

    /// Computes the derivative of softmax function
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray (softmax output)
    ///
    /// # Returns
    ///
    /// NDArray with softmax derivative
    pub fn softmax_derivative(x: &NDArray) -> NDArray {
        x.map(|val| val * (1.0 - val))
    }

    /// Computes the Leaky ReLU function element-wise
    ///
    /// LeakyReLU(x) = max(alpha * x, x)
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `alpha` - Slope for negative values (default: 0.01)
    ///
    /// # Returns
    ///
    /// NDArray with Leaky ReLU applied element-wise
    pub fn leaky_relu(x: &NDArray, alpha: Option<f64>) -> NDArray {
        let alpha = alpha.unwrap_or(0.01);
        x.map(|val| if val > 0.0 { val } else { alpha * val })
    }

    /// Computes the derivative of Leaky ReLU function
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `alpha` - Slope for negative values (default: 0.01)
    ///
    /// # Returns
    ///
    /// NDArray with Leaky ReLU derivative
    pub fn leaky_relu_derivative(x: &NDArray, alpha: Option<f64>) -> NDArray {
        let alpha = alpha.unwrap_or(0.01);
        x.map(|val| if val > 0.0 { 1.0 } else { alpha })
    }

    /// Computes the ELU (Exponential Linear Unit) function
    ///
    /// ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `alpha` - Scale for negative values (default: 1.0)
    ///
    /// # Returns
    ///
    /// NDArray with ELU applied element-wise
    pub fn elu(x: &NDArray, alpha: Option<f64>) -> NDArray {
        let alpha = alpha.unwrap_or(1.0);
        x.map(|val| if val > 0.0 { val } else { alpha * (val.exp() - 1.0) })
    }

    /// Computes the derivative of ELU function
    ///
    /// # Arguments
    ///
    /// * `x` - Input NDArray
    /// * `alpha` - Scale for negative values (default: 1.0)
    ///
    /// # Returns
    ///
    /// NDArray with ELU derivative
    pub fn elu_derivative(x: &NDArray, alpha: Option<f64>) -> NDArray {
        let alpha = alpha.unwrap_or(1.0);
        x.map(|val| if val > 0.0 { 1.0 } else { alpha * val.exp() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt() {
        let arr = NDArray::from_vec(vec![1.0, 4.0, 9.0]);
        let sqrt_arr = arr.sqrt();
        assert_eq!(sqrt_arr.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_exp() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0]);
        let exp_arr = arr.exp();
        assert!((exp_arr.data()[0] - 1.0).abs() < 1e-4);
        assert!((exp_arr.data()[1] - std::f64::consts::E).abs() < 1e-4);
        assert!((exp_arr.data()[2] - std::f64::consts::E.powi(2)).abs() < 1e-4);
    }

    /// Tests sigmoid function computation
    #[test]
    fn test_sigmoid() {
        let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
        let result = NabMath::sigmoid(&x);
        
        // Test output range (0 to 1)
        for &val in result.data() {
            assert!(val > 0.0 && val < 1.0);
        }
        
        // Test sigmoid(0) = 0.5
        assert!((result.data()[1] - 0.5).abs() < 1e-6);
        
        // Test symmetry: sigmoid(-x) = 1 - sigmoid(x)
        assert!((result.data()[0] - (1.0 - result.data()[2])).abs() < 1e-6);
    }

    /// Tests sigmoid derivative computation
    #[test]
    fn test_sigmoid_derivative() {
        let x = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = NabMath::sigmoid_derivative(&x);
        assert!((result.data()[0] - 0.1966).abs() < 1e-4);
        assert!((result.data()[1] - 0.2500).abs() < 1e-4);
        assert!((result.data()[2] - 0.1966).abs() < 1e-4);
    }

    /// Tests tanh function computation
    #[test]
    fn test_tanh() {
        let x = NDArray::from_vec(vec![-2.0, 0.0, 2.0]);
        let result = NabMath::tanh(&x);
        
        // Test output range (-1 to 1)
        for &val in result.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
        
        // Test tanh(0) = 0
        assert!(result.data()[1].abs() < 1e-6);
        
        // Test symmetry: tanh(-x) = -tanh(x)
        assert!((result.data()[0] + result.data()[2]).abs() < 1e-6);
    }

    /// Tests tanh derivative computation
    #[test]
    fn test_tanh_derivative() {
        let x = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = NabMath::tanh_derivative(&x);
        assert!((result.data()[0] - 0.4199).abs() < 1e-4);
        assert!((result.data()[1] - 1.0000).abs() < 1e-4);
        assert!((result.data()[2] - 0.4199).abs() < 1e-4);
    }

    /// Tests ReLU function computation
    #[test]
    fn test_relu() {
        let x = NDArray::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = NabMath::relu(&x);
        
        // Test positive values remain unchanged
        assert_eq!(result.data()[3], 1.0);
        assert_eq!(result.data()[4], 2.0);
        
        // Test negative values become zero
        assert_eq!(result.data()[0], 0.0);
        assert_eq!(result.data()[1], 0.0);
        
        // Test zero remains zero
        assert_eq!(result.data()[2], 0.0);
    }

    /// Tests ReLU derivative computation
    #[test]
    fn test_relu_derivative() {
        let x = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = NabMath::relu_derivative(&x);
        assert_eq!(result.data(), &[0.0, 0.0, 1.0]);
    }

    /// Tests softmax computation on different dimensions
    #[test]
    fn test_softmax() {
        // Test 1D array
        let x = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = NabMath::softmax(&x, None);
        
        // Test sum equals 1
        let sum: f64 = result.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Test monotonicity (larger inputs -> larger probabilities)
        for i in 1..result.data().len() {
            assert!(result.data()[i] > result.data()[i-1]);
        }

        // Test 2D array
        let x = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ]);
        let result = NabMath::softmax(&x, Some(1));
        
        // Test each row sums to 1
        for i in 0..2 {
            let row_sum: f64 = result.data()[i*3..(i+1)*3].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    /// Tests softmax derivative computation
    #[test]
    fn test_softmax_derivative() {
        let x = NDArray::from_vec(vec![0.1, 0.7, 0.2]);
        let result = NabMath::softmax_derivative(&x);
        assert_eq!(result.shape(), &[3]);
        // Verify derivative values
        for &val in result.data() {
            assert!(val >= 0.0 && val <= 0.25); // Maximum value is 0.25 for softmax derivative
        }
    }

    /// Tests Leaky ReLU computation with different alphas
    #[test]
    fn test_leaky_relu() {
        let x = NDArray::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        
        // Test with default alpha
        let result = NabMath::leaky_relu(&x, None);
        assert_eq!(result.data()[3], 1.0);  // Positive values unchanged
        assert_eq!(result.data()[4], 2.0);
        assert_eq!(result.data()[0], -0.02); // Negative values scaled by 0.01
        assert_eq!(result.data()[2], 0.0);   // Zero unchanged
        
        // Test with custom alpha
        let result = NabMath::leaky_relu(&x, Some(0.1));
        assert_eq!(result.data()[3], 1.0);   // Positive values unchanged
        assert_eq!(result.data()[0], -0.2);  // Negative values scaled by 0.1
    }

    /// Tests ELU computation with different alphas
    #[test]
    fn test_elu() {
        let x = NDArray::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        
        // Test with default alpha
        let result = NabMath::elu(&x, None);
        assert!(result.data()[0] < -0.8); // ELU(-2) ≈ -0.86
        assert_eq!(result.data()[3], 1.0);

        // Test with custom alpha
        let result = NabMath::elu(&x, Some(2.0));
        assert!(result.data()[0] < -1.7); // ELU(-2) with alpha=2 ≈ -1.73
        assert_eq!(result.data()[3], 1.0);
    }

    /// Tests ELU derivative computation
    #[test]
    fn test_elu_derivative() {
        let x = NDArray::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = NabMath::elu_derivative(&x, None);
        assert!(result.data()[0] > 0.0 && result.data()[0] < 1.0);
        assert_eq!(result.data()[3], 1.0);
    }
} 
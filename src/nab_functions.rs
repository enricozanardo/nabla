use crate::nab_array::NDArray;
use crate::nabla::ActivationType;

impl NDArray {

    /// Calculates the hyperbolic tangent of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the hyperbolic tangent of each element.
    #[allow(dead_code)]
    pub fn tanh(&self) -> Self {
        let data = self.data().iter().map(|x| x.tanh()).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the ReLU function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the ReLU function applied to each element.
    #[allow(dead_code)]
    pub fn relu(&self) -> Self {
        let data = self.data().iter().map(|x| x.max(0.0)).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the Leaky ReLU function to each element in the array
    ///
    /// # Arguments
    ///
    /// * `alpha` - The slope for negative values.
    ///
    /// # Returns
    ///
    /// A new NDArray with the Leaky ReLU function applied to each element.
    #[allow(dead_code)]
    pub fn leaky_relu(&self, alpha: f64) -> Self {
        let data = self.data().iter().map(|x| if *x > 0.0 { *x } else { alpha * *x }).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the Sigmoid function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the Sigmoid function applied to each element.
    #[allow(dead_code)]
    pub fn sigmoid(&self) -> Self {
        let data = self.data().iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the specified activation function to the array
    pub fn activate(&self, activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::Sigmoid => self.sigmoid(),
            ActivationType::ReLU => self.relu(),
            ActivationType::LeakyReLU => self.leaky_relu(0.01),
            ActivationType::Softmax => self.softmax(),
            ActivationType::Tanh => self.tanh(),
        }
    }

    /// Applies the softmax function to the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the softmax function applied
    pub fn softmax(&self) -> Self {
        assert!(self.ndim() == 1 || self.ndim() == 2, "Softmax is only defined for 1D or 2D arrays");

        let exp = self.exp();
        
        if self.ndim() == 1 {
            // For 1D arrays
            let sum = exp.sum();
            exp.divide_scalar(sum)
        } else {
            // For 2D arrays
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let sum = exp.sum_axis(1);  // Shape: [rows, 1]
            
            // Create broadcasted sum array
            let mut broadcasted_sum = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    broadcasted_sum[i * cols + j] = sum.data()[i];
                }
            }
            let sum_array = NDArray::new(broadcasted_sum, self.shape.clone());
            
            exp.divide(&sum_array)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let arr = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
        let relu_arr = arr.relu();
        assert_eq!(relu_arr.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let arr = NDArray::from_vec(vec![-1.0, 0.0, 1.0]);
        let leaky_relu_arr = arr.leaky_relu(0.01);
        assert_eq!(leaky_relu_arr.data(), &[-0.01, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, -1.0]);
        let sigmoid_arr = arr.sigmoid();
        assert!((sigmoid_arr.data()[0] - 0.5).abs() < 1e-4);
        assert!((sigmoid_arr.data()[1] - 0.7311).abs() < 1e-4);
        assert!((sigmoid_arr.data()[2] - 0.2689).abs() < 1e-4);
    }


    #[test]
    fn test_tanh() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, -1.0]);
        let tanh_arr = arr.tanh();
        assert!((tanh_arr.data()[0] - 0.0).abs() < 1e-4);
        assert!((tanh_arr.data()[1] - 0.7616).abs() < 1e-4);
        assert!((tanh_arr.data()[2] + 0.7616).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let input = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 0.5],
            vec![1.0, 1.0, 1.0],
        ]);
        let softmax = input.softmax();
        
        // Check that outputs sum to 1 for each row
        let sum_row_1: f64 = softmax.data()[0..3].iter().sum();
        let sum_row_2: f64 = softmax.data()[3..6].iter().sum();
        assert!((sum_row_1 - 1.0).abs() < 1e-6);
        assert!((sum_row_2 - 1.0).abs() < 1e-6);
        
        // Check that all values are between 0 and 1
        assert!(softmax.data().iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_sum_axis() {
        let arr = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let result = arr.sum_axis(0);
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]); // Sum along columns
        assert_eq!(result.shape(), &[1, 3]); // Shape should be [1, 3]

        let result = arr.sum_axis(1);
        assert_eq!(result.data(), &[6.0, 15.0]); // Sum along rows
        assert_eq!(result.shape(), &[2, 1]); // Shape should be [2, 1]
    }
}
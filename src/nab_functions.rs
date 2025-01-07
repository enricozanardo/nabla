use crate::nab_array::NDArray;

impl NDArray {

    /// Calculates the hyperbolic tangent of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the hyperbolic tangent of each element.
    pub fn tanh(&self) -> Self {
        let data = self.data().iter().map(|x| x.tanh()).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the ReLU function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the ReLU function applied to each element.
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
    pub fn leaky_relu(&self, alpha: f64) -> Self {
        let data = self.data().iter().map(|x| if *x > 0.0 { *x } else { alpha * *x }).collect();
        NDArray::new(data, self.shape().to_vec())
    }

    /// Applies the Sigmoid function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the Sigmoid function applied to each element.
    pub fn sigmoid(&self) -> Self {
        let data = self.data().iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        NDArray::new(data, self.shape().to_vec())
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

}
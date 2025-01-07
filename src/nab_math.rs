use crate::nab_array::NDArray;

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


    
} 
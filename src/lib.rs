use std::fmt;
use rand::Rng;
use rand_distr::StandardNormal;
use std::ops::Add;
use std::ops::Sub;
use std::ops::{Mul, Div};
use std::f64;
use std::ops::{Index, IndexMut};

/// A multi-dimensional array implementation inspired by NumPy's ndarray
#[derive(Debug, Clone)]
pub struct NDArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl NDArray {
    /// Creates a new NDArray with the given data and shape
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    /// Creates a 1D array (vector) from a vector
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    /// Creates a 2D array (matrix) from a vector of vectors
    pub fn from_matrix(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data.get(0).map_or(0, |row| row.len());
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        Self::new(flat_data, vec![rows, cols])
    }

    /// Returns the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns a reference to the underlying data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Creates a 1D array with a range of numbers
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the range (inclusive).
    /// * `stop` - The stopping value of the range (exclusive).
    /// * `step` - The step size between each value in the range.
    ///
    /// # Returns
    ///
    /// A 1D NDArray containing the range of numbers.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut current = start;
        while current < stop {
            data.push(current);
            current += step;
        }
        Self::from_vec(data)
    }

    /// Creates a 1D array filled with zeros
    pub fn zeros(size: usize) -> Self {
        Self::from_vec(vec![0.0; size])
    }

    /// Creates a 2D array (matrix) filled with zeros
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![0.0; rows * cols], vec![rows, cols])
    }

    /// Creates a 1D array filled with ones
    pub fn ones(size: usize) -> Self {
        Self::from_vec(vec![1.0; size])
    }

    /// Creates a 2D array (matrix) filled with ones
    pub fn ones_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![1.0; rows * cols], vec![rows, cols])
    }

    /// Creates a 1D array with evenly spaced numbers over a specified interval
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the interval.
    /// * `end` - The ending value of the interval.
    /// * `num` - The number of evenly spaced samples to generate.
    /// * `precision` - The number of decimal places to round each value to.
    ///
    /// # Returns
    ///
    /// A 1D NDArray containing the evenly spaced numbers.
    pub fn linspace(start: f64, end: f64, num: usize, precision: usize) -> Self {
        assert!(num > 1, "Number of samples must be greater than 1");
        let step = (end - start) / (num - 1) as f64;
        let mut data = Vec::with_capacity(num);
        let factor = 10f64.powi(precision as i32);
        for i in 0..num {
            let value = start + step * i as f64;
            let rounded_value = (value * factor).round() / factor;
            data.push(rounded_value);
        }
        Self::from_vec(data)
    }

    /// Creates an identity matrix of size `n x n`
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the identity matrix.
    ///
    /// # Returns
    ///
    /// An `n x n` identity matrix as an NDArray.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(data, vec![n, n])
    }

    /// Creates a 1D array of random numbers between 0 and 1
    pub fn rand(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random numbers between 0 and 1
    pub fn rand_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen()).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Creates a 1D array of random numbers following a normal distribution
    pub fn randn(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random numbers following a normal distribution
    pub fn randn_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.sample(StandardNormal)).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Creates a 1D array of random integers between `low` and `high`
    pub fn randint(low: i32, high: i32, size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen_range(low..high) as f64).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random integers between `low` and `high`
    pub fn randint_2d(low: i32, high: i32, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(low..high) as f64).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Reshapes the array to the specified shape
    ///
    /// # Arguments
    ///
    /// * `new_shape` - A vector representing the new shape.
    ///
    /// # Returns
    ///
    /// A new NDArray with the specified shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), new_size, "New shape must have the same number of elements as the original array");
        Self::new(self.data.clone(), new_shape)
    }

    /// Returns the maximum value in the array
    pub fn max(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the maximum value in the array
    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Returns the minimum value in the array
    pub fn min(&self) -> f64 {
        *self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the minimum value in the array
    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Calculates the square root of each element in the array
    pub fn sqrt(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sqrt()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the exponential (e^x) of each element in the array
    pub fn exp(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.exp()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the sine of each element in the array
    pub fn sin(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sin()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the cosine of each element in the array
    pub fn cos(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.cos()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the natural logarithm of each element in the array
    pub fn ln(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.ln()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Returns a specific element from the array
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Returns a slice of the array from start to end (exclusive)
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let data = self.data[start..end].to_vec();
        Self::from_vec(data)
    }

    /// Sets a specific element in the array
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Sets a range of elements in the array to a specific value
    pub fn set_range(&mut self, start: usize, end: usize, value: f64) {
        for i in start..end {
            self.data[i] = value;
        }
    }

    /// Returns a copy of the array
    pub fn copy(&self) -> Self {
        Self::new(self.data.clone(), self.shape.clone())
    }

    /// Returns a view (slice) of the array from start to end (exclusive)
    pub fn view(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start..end]
    }

    /// Returns a mutable view (slice) of the array from start to end (exclusive)
    pub fn view_mut(&mut self, start: usize, end: usize) -> &mut [f64] {
        &mut self.data[start..end]
    }

    /// Returns a specific element from a 2D array
    pub fn get_2d(&self, row: usize, col: usize) -> f64 {
        assert_eq!(self.ndim(), 2, "get_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col]
    }

    /// Sets a specific element in a 2D array
    pub fn set_2d(&mut self, row: usize, col: usize, value: f64) {
        assert_eq!(self.ndim(), 2, "set_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col] = value;
    }

    /// Returns a sub-matrix from a 2D array
    pub fn sub_matrix(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        assert_eq!(self.ndim(), 2, "sub_matrix is only applicable to 2D arrays");
        let cols = self.shape[1];
        let mut data = Vec::new();
        for row in row_start..row_end {
            for col in col_start..col_end {
                data.push(self.data[row * cols + col]);
            }
        }
        Self::new(data, vec![row_end - row_start, col_end - col_start])
    }

    /// Returns a boolean array indicating whether each element satisfies the condition
    pub fn greater_than(&self, threshold: f64) -> Vec<bool> {
        self.data.iter().map(|&x| x > threshold).collect()
    }

    /// Returns a new array containing only the elements that satisfy the condition
    pub fn filter(&self, condition: impl Fn(&f64) -> bool) -> Self {
        let data: Vec<f64> = self.data.iter().cloned().filter(condition).collect();
        Self::from_vec(data)
    }
}

impl fmt::Display for NDArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.ndim() == 1 {
            write!(f, "array([{}])", self.data.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", "))
        } else {
            let mut result = String::from("array([");
            for i in 0..self.shape[0] {
                if i > 0 {
                    result.push_str(",\n       ");
                }
                result.push('[');
                let row_start = i * self.shape[1];
                let row_end = row_start + self.shape[1];
                result.push_str(&self.data[row_start..row_end]
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", "));
                result.push(']');
            }
            result.push_str("])");
            write!(f, "{}", result)
        }
    }
}

impl Add<f64> for NDArray {
    type Output = Self;

    fn add(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Add for NDArray {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must be the same for element-wise addition");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Sub<f64> for NDArray {
    type Output = Self;

    fn sub(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x - scalar).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Sub for NDArray {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must be the same for element-wise subtraction");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a - b).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Mul<f64> for NDArray {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Mul for NDArray {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must be the same for element-wise multiplication");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Div<f64> for NDArray {
    type Output = Self;

    fn div(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Div for NDArray {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must be the same for element-wise division");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a / b).collect();
        Self::new(data, self.shape.clone())
    }
}

impl Index<usize> for NDArray {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for NDArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_array_creation() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(arr.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.to_string(), "array([1, 2, 3])");
    }

    #[test]
    fn test_2d_array_creation() {
        let arr = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        assert_eq!(arr.shape(), &[3, 3]);
        assert_eq!(arr.to_string(), "array([[1, 2, 3],\n       [4, 5, 6],\n       [7, 8, 9]])");
    }

    #[test]
    fn test_arange() {
        let arr = NDArray::arange(0.0, 5.0, 1.0);
        assert_eq!(arr.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(arr.to_string(), "array([0, 1, 2, 3, 4])");

        let arr = NDArray::arange(1.0, 11.0, 2.0);
        assert_eq!(arr.data(), &[1.0, 3.0, 5.0, 7.0, 9.0]);
        assert_eq!(arr.to_string(), "array([1, 3, 5, 7, 9])");
    }

    #[test]
    fn test_zeros() {
        let arr = NDArray::zeros(4);
        assert_eq!(arr.data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(arr.to_string(), "array([0, 0, 0, 0])");

        let arr = NDArray::zeros_2d(2, 2);
        assert_eq!(arr.data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(arr.to_string(), "array([[0, 0],\n       [0, 0]])");
    }

    #[test]
    fn test_ones() {
        let arr = NDArray::ones(5);
        assert_eq!(arr.data(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(arr.to_string(), "array([1, 1, 1, 1, 1])");

        let arr = NDArray::ones_2d(3, 3);
        assert_eq!(arr.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(arr.to_string(), "array([[1, 1, 1],\n       [1, 1, 1],\n       [1, 1, 1]])");
    }

    #[test]
    fn test_linspace() {
        let arr = NDArray::linspace(0.0, 1.0, 11, 1);
        let expected = &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        for (a, &e) in arr.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "Value {} is not close to expected {}", a, e);
        }
        
        assert_eq!(arr.to_string(), "array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])");
    }

    #[test]
    fn test_eye() {
        let arr = NDArray::eye(1);
        assert_eq!(arr.data(), &[1.0]);
        assert_eq!(arr.to_string(), "array([[1]])");

        let arr = NDArray::eye(2);
        assert_eq!(arr.data(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(arr.to_string(), "array([[1, 0],\n       [0, 1]])");

        let arr = NDArray::eye(3);
        assert_eq!(arr.data(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(arr.to_string(), "array([[1, 0, 0],\n       [0, 1, 0],\n       [0, 0, 1]])");
    }

    #[test]
    fn test_rand() {
        let arr = NDArray::rand(5);
        assert_eq!(arr.shape(), &[5]);
        // Check that all values are between 0 and 1
        assert!(arr.data().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_rand_2d() {
        let arr = NDArray::rand_2d(2, 3);
        assert_eq!(arr.shape(), &[2, 3]);
        // Check that all values are between 0 and 1
        assert!(arr.data().iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randint() {
        let arr = NDArray::randint(1, 10, 5);
        assert_eq!(arr.shape(), &[5]);
        // Check that all values are between 1 and 9
        assert!(arr.data().iter().all(|&x| x >= 1.0 && x < 10.0));
    }

    #[test]
    fn test_randint_2d() {
        let arr = NDArray::randint_2d(1, 10, 2, 3);
        assert_eq!(arr.shape(), &[2, 3]);
        // Check that all values are between 1 and 9
        assert!(arr.data().iter().all(|&x| x >= 1.0 && x < 10.0));
    }

    #[test]
    fn test_reshape() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(arr.shape(), &[6]);

        let reshaped = arr.reshape(vec![2, 3]);
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.to_string(), "array([[0, 1, 2],\n       [3, 4, 5]])");
    }

    #[test]
    fn test_max_min() {
        let arr = NDArray::from_vec(vec![1.0, -2.0, 3.0, 4.0, 5.0]);
        assert_eq!(arr.max(), 5.0);
        assert_eq!(arr.argmax(), 4);
        assert_eq!(arr.min(), -2.0);
        assert_eq!(arr.argmin(), 1);
    }

    #[test]
    fn test_scalar_addition() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.clone() + 2.0;
        assert_eq!(result.data(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_element_wise_addition() {
        let arr1 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr1 + arr2;
        assert_eq!(result.data(), &[0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scalar_subtraction() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.clone() - 10.0;
        assert_eq!(result.data(), &[-10.0, -9.0, -8.0, -7.0]);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let arr1 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr1 - arr2;
        assert_eq!(result.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.clone() * 6.0;
        assert_eq!(result.data(), &[0.0, 6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_element_wise_multiplication() {
        let arr1 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr1 * arr2;
        assert_eq!(result.data(), &[0.0, 1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_scalar_division() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.clone() / 2.0;
        assert_eq!(result.data(), &[0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_element_wise_division() {
        let arr1 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr1 / arr2;
        assert!(result.data().iter().zip(&[f64::NAN, 1.0, 1.0, 1.0]).all(|(a, &b)| a.is_nan() || (a - b).abs() < 1e-9));
    }

    #[test]
    fn test_sqrt() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.sqrt();
        let expected = &[0.0, 1.0, 1.41421356, 1.73205081];
        for (a, &e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-8, "Value {} is not close to expected {}", a, e);
        }
    }

    #[test]
    fn test_exp() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.exp();
        let expected = &[1.0, 2.71828183, 7.3890561, 20.08553692];
        for (a, &e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-8, "Value {} is not close to expected {}", a, e);
        }
    }

    #[test]
    fn test_sin() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.sin();
        let expected = &[0.0, 0.84147098, 0.90929743, 0.14112001];
        for (a, &e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-8, "Value {} is not close to expected {}", a, e);
        }
    }

    #[test]
    fn test_cos() {
        let arr = NDArray::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let result = arr.cos();
        let expected = &[1.0, 0.54030231, -0.41614684, -0.9899925];
        for (a, &e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-8, "Value {} is not close to expected {}", a, e);
        }
    }

    #[test]
    fn test_ln() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.ln();
        let expected = &[0.0, 0.69314718, 1.09861229];
        for (a, &e) in result.data().iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-8, "Value {} is not close to expected {}", a, e);
        }
    }

    #[test]
    fn test_get_element() {
        let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        assert_eq!(arr.get(0), 0.69);
    }

    #[test]
    fn test_slice() {
        let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        let sliced = arr.slice(1, 4);
        assert_eq!(sliced.data(), &[0.94, 0.66, 0.73]);
    }

    #[test]
    fn test_set_element() {
        let mut arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        arr.set(0, 1.0);
        assert_eq!(arr.get(0), 1.0);
    }

    #[test]
    fn test_index_operator() {
        let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        assert_eq!(arr[0], 0.69);
    }

    #[test]
    fn test_index_mut_operator() {
        let mut arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        arr[0] = 1.0;
        assert_eq!(arr[0], 1.0);
    }

    #[test]
    fn test_single_element_assignment() {
        let mut arr = NDArray::from_vec(vec![0.12, 0.94, 0.66, 0.73, 0.83]);
        arr.set(0, 0.0);
        assert_eq!(arr.data(), &[0.0, 0.94, 0.66, 0.73, 0.83]);
    }

    #[test]
    fn test_range_assignment() {
        let mut arr = NDArray::from_vec(vec![0.12, 0.94, 0.66, 0.73, 0.83]);
        arr.set_range(0, arr.data.len(), 0.0);
        assert_eq!(arr.data(), &[0.0, 0.0, 0.0, 0.0, 0.0]);

        arr.set_range(2, 5, 0.5);
        assert_eq!(arr.data(), &[0.0, 0.0, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_array_referencing() {
        let mut arr = NDArray::from_vec(vec![6.0, 7.0, 8.0, 9.0]);
        {
            let view = arr.view_mut(0, 2);
            view[1] = 4.0;
        }
        assert_eq!(arr.data(), &[6.0, 4.0, 8.0, 9.0]); // Original array is changed
    }

    #[test]
    fn test_array_copying() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let mut copied = arr.copy();
        assert_eq!(copied.data(), &[1.0, 2.0, 3.0]);

        copied.set(0, 9.0);
        assert_eq!(copied.data(), &[9.0, 2.0, 3.0]);
        assert_eq!(arr.data(), &[1.0, 2.0, 3.0]); // Original array remains unchanged
    }

    #[test]
    fn test_2d_indexing() {
        let mat = NDArray::from_matrix(vec![
            vec![5.0, 10.0, 15.0],
            vec![20.0, 25.0, 30.0],
            vec![35.0, 40.0, 45.0],
        ]);
        assert_eq!(mat.get_2d(0, 0), 5.0);
        assert_eq!(mat.get_2d(0, 2), 15.0);
        assert_eq!(mat.get_2d(2, 2), 45.0);
    }

    #[test]
    fn test_2d_set() {
        let mut mat = NDArray::from_matrix(vec![
            vec![5.0, 10.0, 15.0],
            vec![20.0, 25.0, 30.0],
            vec![35.0, 40.0, 45.0],
        ]);
        mat.set_2d(0, 0, 50.0);
        assert_eq!(mat.get_2d(0, 0), 50.0);
    }

    #[test]
    fn test_sub_matrix() {
        let mat = NDArray::from_matrix(vec![
            vec![5.0, 10.0, 15.0],
            vec![20.0, 25.0, 30.0],
            vec![35.0, 40.0, 45.0],
        ]);
        let sub_mat = mat.sub_matrix(1, 3, 0, 3);
        assert_eq!(sub_mat.shape(), &[2, 3]);
        assert_eq!(sub_mat.to_string(), "array([[20, 25, 30],\n       [35, 40, 45]])");
    }

    #[test]
    fn test_greater_than() {
        let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        let result = arr.greater_than(0.7);
        assert_eq!(result, vec![false, true, false, true, true]);
    }

    #[test]
    fn test_filter() {
        let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
        let filtered = arr.filter(|&x| x > 0.7);
        assert_eq!(filtered.data(), &[0.94, 0.73, 0.83]);
    }
}

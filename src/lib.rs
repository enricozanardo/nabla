use rand::Rng;
use rand_distr::StandardNormal;
use std::f64;

mod nab_io;

pub use nab_io::{save_nab, load_nab, savez_nab, loadz_nab};

/// A multi-dimensional array implementation inspired by NumPy's ndarray
#[derive(Debug, Clone)]
pub struct NDArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl NDArray {
    /// Creates a new NDArray with the given data and shape
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values representing the array's data.
    /// * `shape` - A vector of usize values representing the dimensions of the array.
    ///
    /// # Returns
    ///
    /// A new NDArray instance.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    /// Creates a 1D array (vector) from a vector
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values.
    ///
    /// # Returns
    ///
    /// A 1D NDArray.
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    /// Creates a 2D array (matrix) from a vector of vectors
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of vectors of f64 values.
    ///
    /// # Returns
    ///
    /// A 2D NDArray.
    pub fn from_matrix(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data.get(0).map_or(0, |row| row.len());
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        Self::new(flat_data, vec![rows, cols])
    }

    /// Returns the shape of the array
    ///
    /// # Returns
    ///
    /// A slice of usize values representing the dimensions of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions
    ///
    /// # Returns
    ///
    /// The number of dimensions of the array.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns a reference to the underlying data
    ///
    /// # Returns
    ///
    /// A slice of f64 values representing the array's data.
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
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with zeros.
    pub fn zeros(size: usize) -> Self {
        Self::from_vec(vec![0.0; size])
    }

    /// Creates a 2D array (matrix) filled with zeros
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with zeros.
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![0.0; rows * cols], vec![rows, cols])
    }

    /// Creates a 1D array filled with ones
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with ones.
    pub fn ones(size: usize) -> Self {
        Self::from_vec(vec![1.0; size])
    }

    /// Creates a 2D array (matrix) filled with ones
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with ones.
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
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random numbers.
    pub fn rand(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random numbers between 0 and 1
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random numbers.
    pub fn rand_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen()).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Creates a 1D array of random numbers following a normal distribution
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random numbers from a normal distribution.
    pub fn randn(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random numbers following a normal distribution
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random numbers from a normal distribution.
    pub fn randn_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.sample(StandardNormal)).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Creates a 1D array of random integers between `low` and `high`
    ///
    /// # Arguments
    ///
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (exclusive).
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random integers.
    pub fn randint(low: i32, high: i32, size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen_range(low..high) as f64).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random integers between `low` and `high`
    ///
    /// # Arguments
    ///
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (exclusive).
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random integers.
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
    ///
    /// # Returns
    ///
    /// The maximum value as an f64.
    pub fn max(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the maximum value in the array
    ///
    /// # Returns
    ///
    /// The index of the maximum value.
    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Returns the minimum value in the array
    ///
    /// # Returns
    ///
    /// The minimum value as an f64.
    pub fn min(&self) -> f64 {
        *self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the minimum value in the array
    ///
    /// # Returns
    ///
    /// The index of the minimum value.
    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Calculates the square root of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the square root of each element.
    pub fn sqrt(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sqrt()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the exponential (e^x) of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the exponential of each element.
    pub fn exp(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.exp()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the sine of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the sine of each element.
    pub fn sin(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.sin()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the cosine of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the cosine of each element.
    pub fn cos(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.cos()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the natural logarithm of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the natural logarithm of each element.
    pub fn ln(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.ln()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Returns a specific element from the array
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    ///
    /// The element at the specified index.
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Returns a slice of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the slice.
    /// * `end` - The ending index of the slice (exclusive).
    ///
    /// # Returns
    ///
    /// A new NDArray containing the specified slice.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let data = self.data[start..end].to_vec();
        Self::from_vec(data)
    }

    /// Sets a specific element in the array
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to set.
    /// * `value` - The value to set the element to.
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Sets a range of elements in the array to a specific value
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the range.
    /// * `end` - The ending index of the range (exclusive).
    /// * `value` - The value to set the elements to.
    pub fn set_range(&mut self, start: usize, end: usize, value: f64) {
        for i in start..end {
            self.data[i] = value;
        }
    }

    /// Returns a copy of the array
    ///
    /// # Returns
    ///
    /// A new NDArray that is a copy of the original.
    pub fn copy(&self) -> Self {
        Self::new(self.data.clone(), self.shape.clone())
    }

    /// Returns a view (slice) of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the view.
    /// * `end` - The ending index of the view (exclusive).
    ///
    /// # Returns
    ///
    /// A slice of f64 values representing the specified view.
    pub fn view(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start..end]
    }

    /// Returns a mutable view (slice) of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the view.
    /// * `end` - The ending index of the view (exclusive).
    ///
    /// # Returns
    ///
    /// A mutable slice of f64 values representing the specified view.
    pub fn view_mut(&mut self, start: usize, end: usize) -> &mut [f64] {
        &mut self.data[start..end]
    }

    /// Returns a specific element from a 2D array
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// The element at the specified row and column.
    pub fn get_2d(&self, row: usize, col: usize) -> f64 {
        assert_eq!(self.ndim(), 2, "get_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col]
    }

    /// Sets a specific element in a 2D array
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    /// * `value` - The value to set the element to.
    pub fn set_2d(&mut self, row: usize, col: usize, value: f64) {
        assert_eq!(self.ndim(), 2, "set_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col] = value;
    }

    /// Returns a sub-matrix from a 2D array
    ///
    /// # Arguments
    ///
    /// * `row_start` - The starting row index of the sub-matrix.
    /// * `row_end` - The ending row index of the sub-matrix (exclusive).
    /// * `col_start` - The starting column index of the sub-matrix.
    /// * `col_end` - The ending column index of the sub-matrix (exclusive).
    ///
    /// # Returns
    ///
    /// A new NDArray representing the specified sub-matrix.
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
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold value to compare each element against.
    ///
    /// # Returns
    ///
    /// A vector of boolean values indicating whether each element is greater than the threshold.
    pub fn greater_than(&self, threshold: f64) -> Vec<bool> {
        self.data.iter().map(|&x| x > threshold).collect()
    }

    /// Returns a new array containing only the elements that satisfy the condition
    ///
    /// # Arguments
    ///
    /// * `condition` - A closure that takes an f64 and returns a boolean.
    ///
    /// # Returns
    ///
    /// A new NDArray containing only the elements that satisfy the condition.
    pub fn filter(&self, condition: impl Fn(&f64) -> bool) -> Self {
        let data: Vec<f64> = self.data.iter().cloned().filter(condition).collect();
        Self::from_vec(data)
    }

    /// Returns the total number of elements in the array
    ///
    /// # Returns
    ///
    /// The total number of elements in the array.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the data type of the elements in the array
    ///
    /// # Returns
    ///
    /// A string representing the data type of the elements.
    pub fn dtype(&self) -> &'static str {
        "f64" // Since we're using f64 for all elements
    }

    /// Adds a new axis to the array at the specified position
    ///
    /// # Arguments
    ///
    /// * `axis` - The position at which to add the new axis.
    ///
    /// # Returns
    ///
    /// A new NDArray with an additional axis.
    pub fn new_axis(&self, axis: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Self::new(self.data.clone(), new_shape)
    }

    /// Expands the dimensions of the array by adding a new axis at the specified index
    ///
    /// # Arguments
    ///
    /// * `axis` - The index at which to add the new axis.
    ///
    /// # Returns
    ///
    /// A new NDArray with expanded dimensions.
    pub fn expand_dims(&self, axis: usize) -> Self {
        self.new_axis(axis)
    }

    /// Calculates the hyperbolic tangent of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the hyperbolic tangent of each element.
    pub fn tanh(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.tanh()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Applies the ReLU function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the ReLU function applied to each element.
    pub fn relu(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Self::new(data, self.shape.clone())
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
        let data: Vec<f64> = self.data.iter().map(|&x| if x > 0.0 { x } else { alpha * x }).collect();
        Self::new(data, self.shape.clone())
    }

    /// Applies the Sigmoid function to each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the Sigmoid function applied to each element.
    pub fn sigmoid(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| 1.0_f64 / (1.0_f64 + (-x).exp())).collect();
        Self::new(data, self.shape.clone())
    }

    /// Calculates the Mean Squared Error (MSE) between two arrays
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values as an NDArray.
    /// * `y_pred` - The predicted values as an NDArray.
    ///
    /// # Returns
    ///
    /// The MSE as a f64.
    pub fn mean_squared_error(y_true: &NDArray, y_pred: &NDArray) -> f64 {
        assert_eq!(y_true.shape(), y_pred.shape(), "Shapes of y_true and y_pred must match");
        let diff = y_true.data.iter().zip(y_pred.data.iter()).map(|(t, p)| (t - p).powi(2)).collect::<Vec<f64>>();
        diff.iter().sum::<f64>() / y_true.data.len() as f64
    }

    /// Calculates the Cross-Entropy Loss between two arrays
    ///
    /// # Arguments
    ///
    /// * `y_true` - The true values as an NDArray (one-hot encoded).
    /// * `y_pred` - The predicted probabilities as an NDArray.
    ///
    /// # Returns
    ///
    /// The Cross-Entropy Loss as a f64.
    pub fn cross_entropy_loss(y_true: &NDArray, y_pred: &NDArray) -> f64 {
        assert_eq!(y_true.shape(), y_pred.shape(), "Shapes of y_true and y_pred must match");
        let epsilon = 1e-8;
        let clipped_pred = y_pred.data.iter().map(|&p| p.clamp(epsilon, 1.0 - epsilon)).collect::<Vec<f64>>();
        let loss = y_true.data.iter().zip(clipped_pred.iter()).map(|(t, p)| t * p.ln()).collect::<Vec<f64>>();
        -loss.iter().sum::<f64>() / y_true.shape()[0] as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ndarray() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let array = NDArray::new(data.clone(), shape.clone());
        assert_eq!(array.data(), &data);
        assert_eq!(array.shape(), &shape);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let array = NDArray::from_vec(data.clone());
        assert_eq!(array.data(), &data);
        assert_eq!(array.shape(), &[3]);
    }

    #[test]
    fn test_arange() {
        let array = NDArray::arange(0.0, 5.0, 1.0);
        assert_eq!(array.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_save_and_load_nab() -> std::io::Result<()> {
        let array = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        save_nab("test.nab", &array)?;
        let loaded_array = load_nab("test.nab")?;
        assert_eq!(array.data(), loaded_array.data());
        assert_eq!(array.shape(), loaded_array.shape());

        // Clean up test file
        std::fs::remove_file("test.nab")?;
        Ok(())
    }

    #[test]
    fn test_savez_and_loadz_nab() -> std::io::Result<()> {
        let array1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let array2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        savez_nab("test_multiple.nab", vec![("x", &array1), ("y", &array2)])?;

        let arrays = loadz_nab("test_multiple.nab")?;
        assert_eq!(arrays.get("x").unwrap().data(), array1.data());
        assert_eq!(arrays.get("y").unwrap().data(), array2.data());

        // Clean up test file
        std::fs::remove_file("test_multiple.nab")?;
        Ok(())
    }

    #[test]
    fn test_mean_squared_error() {
        let y_true = NDArray::from_vec(vec![1.0, 0.0, 1.0, 1.0]);
        let y_pred = NDArray::from_vec(vec![0.9, 0.2, 0.8, 0.6]);
        let mse = NDArray::mean_squared_error(&y_true, &y_pred);
        assert!((mse - 0.0625).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let y_true = NDArray::from_matrix(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let y_pred = NDArray::from_matrix(vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.05, 0.15, 0.8],
        ]);
        let cross_entropy = NDArray::cross_entropy_loss(&y_true, &y_pred);
        assert!((cross_entropy - 0.267654016).abs() < 1e-4);
    }
}

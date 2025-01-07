use rand::Rng;
use rand_distr::StandardNormal;
use std::ops::{Add, Sub, Mul};

#[derive(Debug, Clone)]
pub struct NDArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl NDArray {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    #[allow(dead_code)]
    pub fn from_matrix(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data.get(0).map_or(0, |row| row.len());
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        Self::new(flat_data, vec![rows, cols])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns a reference to the data of the array
    pub fn data(&self) -> &[f64] {
        &self.data
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn max(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the maximum value in the array
    ///
    /// # Returns
    ///
    /// The index of the maximum value.
    #[allow(dead_code)]
    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Returns the minimum value in the array
    ///
    /// # Returns
    ///
    /// The minimum value as an f64.
    #[allow(dead_code)]
    pub fn min(&self) -> f64 {
        *self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Creates an NDArray from a flat vector and a specified shape
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values representing the array's data.
    /// * `shape` - A vector of usize values representing the dimensions of the array.
    ///
    /// # Returns
    ///
    /// A new NDArray instance.
    #[allow(dead_code)]
    pub fn from_vec_reshape(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    /// Extracts a single sample from a batch of N-dimensional arrays
    ///
    /// # Arguments
    ///
    /// * `sample_index` - The index of the sample to extract
    ///
    /// # Returns
    ///
    /// A new NDArray containing just the specified sample with N-1 dimensions
    #[allow(dead_code)]
    pub fn extract_sample(&self, sample_index: usize) -> Self {
        assert!(self.ndim() >= 2, "Array must have at least 2 dimensions");
        assert!(sample_index < self.shape[0], "Sample index out of bounds");

        let sample_size: usize = self.shape.iter().skip(1).product();
        let start_index = sample_index * sample_size;
        let end_index = start_index + sample_size;
        
        // Create new shape without the first dimension
        let new_shape: Vec<usize> = self.shape.iter().skip(1).cloned().collect();
        
        NDArray::new(
            self.data[start_index..end_index].to_vec(),
            new_shape
        )
    }

    /// Pretty prints an N-dimensional array
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places to round each value to.
    #[allow(dead_code)]
    pub fn pretty_print(&self, precision: usize) {
        let indent_str = " ".repeat(precision);
        
        let format_value = |x: f64| -> String {
            if x == 0.0 {
                format!("{:.1}", x)
            } else {
                format!("{:.*}", precision, x)
            }
        };
        
        match self.ndim() {
            1 => println!("{}[{}]", indent_str, self.data.iter()
                .map(|&x| format_value(x))
                .collect::<Vec<_>>()
                .join(" ")),
                
            2 => {
                println!("{}[", indent_str);
                for i in 0..self.shape[0] {
                    print!("{}  [", indent_str);
                    for j in 0..self.shape[1] {
                        print!("{}", format_value(self.get_2d(i, j)));
                        if j < self.shape[1] - 1 {
                            print!(" ");
                        }
                    }
                    println!("]");
                }
                println!("{}]", indent_str);
            },
            
            _ => {
                println!("{}[", indent_str);
                for i in 0..self.shape[0] {
                    let slice = self.extract_sample(i);
                    slice.pretty_print(precision + 2);
                }
                println!("{}]", indent_str);
            }
        }
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
    #[allow(dead_code)]
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn rand(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        Self::from_vec(data)
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
    #[allow(dead_code)]
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

    /// Sets a specific element in the array
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to set.
    /// * `value` - The value to set the element to.
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn set_2d(&mut self, row: usize, col: usize, value: f64) {
        assert_eq!(self.ndim(), 2, "set_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col] = value;
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn expand_dims(&self, axis: usize) -> Self {
        self.new_axis(axis)
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn filter(&self, condition: impl Fn(&f64) -> bool) -> Self {
        let data: Vec<f64> = self.data.iter().cloned().filter(condition).collect();
        Self::from_vec(data)
    }


    /// Returns the data type of the elements in the array
    ///
    /// # Returns
    ///
    /// A string representing the data type of the elements.
    #[allow(dead_code)]
    pub fn dtype(&self) -> &'static str {
        "f64" // Since we're using f64 for all elements
    }

    /// Returns the total number of elements in the array
    ///
    /// # Returns
    ///
    /// The total number of elements in the array.
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the index of the minimum value in the array
    ///
    /// # Returns
    ///
    /// The index of the minimum value.
    #[allow(dead_code)]
    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
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
    #[allow(dead_code)]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let data = self.data[start..end].to_vec();
        Self::from_vec(data)
    }


}

impl Add for NDArray {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise addition");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Add<f64> for NDArray {
    type Output = Self;

    fn add(self, scalar: f64) -> Self::Output {
        let data = self.data.iter().map(|a| a + scalar).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Sub for NDArray {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise subtraction");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Mul<f64> for NDArray {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let data = self.data.iter().map(|a| a * scalar).collect();
        NDArray::new(data, self.shape.clone())
    }
}

// Add std::fmt::Display implementation for convenient printing
impl std::fmt::Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.display())
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
    fn test_element_wise_addition() {
        let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        let sum = arr1.clone() + arr2;
        assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = arr.clone() * 2.0;
        assert_eq!(scaled.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let arr1 = NDArray::from_vec(vec![5.0, 7.0, 9.0]);
        let arr2 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let diff = arr1 - arr2;
        assert_eq!(diff.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scalar_addition() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr + 1.0;
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_combined_operations() {
        let X = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let theta_1 = 2.0;
        let theta_0 = 1.0;
        let predictions = X.clone() * theta_1 + theta_0;
        assert_eq!(predictions.data(), &[3.0, 5.0, 7.0]);
    }

} 
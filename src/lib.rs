use std::fmt;

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
}

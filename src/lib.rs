/// A struct representing a multi-dimensional array.
#[derive(Debug)]
pub struct Array {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Array {
    /// Creates a new `Array` with the given data and shape.
    ///
    /// # Panics
    ///
    /// Panics if the data length does not match the shape dimensions.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        Array { data, shape }
    }

    /// Creates a new `Array` filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Array {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new `Array` filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Array {
            data: vec![1.0; size],
            shape,
        }
    }

    /// Adds two arrays element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the arrays do not have the same shape.
    pub fn add(&self, other: &Array) -> Array {
        assert_eq!(self.shape, other.shape, "Arrays must have same shape");
        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Array::new(new_data, self.shape.clone())
    }

    /// Multiplies two arrays element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the arrays do not have the same shape.
    pub fn multiply(&self, other: &Array) -> Array {
        assert_eq!(self.shape, other.shape, "Arrays must have same shape");
        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Array::new(new_data, self.shape.clone())
    }

    /// Computes the mean of the array elements.
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.data.iter().sum();
        sum / (self.data.len() as f64)
    }
}
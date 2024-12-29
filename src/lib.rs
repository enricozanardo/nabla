use std::cmp::Ordering;

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
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the dimensions of the array.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Array {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new `Array` filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the dimensions of the array.
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
    ///
    /// # Arguments
    ///
    /// * `other` - Another array to add to this array.
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
    ///
    /// # Arguments
    ///
    /// * `other` - Another array to multiply with this array.
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

    /// Computes the dot product of two arrays.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are not aligned for dot product.
    ///
    /// # Arguments
    ///
    /// * `other` - Another array to compute the dot product with.
    pub fn dot(&self, other: &Array) -> f64 {
        assert_eq!(self.shape[1], other.shape[0], "Shapes are not aligned for dot product");
        self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Computes the mean of the array elements.
    pub fn mean(&self) -> f64 {
        let sum: f64 = self.data.iter().sum();
        sum / (self.data.len() as f64)
    }
}

/// Computes the score of a linear model.
///
/// # Arguments
///
/// * `weights` - The weights of the model.
/// * `bias` - The bias of the model.
/// * `features` - The input features.
pub fn score_linear(weights: &Array, bias: f64, features: &Array) -> f64 {
    features.dot(weights) + bias
}

/// Applies the step function to a value.
///
/// # Arguments
///
/// * `x` - The input value.
pub fn step(x: f64) -> i32 {
    match x.partial_cmp(&0.0).unwrap_or(Ordering::Less) {
        Ordering::Greater | Ordering::Equal => 1,
        Ordering::Less => 0,
    }
}

/// Predicts the label for a given set of features using a linear model.
///
/// # Arguments
///
/// * `weights` - The weights of the model.
/// * `bias` - The bias of the model.
/// * `features` - The input features.
pub fn prediction_linear(weights: &Array, bias: f64, features: &Array) -> i32 {
    step(score_linear(weights, bias, features))
}

/// Computes the error of a prediction.
///
/// # Arguments
///
/// * `weights` - The weights of the model.
/// * `bias` - The bias of the model.
/// * `features` - The input features.
/// * `label` - The true label.
pub fn error_linear(weights: &Array, bias: f64, features: &Array, label: i32) -> f64 {
    let pred = prediction_linear(weights, bias, features);
    if pred == label {
        0.0
    } else {
        (score_linear(weights, bias, features) - label as f64).abs()
    }
}

/// Computes the mean perceptron error over a dataset.
///
/// # Arguments
///
/// * `weights` - The weights of the model.
/// * `bias` - The bias of the model.
/// * `features` - The input features.
/// * `labels` - The true labels.
pub fn mean_perceptron_error(weights: &Array, bias: f64, features: &Array, labels: &Array) -> f64 {
    let mut total_error = 0.0;
    for i in 0..features.shape[0] {
        let feature = Array::new(features.data[i * features.shape[1]..(i + 1) * features.shape[1]].to_vec(), vec![features.shape[1]]);
        total_error += error_linear(weights, bias, &feature, labels.data[i] as i32);
    }
    total_error / features.shape[0] as f64
}
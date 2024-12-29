use std::cmp::Ordering;
use std::ops::Index;

/// A struct representing a multi-dimensional tensor.
#[derive(Debug)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Index<usize> for Tensor {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        let row_length = self.shape[1];
        let start = index * row_length;
        let end = start + row_length;
        &self.data[start..end]
    }
}

impl Tensor {
    /// Creates a new `Tensor` with the given data and shape.
    ///
    /// # Panics
    ///
    /// Panics if the data length does not match the shape dimensions.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        Tensor { data, shape }
    }

    pub fn len(&self) -> usize {
        self.shape[0]
    }

    /// Creates a new `Tensor` filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the dimensions of the tensor.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new `Tensor` filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the dimensions of the tensor.
    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; size],
            shape,
        }
    }

    /// Adds two tensors element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the tensors do not have the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another tensor to add to this tensor.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensors must have same shape");
        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Tensor::new(new_data, self.shape.clone())
    }

    /// Multiplies two tensors element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the tensors do not have the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another tensor to multiply with this tensor.
    pub fn multiply(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Tensors must have same shape");
        let new_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Tensor::new(new_data, self.shape.clone())
    }

    /// Multiplies the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply with.
    pub fn multiply_scalar(&self, scalar: f64) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(new_data, self.shape.clone())
    }

    /// Computes the dot product of two tensors.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are not aligned for dot product.
    ///
    /// # Arguments
    ///
    /// * `other` - Another tensor to compute the dot product with.
    pub fn dot(&self, other: &Tensor) -> f64 {
        assert_eq!(self.shape[1], other.shape[0], "Shapes are not aligned for dot product");
        self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Computes the mean of the tensor elements.
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
pub fn score_linear(weights: &Tensor, bias: f64, features: &Tensor) -> f64 {
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
pub fn prediction_linear(weights: &Tensor, bias: f64, features: &Tensor) -> i32 {
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
pub fn error_linear(weights: &Tensor, bias: f64, features: &Tensor, label: i32) -> f64 {
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
pub fn mean_perceptron_error(weights: &Tensor, bias: f64, features: &Tensor, labels: &Tensor) -> f64 {
    let mut total_error = 0.0;
    for i in 0..features.shape[0] {
        let feature = Tensor::new(features.data[i * features.shape[1]..(i + 1) * features.shape[1]].to_vec(), vec![features.shape[1]]);
        total_error += error_linear(weights, bias, &feature, labels.data[i] as i32);
    }
    total_error / features.shape[0] as f64
}
use crate::nab_array::NDArray;

pub trait Loss {
    fn forward(&self, predictions: &NDArray, targets: &NDArray) -> f64;
    fn backward(&self, predictions: &NDArray, targets: &NDArray) -> NDArray;
}

pub struct CategoricalCrossentropy;

impl Loss for CategoricalCrossentropy {
    fn forward(&self, y_pred: &NDArray, y_true: &NDArray) -> f64 {
        // Compute cross-entropy loss
        let epsilon = 1e-8;
        let clipped_pred = y_pred.clip(epsilon, 1.0 - epsilon);
        -y_true.multiply(&clipped_pred.log()).sum() / y_true.shape()[0] as f64
    }

    fn backward(&self, y_pred: &NDArray, y_true: &NDArray) -> NDArray {
        NDArray::crossentropy_nabla(y_pred, y_true)
    }
}

impl NDArray {
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
    #[allow(dead_code)]
    pub fn mean_squared_error(y_true: &NDArray, y_pred: &NDArray) -> f64 {
        assert_eq!(y_true.shape(), y_pred.shape(), "Shapes of y_true and y_pred must match");
        let diff = y_true.data().iter().zip(y_pred.data().iter()).map(|(t, p)| (t - p).powi(2)).collect::<Vec<f64>>();
        diff.iter().sum::<f64>() / y_true.data().len() as f64
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
    #[allow(dead_code)]
    pub fn cross_entropy_loss(y_true: &NDArray, y_pred: &NDArray) -> f64 {
        assert_eq!(y_true.shape(), y_pred.shape(), "Shapes of y_true and y_pred must match");
        let epsilon = 1e-8;
        let clipped_pred = y_pred.data().iter().map(|&p| p.clamp(epsilon, 1.0 - epsilon)).collect::<Vec<f64>>();
        let loss = y_true.data().iter().zip(clipped_pred.iter()).map(|(t, p)| t * p.ln()).collect::<Vec<f64>>();
        -loss.iter().sum::<f64>() / y_true.shape()[0] as f64
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_mean_squared_error() {
        let y_true = NDArray::from_vec(vec![1.0, 0.0, 1.0, 1.0]);
        let y_pred = NDArray::from_vec(vec![0.9, 0.2, 0.8, 0.6]);
        let mse = NDArray::mean_squared_error(&y_true, &y_pred);
        assert!((mse - 0.0625).abs() < 1e-4);
    }

    #[test]
    pub fn test_cross_entropy_loss() {
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
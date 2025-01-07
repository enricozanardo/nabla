use crate::nab_array::NDArray;

impl NDArray {
    /// Calculates the gradients (nabla) for linear regression with multiple features
    ///
    /// # Arguments
    ///
    /// * `X` - The input feature matrix
    /// * `y` - The actual target values
    /// * `y_pred` - The predicted values
    /// * `N` - The number of samples
    ///
    /// # Returns
    ///
    /// A vector containing the gradients for each parameter
    #[allow(non_snake_case)]
    #[allow(dead_code)]
    pub fn nabla(X: &NDArray, y: &NDArray, y_pred: &NDArray, N: usize) -> Vec<f64> {
        let mut gradients = vec![0.0; X.shape()[1] + 1]; // +1 for the intercept
        let errors: Vec<f64> = y.data().iter().zip(y_pred.data().iter()).map(|(&t, &p)| t - p).collect();

        // Gradient for the intercept
        gradients[0] = -(2.0 / N as f64) * errors.iter().sum::<f64>();

        // Gradients for the features
        for j in 0..X.shape()[1] {
            gradients[j + 1] = -(2.0 / N as f64) * X.data().iter().skip(j).step_by(X.shape()[1]).zip(errors.iter()).map(|(&x, &e)| x * e).sum::<f64>();
        }

        gradients
    }
}
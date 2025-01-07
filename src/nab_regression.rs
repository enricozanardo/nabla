use crate::nab_array::NDArray;



impl NDArray {
    /// Performs linear regression using gradient descent with multiple features
    ///
    /// # Arguments
    ///
    /// * `X` - The input feature matrix as an NDArray.
    /// * `y` - The output target as an NDArray.
    /// * `alpha` - The learning rate.
    /// * `epochs` - The number of iterations for gradient descent.
    ///
    /// # Returns
    ///
    /// A tuple containing the optimized parameters and the history of MSE for each epoch.
    #[allow(non_snake_case)]
    pub fn linear_regression(X: &NDArray, y: &NDArray, alpha: f64, epochs: usize) -> (Vec<f64>, Vec<f64>) {
        let N = X.shape()[0];
        let mut theta = vec![0.0; X.shape()[1] + 1]; // +1 for the intercept
        let mut history = Vec::with_capacity(epochs);

        for _ in 0..epochs {
            // Predictions
            let y_pred: Vec<f64> = (0..N).map(|i| {
                theta[0] + X.data().iter().skip(i * X.shape()[1]).take(X.shape()[1]).zip(&theta[1..]).map(|(&x, &t)| x * t).sum::<f64>()
            }).collect();

            // Calculate MSE
            let mse = NDArray::mean_squared_error(y, &NDArray::from_vec(y_pred.clone()));
            history.push(mse);

            // Calculate gradients using nabla
            let gradients = Self::nabla(X, y, &NDArray::from_vec(y_pred), N);

            // Update parameters
            for j in 0..theta.len() {
                theta[j] -= alpha * gradients[j];
            }
        }

        (theta, history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    #[allow(non_snake_case)]
    fn test_linear_regression() {
        let mut rng = rand::thread_rng();
        let X = NDArray::from_matrix((0..100).map(|_| vec![2.0 * rng.gen::<f64>()]).collect());
        let y = NDArray::from_vec(X.data().iter().map(|&x| 4.0 + 3.0 * x + rng.gen::<f64>()).collect());

        let (theta, history) = NDArray::linear_regression(&X, &y, 0.01, 2000);

        assert!((theta[0] - 4.0).abs() < 1.0);
        assert!((theta[1] - 3.0).abs() < 1.0);
        assert!(history.first().unwrap() > history.last().unwrap());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_linear_regression_multiple_features() {
        // Generate a simple dataset with two features
        let X = NDArray::from_matrix(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 2.0],
        ]);
        let y = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]); // y = 1 + 1*x1 + 2*x2

        // Apply linear regression
        let (theta, history) = NDArray::linear_regression(&X, &y, 0.01, 1000);

        println!("{:?}", theta[0]);
        println!("{:?}", theta[1]);
        println!("{:?}", theta[2]);

        // Check if the parameters are close to the expected values
        assert!((theta[0] - 1.0).abs() < 0.1);  // Increased tolerance
        assert!((theta[1] - 1.0).abs() < 0.1);  // Coefficient for x1
        assert!((theta[2] - 2.0).abs() < 0.1);  // Coefficient for x2

        // Ensure the loss decreases over time
        assert!(history.first().unwrap() > history.last().unwrap());
    }

} 
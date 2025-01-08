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

    /// Calculates gradients for neural network layers
    /// 
    /// # Arguments
    /// 
    /// * `input` - Input to the layer
    /// * `weights` - Layer weights
    /// * `output_gradient` - Gradient from the next layer
    /// 
    /// # Returns
    /// 
    /// A tuple containing (weight_gradients, input_gradients)
    pub fn neural_nabla(
        input: &NDArray,
        weights: &NDArray,
        output_gradient: &NDArray,
    ) -> (NDArray, NDArray) {
        // Weight gradients: input.T @ output_gradient
        let weight_gradients = input.transpose().dot(output_gradient);
        
        // Input gradients: output_gradient @ weights.T
        let input_gradients = output_gradient.dot(&weights.transpose());
        
        (weight_gradients, input_gradients)
    }

    /// Calculates gradients for activation functions
    /// 
    /// # Arguments
    /// 
    /// * `output_gradient` - Gradient from the next layer
    /// * `output` - Output of the activation function
    /// * `activation_type` - Type of activation function
    /// 
    /// # Returns
    /// 
    /// Gradient for the activation function
    pub fn activation_nabla(
        output_gradient: &NDArray,
        output: &NDArray,
        activation_type: ActivationType,
    ) -> NDArray {
        match activation_type {
            ActivationType::Sigmoid => {
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                let sigmoid_derivative = output.multiply(&output.scalar_sub(1.0).multiply_scalar(-1.0));
                output_gradient.multiply(&sigmoid_derivative)
            },
            ActivationType::ReLU => {
                // relu'(x) = 1 if x > 0 else 0
                let relu_derivative = output.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
                output_gradient.multiply(&relu_derivative)
            },
            ActivationType::LeakyReLU => {
                // leaky_relu'(x) = 1 if x > 0 else alpha
                let relu_derivative = output.map(|x| if x > 0.0 { 1.0 } else { 0.01 });
                output_gradient.multiply(&relu_derivative)
            },
            ActivationType::Softmax => {
                // For softmax, we typically combine it with cross-entropy loss,
                // which simplifies the gradient calculation
                output_gradient.clone()
            },
            ActivationType::Tanh => {
                // tanh'(x) = 1 - tanh²(x)
                let tanh_derivative = output.multiply(&output)
                    .scalar_sub(1.0)
                    .multiply_scalar(-1.0);
                output_gradient.multiply(&tanh_derivative)
            },
        }
    }

    /// Calculates gradients for categorical cross-entropy loss
    /// 
    /// # Arguments
    /// 
    /// * `y_pred` - Predicted probabilities
    /// * `y_true` - True labels (one-hot encoded)
    /// 
    /// # Returns
    /// 
    /// Gradient for the loss function
    pub fn crossentropy_nabla(y_pred: &NDArray, y_true: &NDArray) -> NDArray {
        let epsilon = 1e-8;
        let batch_size = y_true.shape()[0] as f64;
        
        // Clip predictions to avoid numerical instability
        let clipped_pred = y_pred.clip(epsilon, 1.0 - epsilon);
        
        // Gradient: -(y_true / y_pred) / batch_size
        y_true.divide(&clipped_pred)
            .multiply_scalar(-1.0)
            .divide_scalar(batch_size)
    }

    /// Applies a function to each element in the array
    ///
    /// # Arguments
    ///
    /// * `f` - A function to apply to each element.
    ///
    /// # Returns
    ///
    /// A new NDArray with the function applied to each element.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        NDArray::new(data, self.shape.clone())
    }
}

#[derive(Clone, Copy)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Softmax,
    LeakyReLU,
    Tanh,
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_nabla() {
        let input = NDArray::from_matrix(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let weights = NDArray::from_matrix(vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ]);
        let output_gradient = NDArray::from_matrix(vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ]);

        let (weight_grads, input_grads) = NDArray::neural_nabla(&input, &weights, &output_gradient);

        assert_eq!(weight_grads.shape(), weights.shape());
        assert_eq!(input_grads.shape(), input.shape());
    }

    #[test]
    fn test_activation_nabla() {
        let output = NDArray::from_vec(vec![0.5, 0.2, 0.7]);
        let output_gradient = NDArray::from_vec(vec![1.0, 1.0, 1.0]);

        let sigmoid_grad = NDArray::activation_nabla(
            &output_gradient, 
            &output, 
            ActivationType::Sigmoid
        );
        
        assert_eq!(sigmoid_grad.shape(), output.shape());
    }

    #[test]
    fn test_crossentropy_nabla() {
        let y_true = NDArray::from_matrix(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]);
        let y_pred = NDArray::from_matrix(vec![
            vec![0.7, 0.3],
            vec![0.2, 0.8],
        ]);

        let gradient = NDArray::crossentropy_nabla(&y_pred, &y_true);
        
        assert_eq!(gradient.shape(), y_pred.shape());
    }
}
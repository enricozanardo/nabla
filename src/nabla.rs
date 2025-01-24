use crate::nab_array::NDArray;


/// Nabla is where all the magic happens and where all the gradients are calculated.
pub struct Nabla;


/// Calculates the gradients (nabla) for linear regression with multiple features
///
/// # Arguments
///
/// * `X` - The input feature matrix as an NDArray
/// * `y` - The actual target values as an NDArray
/// * `y_pred` - The predicted values as an NDArray 
/// * `N` - The number of samples
///
/// # Returns
///
/// A vector containing the gradients for each parameter, including intercept
#[allow(non_snake_case)]
impl Nabla {
    pub fn linear_regression_gradients(X: &NDArray, y: &NDArray, y_pred: &NDArray, N: usize) -> Vec<f64> {
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



//     /// Calculates gradients for neural network layers
//     /// 
//     /// # Arguments
//     /// 
//     /// * `input` - Input to the layer
//     /// * `weights` - Layer weights
//     /// * `output_gradient` - Gradient from the next layer
//     /// 
//     /// # Returns
//     /// 
//     /// A tuple containing (weight_gradients, input_gradients)
//     pub fn neural_nabla(
//         input: &NDArray,
//         weights: &NDArray,
//         output_gradient: &NDArray,
//     ) -> (NDArray, NDArray) {
//         // Weight gradients: input.T @ output_gradient
//         let weight_gradients = input.transpose().dot(output_gradient);
        
//         // Input gradients: output_gradient @ weights.T
//         let input_gradients = output_gradient.dot(&weights.transpose());
        
//         (weight_gradients, input_gradients)
//     }

//     /// Calculates gradients for activation functions
//     /// 
//     /// # Arguments
//     /// 
//     /// * `output_gradient` - Gradient from the next layer
//     /// * `output` - Output of the activation function
//     /// * `activation_type` - Type of activation function
//     /// 
//     /// # Returns
//     /// 
//     /// Gradient for the activation function
//     pub fn activation_nabla(
//         output_gradient: &NDArray,
//         output: &NDArray,
//         activation_type: ActivationType,
//     ) -> NDArray {
//         match activation_type {
//             ActivationType::Sigmoid => {
//                 // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
//                 let sigmoid_derivative = output.multiply(&output.scalar_sub(1.0).multiply_scalar(-1.0));
//                 output_gradient.multiply(&sigmoid_derivative)
//             },
//             ActivationType::ReLU => {
//                 // relu'(x) = 1 if x > 0 else 0
//                 let relu_derivative = output.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
//                 output_gradient.multiply(&relu_derivative)
//             },
//             ActivationType::LeakyReLU => {
//                 // leaky_relu'(x) = 1 if x > 0 else alpha
//                 let relu_derivative = output.map(|x| if x > 0.0 { 1.0 } else { 0.01 });
//                 output_gradient.multiply(&relu_derivative)
//             },
//             ActivationType::Softmax => {
//                 // For softmax, we typically combine it with cross-entropy loss,
//                 // which simplifies the gradient calculation
//                 output_gradient.clone()
//             },
//             ActivationType::Tanh => {
//                 // tanh'(x) = 1 - tanh²(x)
//                 let tanh_derivative = output.multiply(&output)
//                     .scalar_sub(1.0)
//                     .multiply_scalar(-1.0);
//                 output_gradient.multiply(&tanh_derivative)
//             },
//         }
//     }

//     /// Calculates gradients for categorical cross-entropy loss
//     /// 
//     /// # Arguments
//     /// 
//     /// * `y_pred` - Predicted probabilities
//     /// * `y_true` - True labels (one-hot encoded)
//     /// 
//     /// # Returns
//     /// 
//     /// Gradient for the loss function
//     pub fn crossentropy_nabla(y_pred: &NDArray, y_true: &NDArray) -> NDArray {
//         let epsilon = 1e-8;
//         let batch_size = y_true.shape()[0] as f64;
        
//         // Clip predictions to avoid numerical instability
//         let clipped_pred = y_pred.clip(epsilon, 1.0 - epsilon);
        
//         // Gradient: -(y_true / y_pred) / batch_size
//         y_true.divide(&clipped_pred)
//             .multiply_scalar(-1.0)
//             .divide_scalar(batch_size)
//     }

//     /// Applies a function to each element in the array
//     ///
//     /// # Arguments
//     ///
//     /// * `f` - A function to apply to each element.
//     ///
//     /// # Returns
//     ///
//     /// A new NDArray with the function applied to each element.
//     pub fn map<F>(&self, f: F) -> Self
//     where
//         F: Fn(f64) -> f64,
//     {
//         let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
//         NDArray::new(data, self.shape.clone())
//     }
// }

// #[derive(Clone, Copy)]
// pub enum ActivationType {
//     Sigmoid,
//     ReLU,
//     Softmax,
//     LeakyReLU,
//     Tanh,
// }

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;

    /// Tests the linear_regression_nabla function
    #[test]
    fn test_linear_regression_nabla() {
        let X = NDArray::from_matrix(vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
        ]);
        let y = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = NDArray::from_vec(vec![1.5, 2.5, 3.5]);
        let N = 3;

        let gradients = Nabla::linear_regression_gradients(&X, &y, &y_pred, N);

        // Correct expected gradients based on calculation
        let expected_gradients = vec![1.0, 2.0, 3.0];

        assert_eq!(gradients.len(), expected_gradients.len());
        for (calculated, expected) in gradients.iter().zip(expected_gradients.iter()) {
            assert!((calculated - expected).abs() < 1e-4);
        }
    }
}
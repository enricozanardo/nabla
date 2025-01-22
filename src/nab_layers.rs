use crate::nab_array::NDArray;
use crate::nab_activations::Activation;

// pub trait Layer {
//     fn forward(&mut self, input: &NDArray) -> NDArray;
//     fn backward(&mut self, gradient: &NDArray) -> NDArray;
//     fn update(&mut self, learning_rate: f64);
//     fn get_weights_mut(&mut self) -> &mut NDArray;
//     fn get_gradients(&self) -> Option<(NDArray, NDArray)>;
//     fn output_shape(&self) -> Vec<usize>;
//     fn parameter_count(&self) -> usize;
//     fn layer_type(&self) -> String;
// }

// pub struct Dense {
//     weights: NDArray,
//     biases: NDArray,
//     input: Option<NDArray>,
//     gradients: Option<(NDArray, NDArray)>, // (weight_gradients, bias_gradients)
// }

// impl Dense {
//     pub fn new(input_size: usize, output_size: usize) -> Self {
//         // He initialization with adjusted scale
//         let scale = (2.0 / input_size as f64).sqrt() * 0.5;  // Increased from 0.1
        
//         Dense {
//             weights: NDArray::randn_2d(input_size, output_size)
//                 .multiply_scalar(scale),
//             biases: NDArray::zeros(vec![1, output_size]),
//             input: None,
//             gradients: None,
//         }
//     }
// }

// impl Layer for Dense {
//     fn forward(&mut self, input: &NDArray) -> NDArray {
//         println!("Dense forward pass:");
//         println!("  Input shape: {:?}", input.shape());
//         println!("  Weights shape: {:?}", self.weights.shape());
//         println!("  Biases shape: {:?}", self.biases.shape());
        
//         self.input = Some(input.clone());
//         let output = input.dot(&self.weights);
//         println!("  After dot product shape: {:?}", output.shape());
        
//         // No need to reshape biases - they're already [1, output_size]
//         println!("  Adding biases with shape: {:?}", self.biases.shape());
//         let normalized = output.batch_normalize();  // Use batch normalization
//         let final_output = normalized + &self.biases;
//         println!("  Final output shape: {:?}", final_output.shape());
        
//         final_output
//     }

//     fn backward(&mut self, gradient: &NDArray) -> NDArray {
//         let input = self.input.as_ref().unwrap();
//         let weight_gradients = input.transpose().dot(gradient);
//         let bias_gradients = gradient.sum_axis(0);
        
//         // Increase clip value since we're using batch norm
//         let clip_value = 5.0;
//         let clipped_weight_gradients = weight_gradients.clip(-clip_value, clip_value);
//         let clipped_bias_gradients = bias_gradients.clip(-clip_value, clip_value);
        
//         self.gradients = Some((clipped_weight_gradients, clipped_bias_gradients));
//         gradient.dot(&self.weights.transpose())
//     }

//     fn update(&mut self, learning_rate: f64) {
//         if let Some((ref weight_gradients, ref bias_gradients)) = self.gradients {
//             // Update weights and biases
//             self.weights = self.weights.subtract(&weight_gradients.multiply_scalar(learning_rate));
//             self.biases = self.biases.subtract(&bias_gradients.multiply_scalar(learning_rate));
//         }
//     }

//     fn get_weights_mut(&mut self) -> &mut NDArray {
//         &mut self.weights
//     }

//     fn get_gradients(&self) -> Option<(NDArray, NDArray)> {
//         self.gradients.as_ref().map(|(w, b)| (w.clone(), b.clone()))
//     }

//     fn output_shape(&self) -> Vec<usize> {
//         vec![self.weights.shape()[1]]  // Output size is determined by number of units
//     }
    
//     fn parameter_count(&self) -> usize {
//         let (input_size, output_size) = (self.weights.shape()[0], self.weights.shape()[1]);
//         (input_size * output_size) + output_size  // weights + biases
//     }
    
//     fn layer_type(&self) -> String {
//         "Dense".to_string()
//     }
// }

// pub struct ActivationLayer {
//     activation: Box<dyn Activation>,
//     input: Option<NDArray>,
// }

// impl ActivationLayer {
//     pub fn new(activation: Box<dyn Activation>) -> Self {
//         ActivationLayer { activation, input: None }
//     }
// }

// impl Layer for ActivationLayer {
//     fn forward(&mut self, input: &NDArray) -> NDArray {
//         // Store input for backward pass
//         self.input = Some(input.clone());
//         self.activation.forward(input)
//     }

//     fn backward(&mut self, gradient: &NDArray) -> NDArray {
//         // Get input from stored value
//         let input = self.input.as_ref().expect("No input found. Forward pass must be called before backward pass");
//         self.activation.backward(gradient, input)
//     }

//     fn update(&mut self, _learning_rate: f64) {
//         // No update needed for activation layers
//     }

//     fn get_weights_mut(&mut self) -> &mut NDArray {
//         panic!("ActivationLayer does not have weights")
//     }

//     fn get_gradients(&self) -> Option<(NDArray, NDArray)> {
//         None
//     }

//     fn output_shape(&self) -> Vec<usize> {
//         // Activation layers don't change the shape
//         vec![]  // This will be the same as input shape
//     }
    
//     fn parameter_count(&self) -> usize {
//         0  // Activation layers have no parameters
//     }
    
//     fn layer_type(&self) -> String {
//         "Activation".to_string()
//     }
// } 
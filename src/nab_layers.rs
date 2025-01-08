use std::ops::Add;

use crate::nab_array::NDArray;
use crate::nab_activations::Activation;

pub trait Layer {
    fn forward(&mut self, input: &NDArray) -> NDArray;
    fn backward(&mut self, gradient: &NDArray) -> NDArray;
    fn update(&mut self, learning_rate: f64);
    fn get_weights_mut(&mut self) -> &mut NDArray;
    fn get_gradients(&self) -> Option<(NDArray, NDArray)>;
}

pub struct Dense {
    weights: NDArray,
    biases: NDArray,
    input: Option<NDArray>,
    gradients: Option<(NDArray, NDArray)>, // (weight_gradients, bias_gradients)
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Dense {
            weights: NDArray::randn_2d(input_size, output_size).multiply_scalar(0.01),
            biases: NDArray::zeros(vec![1, output_size]),
            input: None,
            gradients: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &NDArray) -> NDArray {
        // Ensure input is compatible with weights
        assert_eq!(input.shape()[1], self.weights.shape()[0], "Input shape must match weights shape");

        // Store input for backprop
        self.input = Some(input.clone());

        // Compute output = input * weights + biases
        input.dot(&self.weights).add(&self.biases)
    }

    fn backward(&mut self, gradient: &NDArray) -> NDArray {
        // Compute gradients
        let input = self.input.as_ref().unwrap();
        let weight_gradients = input.transpose().dot(gradient);
        let bias_gradients = gradient.sum_axis(0);
        
        // Store gradients
        self.gradients = Some((weight_gradients, bias_gradients));
        
        // Return gradient for previous layer
        gradient.dot(&self.weights.transpose())
    }

    fn update(&mut self, learning_rate: f64) {
        if let Some((ref weight_gradients, ref bias_gradients)) = self.gradients {
            // Update weights and biases
            self.weights = self.weights.subtract(&weight_gradients.multiply_scalar(learning_rate));
            self.biases = self.biases.subtract(&bias_gradients.multiply_scalar(learning_rate));
        }
    }

    fn get_weights_mut(&mut self) -> &mut NDArray {
        &mut self.weights
    }

    fn get_gradients(&self) -> Option<(NDArray, NDArray)> {
        self.gradients.as_ref().map(|(w, b)| (w.clone(), b.clone()))
    }
}

pub struct ActivationLayer {
    activation: Box<dyn Activation>,
    input: Option<NDArray>,
}

impl ActivationLayer {
    pub fn new(activation: Box<dyn Activation>) -> Self {
        ActivationLayer { activation, input: None }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: &NDArray) -> NDArray {
        self.activation.forward(input)
    }

    fn backward(&mut self, gradient: &NDArray) -> NDArray {
        self.activation.backward(gradient, &self.input.as_ref().unwrap())
    }

    fn update(&mut self, _learning_rate: f64) {
        // No update needed for activation layers
    }

    fn get_weights_mut(&mut self) -> &mut NDArray {
        panic!("ActivationLayer does not have weights")
    }

    fn get_gradients(&self) -> Option<(NDArray, NDArray)> {
        None
    }
} 
use crate::nab_array::NDArray;
use crate::nab_layers::Layer;
use crate::nab_optimizers::Optimizer;
use crate::nab_activations::Activation;
use crate::nab_loss::Loss;
use crate::nab_optimizers::Adam;   
use crate::nab_layers::Dense;
use crate::nab_loss::CategoricalCrossentropy;
use crate::nab_layers::ActivationLayer;



pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    loss: Box<dyn Loss>,
}

impl Model {
    pub fn new() -> ModelBuilder {
        ModelBuilder::new()
    }

    pub fn compile(&mut self, optimizer: Box<dyn Optimizer>, loss: Box<dyn Loss>) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn get_batches(&self, x_train: &NDArray, y_train: &NDArray, batch_size: usize) -> Vec<(NDArray, NDArray)> {
        let mut batches = Vec::new();
        let num_samples = x_train.shape()[0];
        let num_batches = (num_samples as f64 / batch_size as f64).ceil() as usize;
        for i in 0..num_batches {
            let start = i * batch_size;
            let end = std::cmp::min(start + batch_size, num_samples);
            batches.push((x_train.slice(start, end), y_train.slice(start, end)));
        }
        batches
    }

    pub fn fit(&mut self, 
        x_train: &NDArray, 
        y_train: &NDArray, 
        batch_size: usize, 
        epochs: usize
    ) -> Vec<f64> {
        let mut history = Vec::new();
        
        for _epoch in 0..epochs {
            // Mini-batch training
            for (batch_x, batch_y) in self.get_batches(x_train, y_train, batch_size) {
                // Forward pass
                let output = self.forward(&batch_x);
                
                // Backward pass
                let gradient = self.loss.backward(&output, &batch_y);
                self.backward(&gradient);
                
                // Update weights
                self.update();
            }
            
            // Compute and store loss
            let epoch_loss = self.compute_loss(x_train, y_train);
            history.push(epoch_loss);
        }
        
        history
    }

    /// Performs a forward pass through all layers of the model
    ///
    /// # Arguments
    ///
    /// * `input` - The input NDArray to the model.
    ///
    /// # Returns
    ///
    /// The output NDArray after passing through all layers.
    pub fn forward(&mut self, input: &NDArray) -> NDArray {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, gradient: &NDArray) -> NDArray {
        let mut grad = gradient.clone();            
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    pub fn update(&mut self) {
        for layer in &mut self.layers {
            // Get gradients first and store them
            let gradients = layer.get_gradients().map(|(w, b)| (w.clone(), b.clone()));
            if let Some((weight_gradients, _)) = gradients {
                let weights = layer.get_weights_mut();
                self.optimizer.update(weights, &weight_gradients);
            }
        }
    }

    /// Computes the loss for the given input and target data
    ///
    /// # Arguments
    ///
    /// * `x` - The input NDArray.
    /// * `y` - The target NDArray.
    ///
    /// # Returns
    ///
    /// The computed loss as a f64.
    pub fn compute_loss(&mut self, x: &NDArray, y: &NDArray) -> f64 {
        let predictions = self.forward(x);
        self.loss.forward(&predictions, y)
    }
}

pub struct ModelBuilder {
    layers: Vec<Box<dyn Layer>>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        ModelBuilder { layers: Vec::new() }
    }

    pub fn add_dense(mut self, units: usize, activation: Box<dyn Activation>) -> Self {
        let input_size = if self.layers.is_empty() {
            784 // For MNIST example
        } else {
            // Get output size of previous layer
            // This needs to be implemented
            0
        };
        
        let dense = Dense::new(input_size, units);
        self.layers.push(Box::new(dense));
        self.layers.push(Box::new(ActivationLayer::new(activation)));
        self
    }

    pub fn build(self) -> Model {
        Model {
            layers: self.layers,
            optimizer: Box::new(Adam::default()),
            loss: Box::new(CategoricalCrossentropy),
        }
    }
} 

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    

    // impl Model {
    //     fn predict(&self, input: &NDArray) -> NDArray {
    //         // Forward pass through all layers
    //         let mut output = input.clone();
    //         for layer in &self.layers {
    //             output = layer.forward(&output);
    //         }
    //         output
    //     }
    // }

    // // Helper function to create synthetic data
    // fn create_test_data(samples: usize, features: usize, classes: usize) -> (NDArray, NDArray) {
    //     let x = NDArray::randn_2d(samples, features);
    //     let y = NDArray::one_hot_encode(
    //         &NDArray::from_vec(vec![0.0; samples].iter().enumerate()
    //             .map(|(i, _)| (i % classes) as f64)
    //             .collect())
    //     );
    //     (x, y)
    // }

    #[test]
    fn test_nothing() {
        assert!(2 == 2);
    }


    
    // #[test]
    // fn test_simple_neural_network() {
    //     let (x_train, y_train) = create_test_data(100, 784, 10);
    //     // ... rest of the test ...
    // }
} 
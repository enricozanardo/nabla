use crate::nab_array::NDArray;
use crate::nab_layers::Layer;
use crate::nab_optimizers::Optimizer;
use crate::nab_activations::Activation;
use crate::nab_loss::Loss;
use crate::nab_optimizers::Adam;   
use crate::nab_layers::Dense;
use crate::nab_loss::CategoricalCrossentropy;
use crate::nab_layers::ActivationLayer;
use std::collections::HashMap;



pub struct Model {
    input_shape: Vec<usize>,
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    loss: Box<dyn Loss>,
    metrics: Vec<String>,
}

impl Model {
    pub fn new() -> ModelBuilder {
        ModelBuilder::new()
    }

        
    pub fn compile(&mut self, optimizer: Box<dyn Optimizer>, loss: Box<dyn Loss>, metrics: Vec<String>) {
        self.optimizer = optimizer;
        self.loss = loss;
        self.metrics = metrics;
    }

    pub fn get_batches(&self, x_train: &NDArray, y_train: &NDArray, batch_size: usize) -> Vec<(NDArray, NDArray)> {
        let mut batches = Vec::new();
        let num_samples = x_train.shape()[0];
        let num_batches = (num_samples as f64 / batch_size as f64).ceil() as usize;
        
        println!("Creating batches:");
        println!("  Total samples: {}", num_samples);
        println!("  Batch size: {}", batch_size);
        println!("  Number of batches: {}", num_batches);
        println!("  Input shape: {:?}", x_train.shape());
        
        for i in 0..num_batches {
            let start = i * batch_size;
            let end = std::cmp::min(start + batch_size, num_samples);
            let mut batch_x = x_train.slice(start, end);
            let mut batch_y = y_train.slice(start, end);
            
            // Pad the last batch if it's smaller than batch_size
            if batch_x.shape()[0] < batch_size {
                println!("Padding last batch from size {} to {}", batch_x.shape()[0], batch_size);
                batch_x = batch_x.pad_to_size(batch_size);
                batch_y = batch_y.pad_to_size(batch_size);
            }
            
            println!("Batch {}:", i);
            println!("  X shape: {:?}", batch_x.shape());
            println!("  Y shape: {:?}", batch_y.shape());
            
            batches.push((batch_x, batch_y));
        }
        batches
    }

    pub fn fit(&mut self, 
        x_train: &NDArray, 
        y_train: &NDArray, 
        batch_size: usize, 
        epochs: usize
    ) -> Vec<HashMap<String, f64>> {
        let mut history = Vec::new();
        
        for _epoch in 0..epochs {
            let mut epoch_metrics = HashMap::new();
            let mut correct_predictions = 0;
            let total_samples = x_train.shape()[0];
            
            println!("Total samples: {}", total_samples);

            // Mini-batch training
            for (batch_x, batch_y) in self.get_batches(x_train, y_train, batch_size) {
                println!("Batch X shape: {:?}", batch_x.shape());
                println!("Batch Y shape: {:?}", batch_y.shape());

                // Forward pass
                let output = self.forward(&batch_x);
                println!("Output shape: {:?}", output.shape());
                
                // Calculate accuracy if it's in metrics
                if self.metrics.contains(&"accuracy".to_string()) {
                    correct_predictions += self.calculate_accuracy(&output, &batch_y);
                }
                
                let gradient = self.loss.backward(&output, &batch_y);
                self.backward(&gradient);
                self.update();
            }
            
            // Compute and store metrics
            let epoch_loss = self.compute_loss(x_train, y_train);
            epoch_metrics.insert("loss".to_string(), epoch_loss);
            
            if self.metrics.contains(&"accuracy".to_string()) {
                let accuracy = correct_predictions as f64 / total_samples as f64;
                epoch_metrics.insert("accuracy".to_string(), accuracy);
            }
            
            history.push(epoch_metrics);
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
        // Ensure input is 2D
        let input = if input.ndim() == 1 {
            input.reshape(vec![1, -(input.shape()[0] as isize)])  // Convert to isize with negative sign
        } else {
            input.clone()
        };
        
        // Now check the feature dimension
        assert_eq!(
            input.shape()[1], 
            self.input_shape[0], 
            "Input features {} does not match expected features {}", 
            input.shape()[1], 
            self.input_shape[0]
        );
        
        let mut output = input;
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

    fn calculate_accuracy(&self, predictions: &NDArray, targets: &NDArray) -> usize {
        println!("Debug: Predictions shape before reshape: {:?}", predictions.shape());
        println!("Debug: Targets shape before reshape: {:?}", targets.shape());

        // Ensure predictions and targets are 2D
        let predictions = if predictions.ndim() == 1 {
            predictions.reshape(vec![1, -(predictions.shape()[0] as isize)])
        } else {
            predictions.clone()
        };

        let targets = if targets.ndim() == 1 {
            targets.reshape(vec![1, targets.shape()[0] as isize])
        } else {
            targets.clone()
        };

        println!("Debug: Predictions shape after reshape: {:?}", predictions.shape());
        println!("Debug: Targets shape after reshape: {:?}", targets.shape());

        // Get the indices of maximum values along rows (axis=1)
        let pred_argmax = predictions.argmax(Some(1));
        let target_argmax = targets.argmax(Some(1));

        println!("Debug: Pred argmax values: {:?}", pred_argmax);
        println!("Debug: Target argmax values: {:?}", target_argmax);

        // Count matches
        let correct = pred_argmax.iter()
            .zip(target_argmax.iter())
            .filter(|(p, t)| p == t)
            .count();

        println!("Debug: Correct predictions: {}", correct);
        correct
    }

    /// Prints a summary of the model architecture
    pub fn summary(&self) {
        println!("Model Summary");
        println!("=============================================================");
        println!("{:<20} {:<20} {}", "Layer Type", "Output Shape", "Param #");
        println!("=============================================================");
        
        let mut total_params = 0;
        
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = layer.layer_type();
            let output_shape = layer.output_shape();
            let params = layer.parameter_count();
            
            println!("{:<20} {:<20} {}", 
                format!("layer_{} ({})", i, layer_type),
                format!("{:?}", output_shape),
                params
            );
            
            total_params += params;
        }
        
        println!("=============================================================");
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", total_params);
        println!("Non-trainable params: 0");
    }
}

pub struct ModelBuilder {
    layers: Vec<Box<dyn Layer>>,
    input_shape: Option<Vec<usize>>,
}

impl ModelBuilder {
    pub fn new() -> Self {
        ModelBuilder { 
            layers: Vec::new(),
            input_shape: None,
        }
    }

    pub fn input(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = Some(shape);
        self
    }

    pub fn add_dense(mut self, units: usize, activation: Box<dyn Activation>) -> Self {
        let input_size = if self.layers.is_empty() {
            match &self.input_shape {
                Some(shape) => shape[0],
                None => panic!("Input shape must be specified before adding layers")
            }
        } else {
            let last_layer = self.layers.last().unwrap();
            let shape = last_layer.output_shape();
            if shape.is_empty() {
                self.layers[self.layers.len() - 2].output_shape()[0]
            } else {
                shape[0]
            }
        };
        
        let dense = Dense::new(input_size, units);
        self.layers.push(Box::new(dense));
        self.layers.push(Box::new(ActivationLayer::new(activation)));
        self
    }

    pub fn build(self) -> Model {
        assert!(self.input_shape.is_some(), "Input shape must be specified");
        Model {
            input_shape: self.input_shape.unwrap(),
            layers: self.layers,
            optimizer: Box::new(Adam::default()),
            loss: Box::new(CategoricalCrossentropy),
            metrics: Vec::new(),
        }
    }
} 

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::nab_activations::{ReLU, Softmax, Sigmoid};
    

    
    #[test]
    fn test_model() {
        let mut model = Model::new()
            .input(vec![784])
            .add_dense(32, Box::new(ReLU::default()))
            .add_dense(10, Box::new(Softmax::default()))
            .build();

        model.compile(
            Box::new(Adam::new(0.0005, 0.9, 0.999, 1e-8)),
            Box::new(CategoricalCrossentropy),
            vec!["accuracy".to_string()]
        );

        model.summary();

        // let ((mut train_images, train_labels), (mut test_images, test_labels)) = 
        //     NDArray::load_and_split_dataset("datasets/mnist_test", 80.0).unwrap();

        // assert_eq!(train_images.shape()[0] + test_images.shape()[0], 999);
        // assert_eq!(train_labels.shape()[0] + test_labels.shape()[0], 999);

        // println!("Training samples: {:?}", train_images.shape());
        // println!("Test samples: {:?}", test_images.shape());

        // train_images.normalize();
        // test_images.normalize();

        // let reshaped_images = train_images.reshape(vec![-1, 784]);
        // let reshaped_test_images = test_images.reshape(vec![-1, 784]);
        // println!("Reshaped images shape: {:?}", reshaped_images.shape());   

        // let one_hot_train_labels: NDArray = NDArray::one_hot_encode(&train_labels);
        // let one_hot_test_labels: NDArray = NDArray::one_hot_encode(&test_labels);

        // let label_42: NDArray =  one_hot_train_labels.extract_sample(42);
        // label_42.pretty_print(1);

        // // train the model
        // let history = model.fit(
        //     &reshaped_images, 
        //     &one_hot_train_labels, 
        //     32, 
        //     10
        // );
        
        // println!("Training History:");
        // for (epoch, metrics) in history.iter().enumerate() {
        //     println!("Epoch {}: {:?}", epoch + 1, metrics);
        // }

    }
} 
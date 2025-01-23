use crate::nab_array::NDArray;
use crate::nab_layers::NabLayer;
use crate::nab_optimizers::NablaOptimizer;
use crate::nab_loss::NabLoss;
use std::collections::HashMap;

/// Represents a node in the computation graph
pub struct Node {
    pub layer: NabLayer,
    pub inputs: Vec<usize>,  // Indices of input nodes
    pub output_shape: Vec<usize>,
}

/// Represents a model using the Functional API
#[allow(dead_code)]
#[derive(Clone)]
pub struct NabModel {
    layers: Vec<NabLayer>,
    optimizer_type: String,
    learning_rate: f64,
    loss_type: String,  // e.g. "mse", "categorical_crossentropy"
    metrics: Vec<String>,
}

/// Represents an input node in the computation graph
#[derive(Clone)]
pub struct Input {
    shape: Vec<usize>,
    node_index: Option<usize>,
}

/// Represents an output node in the computation graph
#[derive(Clone)]
pub struct Output {
    layer: NabLayer,
    inputs: Vec<usize>,
    output_shape: Vec<usize>,
    previous_output: Option<Box<Output>>,
}

impl Input {
    /// Applies a layer to this input, preserving node connectivity
    pub fn apply<L: Into<NabLayer>>(&self, layer: L) -> Output {
        let mut layer = layer.into();
        let output_shape = layer.compute_output_shape(&self.shape);
        
        // Generate new node index for this layer
        static mut NEXT_LAYER_ID: usize = 1; // 0 is reserved for input
        let layer_id = unsafe {
            let id = NEXT_LAYER_ID;
            NEXT_LAYER_ID += 1;
            id
        };
        
        // Set layer's node index and inputs
        layer.set_node_index(layer_id);
        
        println!("Connecting layer {} (id: {}) to input (id: {})", 
            layer.get_name(), 
            layer_id,
            self.node_index.unwrap()
        );
        
        Output {
            layer,
            inputs: vec![self.node_index.unwrap()],
            output_shape,
            previous_output: None,
        }
    }
}

impl Output {
    /// Applies a layer to this output, maintaining the graph structure
    pub fn apply<L: Into<NabLayer>>(&self, layer: L) -> Output {
        let mut layer = layer.into();
        let output_shape = layer.compute_output_shape(&self.output_shape);
        
        // Generate new node index for this layer
        static mut NEXT_LAYER_ID: usize = 1; // 0 is reserved for input
        let layer_id = unsafe {
            let id = NEXT_LAYER_ID;
            NEXT_LAYER_ID += 1;
            id
        };
        
        // Set layer's node index and inputs
        layer.set_node_index(layer_id);
        
        println!("Connecting layer {} (id: {}) to {} (id: {})", 
            layer.get_name(), 
            layer_id,
            self.layer.get_name(), 
            self.layer.node_index.expect("Layer node index not set")
        );
        
        Output {
            layer,
            inputs: vec![layer_id],
            output_shape,
            previous_output: Some(Box::new(self.clone())),
        }
    }

    /// Returns the previous layer that produced this output
    pub fn get_previous_layer(&self) -> Option<&NabLayer> {
        // Return layer that produced this output
        None // TODO: Implement layer tracking
    }
}

impl NabModel {
    /// Creates a new input layer with initialized node index
    pub fn input(shape: Vec<usize>) -> Input {
        static mut NEXT_NODE_ID: usize = 0;
        let node_index = unsafe {
            let id = NEXT_NODE_ID;
            NEXT_NODE_ID += 1;
            id
        };
        
        Input {
            shape,
            node_index: Some(node_index),
        }
    }

    /// Creates a new model
    pub fn new() -> Self {
        NabModel {
            layers: Vec::new(),
            optimizer_type: String::new(),
            learning_rate: 0.0,
            loss_type: String::new(),
            metrics: Vec::new(),
        }
    }

    /// Adds a layer to the model
    pub fn add(&mut self, layer: NabLayer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Compiles the model with optimizer, loss function and metrics
    pub fn compile(
        &mut self,
        optimizer_type: &str,
        learning_rate: f64,
        loss_type: &str,
        metrics: Vec<String>,
    ) {
        self.optimizer_type = optimizer_type.to_string();
        self.learning_rate = learning_rate;
        self.loss_type = loss_type.to_string();
        self.metrics = metrics;
    }

    /// Trains for one epoch
    fn train_epoch(&mut self, x: &NDArray, y: &NDArray, _batch_size: usize) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let predictions = self.predict(x);
        
        let loss = self.calculate_loss(&predictions, y);
        metrics.insert("loss".to_string(), loss);
        
        let acc = self.calculate_accuracy(&predictions, y);
        metrics.insert("accuracy".to_string(), acc);
        
        metrics
    }

    /// Creates a new model from input and output nodes
    pub fn new_functional(inputs: Vec<Input>, outputs: Vec<Output>) -> Self {
        let mut layers = Vec::new();
        let mut node_to_layer = HashMap::new();
        
        // First pass: Add input layers
        for input in inputs {
            let mut layer = NabLayer::input(input.shape.clone(), None);
            layer.set_node_index(input.node_index.unwrap());
            node_to_layer.insert(input.node_index.unwrap(), layers.len());
            layers.push(layer);
        }
        
        // Second pass: Build layer graph
        for output in outputs {
            let mut current = output;
            let mut layer_stack = vec![];
            
            // Collect all layers from output to input
            while !node_to_layer.contains_key(&current.layer.node_index.unwrap()) {
                println!("Processing layer: {} (id: {})", 
                    current.layer.get_name(),
                    current.layer.node_index.unwrap()
                );
                
                layer_stack.push(current.layer.clone());
                
                if let Some(prev) = current.previous_output {
                    current = *prev;
                } else {
                    break;
                }
            }
            
            // Add layers in correct order
            for layer in layer_stack.into_iter().rev() {
                println!("Adding layer: {} -> {:?}", layer.get_name(), layer.get_output_shape());
                layers.push(layer);
            }
        }

        // Print model summary
        println!("\nModel Summary:");
        for (i, layer) in layers.iter().enumerate() {
            println!("Layer {}: {} (id: {}) -> {:?}", 
                i,
                layer.get_name(),
                layer.node_index.unwrap(),
                layer.get_output_shape()
            );
        }

        NabModel { layers, optimizer_type: String::new(), learning_rate: 0.0, loss_type: String::new(), metrics: Vec::new() }
    }

    /// Trains the model on input data
    pub fn fit(
        &mut self,
        x_train: &NDArray,
        y_train: &NDArray,
        batch_size: usize,
        epochs: usize,
        validation_data: Option<(&NDArray, &NDArray)>,
    ) -> HashMap<String, Vec<f64>> {
        let mut history = HashMap::new();
        let mut train_metrics = Vec::new();
        let mut val_metrics = Vec::new();

        for epoch in 0..epochs {
            // Training phase
            let metrics = self.train_epoch(x_train, y_train, batch_size);
            train_metrics.push(metrics);

            // Validation phase
            if let Some((x_val, y_val)) = validation_data {
                let val_metric = self.evaluate(x_val, y_val, batch_size);
                val_metrics.push(val_metric);
            }

            // Print progress
            self.print_progress(epoch + 1, epochs, &train_metrics[epoch], 
                              val_metrics.last());
        }

        // Store history
        history.insert("loss".to_string(), 
            train_metrics.iter().map(|m| m["loss"]).collect());
        history.insert("accuracy".to_string(), 
            train_metrics.iter().map(|m| m["accuracy"]).collect());

        if !val_metrics.is_empty() {
            history.insert("val_loss".to_string(), 
                val_metrics.iter().map(|m| m["loss"]).collect());
            history.insert("val_accuracy".to_string(), 
                val_metrics.iter().map(|m| m["accuracy"]).collect());
        }

        history
    }

    /// Prints training progress
    fn print_progress(
        &self,
        epoch: usize,
        total_epochs: usize,
        train_metrics: &HashMap<String, f64>,
        val_metrics: Option<&HashMap<String, f64>>,
    ) {
        print!("Epoch {}/{} - ", epoch, total_epochs);
        for (name, value) in train_metrics {
            print!("{}: {:.4} ", name, value);
        }
        if let Some(val_metrics) = val_metrics {
            for (name, value) in val_metrics {
                print!("val_{}: {:.4} ", name, value);
            }
        }
        println!();
    }

    /// Evaluates the model on test data
    pub fn evaluate(
        &mut self,
        x_test: &NDArray,
        y_test: &NDArray,
        batch_size: usize,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let predictions = self.predict(x_test);
        
        // Calculate loss
        let loss = self.calculate_loss(&predictions, y_test);
        metrics.insert("loss".to_string(), loss);
        
        // Calculate other metrics
        for metric in &self.metrics {
            match metric.as_str() {
                "accuracy" => {
                    let acc = self.calculate_accuracy(&predictions, y_test);
                    metrics.insert("accuracy".to_string(), acc);
                }
                _ => {}
            }
        }
        
        metrics
    }

    /// Calculates accuracy for classification tasks
    fn calculate_accuracy(&self, predictions: &NDArray, targets: &NDArray) -> f64 {
        let pred_classes = predictions.argmax(Some(1));
        let true_classes = targets.argmax(Some(1));
        
        let correct = pred_classes.iter()
            .zip(true_classes.iter())
            .filter(|(&p, &t)| p == t)
            .count();
            
        correct as f64 / predictions.shape()[0] as f64
    }

    /// Makes predictions on input data
    pub fn predict(&mut self, x: &NDArray) -> NDArray {
        let mut current = x.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current, false);
        }
        current
    }

    fn update_weights(&mut self, weights: &mut NDArray, gradients: &NDArray) {
        match self.optimizer_type.as_str() {
            "sgd" => NablaOptimizer::sgd_update(weights, gradients, self.learning_rate),
            "momentum" => {
                let mut velocity = NDArray::zeros(weights.shape().to_vec());
                NablaOptimizer::sgd_momentum_update(
                    weights, 
                    gradients, 
                    &mut velocity,
                    self.learning_rate,
                    0.9
                );
            }
            _ => NablaOptimizer::sgd_update(weights, gradients, self.learning_rate),
        }
    }

    fn calculate_loss(&self, predictions: &NDArray, targets: &NDArray) -> f64 {
        match self.loss_type.as_str() {
            "mse" => NabLoss::mean_squared_error(predictions, targets),
            "categorical_crossentropy" => NabLoss::cross_entropy_loss(predictions, targets),
            _ => NabLoss::mean_squared_error(predictions, targets),
        }
    }

    fn calculate_loss_gradient(&self, predictions: &NDArray, targets: &NDArray) -> NDArray {
        match self.loss_type.as_str() {
            "mse" => predictions.subtract(targets).divide_scalar(predictions.shape()[0] as f64),
            "categorical_crossentropy" => predictions.subtract(targets).divide_scalar(predictions.shape()[0] as f64),
            _ => predictions.subtract(targets).divide_scalar(predictions.shape()[0] as f64),
        }
    }

    // Add debug method
    pub fn print_layers(&self) {
        println!("\nLayer stack:");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("{}: {} -> {:?}", i, layer.get_name(), layer.get_output_shape());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_activations::NablaActivation;
    use crate::nab_optimizers::NablaOptimizer;
    use crate::nab_loss::NabLoss;

    #[test]
    fn test_mnist_classification() {
        // Create model architecture
        let input = NabModel::input(vec![784]);
        
        // Build network with proper layer order
        let dense1 = NabLayer::dense(784, 128, Some("relu"), Some("dense1"));
        let x = input.apply(dense1);
        
        let dense2 = NabLayer::dense(128, 64, Some("relu"), Some("dense2"));
        let x = x.apply(dense2);
        
        let output_layer = NabLayer::dense(64, 10, Some("softmax"), Some("output"));
        let output = x.apply(output_layer);

        // Create model with proper layer ordering
        let mut model = NabModel::new_functional(vec![input], vec![output]);
        
        // Debug layer setup
        model.print_layers();
        
        model.compile(
            "sgd",                      // optimizer type
            0.01,                       // learning rate
            "categorical_crossentropy", // loss type
            vec!["accuracy".to_string()]
        );

        // Generate dummy data for testing
        let x_train = NDArray::rand_2d(100, 784);
        let y_train = NDArray::rand_2d(100, 10);
        let x_test = NDArray::rand_2d(20, 784);
        let y_test = NDArray::rand_2d(20, 10);

        // Train model
        let history = model.fit(
            &x_train,
            &y_train,
            32,
            5,
            Some((&x_test, &y_test))
        );

        // Verify history contains expected metrics
        assert!(history.contains_key("loss"));
        assert!(history.contains_key("accuracy"));
        assert!(history.contains_key("val_loss"));
        assert!(history.contains_key("val_accuracy"));
    }
}


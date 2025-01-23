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
    fn train_epoch(&mut self, x: &NDArray, y: &NDArray, batch_size: usize) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let num_samples = x.shape()[0];
        let num_batches = (num_samples + batch_size - 1) / batch_size;

        // Process mini-batches
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(num_samples);
            
            // Get batch data
            let x_batch = x.slice(start_idx, end_idx);
            let y_batch = y.slice(start_idx, end_idx);
            
            // Forward and backward pass as one operation
            let (predictions, loss) = self.forward_backward(&x_batch, &y_batch);
            
            // Accumulate metrics
            total_loss += loss * (end_idx - start_idx) as f64;
            total_correct += self.count_correct(&predictions, &y_batch);
        }
        
        // Calculate average metrics
        metrics.insert("loss".to_string(), total_loss / num_samples as f64);
        metrics.insert("accuracy".to_string(), total_correct as f64 / num_samples as f64);
        
        metrics
    }

    fn forward_backward(&mut self, x_batch: &NDArray, y_batch: &NDArray) -> (NDArray, f64) {
        // Forward pass
        let predictions = self.predict(x_batch);
        let loss = self.calculate_loss(&predictions, y_batch);
        let loss_grad = self.calculate_loss_gradient(&predictions, y_batch);
        
        // Backward pass
        let mut gradient = loss_grad;
        let learning_rate = self.learning_rate;  // Cache learning rate
        
        for layer in self.layers.iter_mut().rev() {
            if layer.is_trainable() {
                gradient = layer.backward(&gradient);
                
                // Update weights using cached learning rate
                if let Some(weights) = layer.weights.as_mut() {
                    let weight_grads = layer.weight_gradients.as_ref().unwrap();
                    NablaOptimizer::sgd_update(weights, weight_grads, learning_rate);
                }
                if let Some(biases) = layer.biases.as_mut() {
                    let bias_grads = layer.bias_gradients.as_ref().unwrap();
                    NablaOptimizer::sgd_update(biases, bias_grads, learning_rate);
                }
            }
        }
        
        (predictions, loss)
    }

    fn count_correct(&self, predictions: &NDArray, targets: &NDArray) -> usize {
        let pred_classes = predictions.argmax(Some(1));
        let true_classes = targets.argmax(Some(1));
        
        pred_classes.iter()
            .zip(true_classes.iter())
            .filter(|(&p, &t)| p == t)
            .count()
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
    use crate::nab_mnist::NabMnist;
    use crate::nab_utils::NabUtils;
    #[test]
    fn test_mnist_full_pipeline() {
        // Step 1: Load MNIST data
        println!("Loading MNIST data...");
        let ((x_train, y_train), (x_test, y_test)) = NabUtils::load_and_split_dataset("datasets/mnist_test", 80.0).unwrap();

        // Step 2: Normalize input data (scale pixels to 0-1)
        println!("Normalizing data...");
        let x_train = x_train.divide_scalar(255.0);
        let x_test = x_test.divide_scalar(255.0);

        // Step 2.5: Reshape input data
        let x_train = x_train.reshape(&[x_train.shape()[0], 784])
            .expect("Failed to reshape training data");
        let x_test = x_test.reshape(&[x_test.shape()[0], 784])
            .expect("Failed to reshape test data");

        // Step 2.6: One-hot encode target data
        println!("One-hot encoding targets...");
        let y_train = NDArray::one_hot_encode(&y_train);
        let y_test = NDArray::one_hot_encode(&y_test);
            

        println!("Data shapes:");
        println!("x_train: {:?}", x_train.shape());
        println!("y_train: {:?}", y_train.shape());
        println!("x_test: {:?}", x_test.shape());
        println!("y_test: {:?}", y_test.shape());

        // Step 3: Create model architecture
        println!("Creating model...");
        let input = NabModel::input(vec![784]);  // 28x28 = 784 pixels

        // Dense layer with 512 units and ReLU activation
        let dense1 = NabLayer::dense(784, 512, Some("relu"), Some("dense1"));
        let x = input.apply(dense1);

        // Dense layer with 256 units and ReLU activation
        let dense2 = NabLayer::dense(512, 256, Some("relu"), Some("dense2"));
        let x = x.apply(dense2);

        // Output layer with 10 units (one per digit) and softmax activation
        let output_layer = NabLayer::dense(256, 10, Some("softmax"), Some("output"));
        let output = x.apply(output_layer);

        // Step 4: Create and compile model
        println!("Compiling model...");
        let mut model = NabModel::new_functional(vec![input], vec![output]);
        model.compile(
            "sgd",                      
            0.1,                        // Increase learning rate from 0.01 to 0.1
            "categorical_crossentropy", 
            vec!["accuracy".to_string()]
        );

        // Step 5: Train model
        println!("Training model...");
        let history = model.fit(
            &x_train,
            &y_train,
            64,             // Increase batch size from 32 to 64
            5,             // Increase epochs from 2 to 10
            Some((&x_test, &y_test))
        );

        // Step 6: Evaluate final model
        println!("Evaluating model...");
        let eval_metrics = model.evaluate(&x_test, &y_test, 32);
        
        // Print final results
        println!("Final test accuracy: {:.2}%", eval_metrics["accuracy"] * 100.0);
        
        // Verify model achieved reasonable accuracy (>85%)
        assert!(eval_metrics["accuracy"] > 0.85, 
            "Model accuracy ({:.2}%) below expected threshold", 
            eval_metrics["accuracy"] * 100.0
        );

        // Verify training history contains expected metrics
        assert!(history.contains_key("loss"));
        assert!(history.contains_key("accuracy"));
        assert!(history.contains_key("val_loss"));
        assert!(history.contains_key("val_accuracy"));
    }
}


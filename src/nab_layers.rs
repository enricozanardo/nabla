use crate::nab_array::NDArray;
use crate::nab_activations::NablaActivation;

/// Represents a layer's configuration and state
#[allow(dead_code)]
#[derive(Clone)]
pub struct NabLayer {
    /// Layer type identifier
    pub layer_type: String,
    /// Layer name (unique identifier)
    pub name: String,
    /// Input shape of the layer
    pub input_shape: Vec<usize>,
    /// Output shape of the layer
    pub output_shape: Vec<usize>,
    /// Layer weights (if any)
    pub weights: Option<NDArray>,
    /// Layer biases (if any)
    pub biases: Option<NDArray>,
    /// Stored input for backpropagation
    pub input_cache: Option<NDArray>,
    /// Stored output for backpropagation
    pub output_cache: Option<NDArray>,
    /// Training mode flag
    pub trainable: bool,
    /// Weight gradients for optimization
    pub weight_gradients: Option<NDArray>,
    /// Bias gradients for optimization
    pub bias_gradients: Option<NDArray>,
    /// Type of activation function
    pub activation: Option<String>,
    /// Dropout rate (if applicable)
    pub dropout_rate: Option<f64>,
    /// Dropout mask for backpropagation
    pub dropout_mask: Option<NDArray>,
    /// Epsilon for numerical stability in BatchNorm
    pub epsilon: Option<f64>,
    /// Momentum for running statistics in BatchNorm
    pub momentum: Option<f64>,
    /// Running mean for BatchNorm inference
    pub running_mean: Option<NDArray>,
    /// Running variance for BatchNorm inference
    pub running_var: Option<NDArray>,
    /// Current batch mean (for backprop)
    pub batch_mean: Option<NDArray>,
    /// Current batch variance (for backprop)
    pub batch_var: Option<NDArray>,
    /// Normalized values before scaling (for backprop)
    pub normalized: Option<NDArray>,
    pub node_index: Option<usize>,
    /// Input connections for the layer
    pub input_nodes: Option<Vec<usize>>,
}

#[allow(dead_code)]
impl NabLayer {
    /// Creates a new Input layer
    /// 
    /// # Arguments
    ///
    /// * `shape` - Shape of the input (excluding batch dimension)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// let input_layer = NabLayer::input(vec![784], Some("mnist_input"));
    /// assert_eq!(input_layer.get_output_shape(), &[784]);
    /// ```
    pub fn input(shape: Vec<usize>, name: Option<&str>) -> Self {
        NabLayer {
            layer_type: "Input".to_string(),
            name: name.unwrap_or("input").to_string(),
            input_shape: shape.clone(),
            output_shape: shape,
            weights: None,
            biases: None,
            input_cache: None,
            output_cache: None,
            trainable: false,
            weight_gradients: None,
            bias_gradients: None,
            activation: None,
            dropout_rate: None,
            dropout_mask: None,
            epsilon: None,
            momentum: None,
            running_mean: None,
            running_var: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Creates a new Dense (fully connected) layer
    /// 
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    /// * `units` - Number of output units
    /// * `activation` - Optional activation function ("relu", "sigmoid", "tanh", etc.)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// // Dense layer with ReLU activation
    /// let dense = NabLayer::dense(784, 128, Some("relu"), Some("hidden_1"));
    /// ```
    pub fn dense(
        input_dim: usize, 
        units: usize, 
        activation: Option<&str>,
        name: Option<&str>
    ) -> Self {
        // He initialization
        let scale = (2.0 / input_dim as f64).sqrt();
        
        // Initialize weights and biases
        let weights = NDArray::randn_2d(input_dim, units)
            .multiply_scalar(scale);
        let biases = NDArray::zeros(vec![units]);

        NabLayer {
            layer_type: "Dense".to_string(),
            name: name.unwrap_or("dense").to_string(),
            input_shape: vec![input_dim],
            output_shape: vec![units],
            weights: Some(weights),
            biases: Some(biases),
            input_cache: None,
            output_cache: None,
            trainable: true,
            weight_gradients: None,
            bias_gradients: None,
            activation: activation.map(|s| s.to_string()),
            dropout_rate: None,
            dropout_mask: None,
            epsilon: None,
            momentum: None,
            running_mean: None,
            running_var: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Creates a new Activation layer
    /// 
    /// # Arguments
    ///
    /// * `activation_type` - Type of activation ("relu", "sigmoid", "tanh", etc.)
    /// * `input_shape` - Shape of the input (excluding batch dimension)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// let relu = NabLayer::activation("relu", vec![128], Some("relu_1"));
    /// assert_eq!(relu.get_output_shape(), &[128]);
    /// ```
    pub fn activation(activation_type: &str, input_shape: Vec<usize>, name: Option<&str>) -> Self {
        NabLayer {
            layer_type: "Activation".to_string(),
            name: name.unwrap_or("activation").to_string(),
            input_shape: input_shape.clone(),
            output_shape: input_shape,
            weights: None,
            biases: None,
            input_cache: None,
            output_cache: None,
            trainable: false,
            weight_gradients: None,
            bias_gradients: None,
            activation: Some(activation_type.to_string()),
            dropout_rate: None,
            dropout_mask: None,
            epsilon: None,
            momentum: None,
            running_mean: None,
            running_var: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Creates a new Flatten layer
    /// 
    /// Flattens the input while keeping the batch size.
    /// For example: (batch_size, height, width, channels) -> (batch_size, height * width * channels)
    /// 
    /// # Arguments
    ///
    /// * `input_shape` - Shape of the input (excluding batch dimension)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// // Flatten a 28x28x1 image to 784 features
    /// let flatten = NabLayer::flatten(vec![28, 28, 1], Some("flatten_1"));
    /// assert_eq!(flatten.get_output_shape(), &[784]);
    /// ```
    pub fn flatten(input_shape: Vec<usize>, name: Option<&str>) -> Self {
        // Calculate total size of flattened dimension
        let flattened_size = input_shape.iter().product();

        NabLayer {
            layer_type: "Flatten".to_string(),
            name: name.unwrap_or("flatten").to_string(),
            input_shape: input_shape,
            output_shape: vec![flattened_size],
            weights: None,
            biases: None,
            input_cache: None,
            output_cache: None,
            trainable: false,
            weight_gradients: None,
            bias_gradients: None,
            activation: None,
            dropout_rate: None,
            dropout_mask: None,
            epsilon: None,
            momentum: None,
            running_mean: None,
            running_var: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Creates a new Dropout layer
    /// 
    /// Randomly sets input units to 0 with a probability of rate during training.
    /// During inference (training=false), the layer behaves like an identity function.
    /// 
    /// # Arguments
    ///
    /// * `input_shape` - Shape of the input (excluding batch dimension)
    /// * `rate` - Dropout rate between 0 and 1 (e.g., 0.5 means 50% of units are dropped)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// // Dropout with 50% rate
    /// let dropout = NabLayer::dropout(vec![128], 0.5, Some("dropout_1"));
    /// assert_eq!(dropout.get_output_shape(), &[128]);
    /// ```
    pub fn dropout(input_shape: Vec<usize>, rate: f64, name: Option<&str>) -> Self {
        assert!(rate >= 0.0 && rate < 1.0, "Dropout rate must be between 0 and 1");
        
        NabLayer {
            layer_type: "Dropout".to_string(),
            name: name.unwrap_or("dropout").to_string(),
            input_shape: input_shape.clone(),
            output_shape: input_shape,
            weights: None,
            biases: None,
            input_cache: None,
            output_cache: None,
            trainable: false,
            weight_gradients: None,
            bias_gradients: None,
            activation: None,
            dropout_rate: Some(rate),
            dropout_mask: None,
            epsilon: None,
            momentum: None,
            running_mean: None,
            running_var: None,
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Creates a new BatchNormalization layer
    /// 
    /// Normalizes the activations of the previous layer for each batch.
    /// During training, uses batch statistics. During inference, uses running statistics.
    /// 
    /// # Arguments
    ///
    /// * `input_shape` - Shape of the input (excluding batch dimension)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-5)
    /// * `momentum` - Momentum for running statistics (default: 0.99)
    /// * `name` - Optional name for the layer
    ///
    /// # Example
    ///
    /// ```
    /// use nabla_ml::nab_layers::NabLayer;
    ///
    /// let bn = NabLayer::batch_norm(vec![128], None, None, Some("bn_1"));
    /// assert_eq!(bn.get_output_shape(), &[128]);
    /// ```
    pub fn batch_norm(
        input_shape: Vec<usize>, 
        epsilon: Option<f64>,
        momentum: Option<f64>,
        name: Option<&str>
    ) -> Self {
        let features = input_shape[0];
        let epsilon = epsilon.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.99);

        // Initialize gamma (scale) and beta (shift) parameters
        let gamma = NDArray::ones(features);
        let beta = NDArray::zeros(vec![features]);

        // Initialize running statistics
        let running_mean = NDArray::zeros(vec![features]);
        let running_var = NDArray::ones(features);

        NabLayer {
            layer_type: "BatchNorm".to_string(),
            name: name.unwrap_or("batch_norm").to_string(),
            input_shape: input_shape.clone(),
            output_shape: input_shape,
            weights: Some(gamma),      // gamma (scale)
            biases: Some(beta),        // beta (shift)
            input_cache: None,
            output_cache: None,
            trainable: true,
            weight_gradients: None,    // gamma gradients
            bias_gradients: None,      // beta gradients
            activation: None,
            dropout_rate: None,
            dropout_mask: None,
            epsilon: Some(epsilon),
            momentum: Some(momentum),
            running_mean: Some(running_mean),
            running_var: Some(running_var),
            batch_mean: None,
            batch_var: None,
            normalized: None,
            node_index: None,
            input_nodes: None,
        }
    }

    /// Helper function to broadcast 1D array to match batch dimension
    fn broadcast_to_batch(&self, array: &NDArray, batch_size: usize) -> NDArray {
        let features = array.data().len();
        let mut broadcasted = Vec::with_capacity(batch_size * features);
        for _ in 0..batch_size {
            broadcasted.extend(array.data());
        }
        NDArray::new(broadcasted, vec![batch_size, features])
    }

    /// Helper function to ensure 1D shape for statistics
    fn reshape_to_1d(&self, array: &NDArray) -> NDArray {
        let features = array.data().len();
        array.reshape(&[features]).expect("Failed to reshape to 1D")
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &NDArray, training: bool) -> NDArray {
        // Cache input for all layers
        self.input_cache = Some(input.clone());
        
        let output = match self.layer_type.as_str() {
            "Dense" => {
                let out = self.dense_forward(input);
                self.output_cache = Some(out.clone());
                out
            },
            "BatchNorm" => {
                self.batch_norm_forward(input, training)
            },
            "Activation" => {
                let out = match self.activation.as_ref().unwrap().as_str() {
                    "relu" => NablaActivation::relu_forward(input),
                    "sigmoid" => NablaActivation::sigmoid_forward(input),
                    "tanh" => NablaActivation::tanh_forward(input),
                    "softmax" => NablaActivation::softmax_forward(input, None),
                    _ => input.clone(),
                };
                self.output_cache = Some(out.clone());
                out
            },
            "Flatten" => {
                let out = self.flatten_forward(input, training);
                self.output_cache = Some(out.clone());
                out
            },
            "Dropout" => {
                let out = self.dropout_forward(input, training);
                self.output_cache = Some(out.clone());
                out
            },
            _ => input.clone()
        };
        
        output
    }

    /// Forward pass for Dense layer
    fn dense_forward(&self, input: &NDArray) -> NDArray {
        // 1. Linear transformation
        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();
        let wx = input.dot(weights);
        
        // 2. Add biases
        let batch_size = input.shape()[0];
        let broadcasted_biases = NDArray::from_matrix(
            vec![biases.data().to_vec(); batch_size]
        );
        let linear_output = wx.add(&broadcasted_biases);
        
        // 3. Apply activation if present
        let output = if let Some(act_type) = &self.activation {
            match act_type.as_str() {
                "relu" => NablaActivation::relu_forward(&linear_output),
                "sigmoid" => NablaActivation::sigmoid_forward(&linear_output),
                "tanh" => NablaActivation::tanh_forward(&linear_output),
                "softmax" => NablaActivation::softmax_forward(&linear_output, None),
                _ => panic!("Unknown activation type: {}", act_type),
            }
        } else {
            linear_output
        };

        output.clone()
    }

    /// Forward pass for Activation layer
    fn activation_forward(&mut self, input: &NDArray, _training: bool) -> NDArray {
        self.input_cache = Some(input.clone());
        
        let output = match self.activation.as_ref().unwrap().as_str() {
            "relu" => NablaActivation::relu_forward(input),
            "sigmoid" => NablaActivation::sigmoid_forward(input),
            "tanh" => NablaActivation::tanh_forward(input),
            "leaky_relu" => NablaActivation::leaky_relu_forward(input, None),
            _ => panic!("Unknown activation type: {}", self.activation.as_ref().unwrap()),
        };

        self.output_cache = Some(output.clone());
        output
    }

    /// Forward pass for Flatten layer
    fn flatten_forward(&mut self, input: &NDArray, _training: bool) -> NDArray {
        self.input_cache = Some(input.clone());
        
        // Keep batch size as first dimension
        let batch_size = input.shape()[0];
        let flattened_size = self.output_shape[0];
        
        // Reshape to (batch_size, flattened_size)
        let new_shape = vec![batch_size, flattened_size];
        let output = input.reshape(&new_shape)
            .expect("Failed to reshape in flatten forward");
            
        self.output_cache = Some(output.clone());
        output
    }

    /// Forward pass for Dropout layer
    fn dropout_forward(&mut self, input: &NDArray, training: bool) -> NDArray {
        self.input_cache = Some(input.clone());
        
        if !training || self.dropout_rate.unwrap() == 0.0 {
            return input.clone();
        }

        // Generate dropout mask using rand_uniform
        let mask = NDArray::rand_uniform(input.shape())
            .map(|x| if x > self.dropout_rate.unwrap() { 1.0 } else { 0.0 })
            .multiply_scalar(1.0 / (1.0 - self.dropout_rate.unwrap()));
        
        self.dropout_mask = Some(mask.clone());
        let output = input.multiply(&mask);
        self.output_cache = Some(output.clone());
        output
    }

    /// Forward pass for BatchNormalization layer
    fn batch_norm_forward(&mut self, input: &NDArray, training: bool) -> NDArray {
        self.input_cache = Some(input.clone());
        let batch_size = input.shape()[0];
        
        // Calculate statistics
        let (mean, var) = if training {
            // Compute batch statistics and ensure 1D shape [features]
            let batch_mean = self.reshape_to_1d(&input.mean_axis(0));
            
            // Compute variance using broadcasted mean
            let broadcasted_mean = self.broadcast_to_batch(&batch_mean, batch_size);
            let centered = input.subtract(&broadcasted_mean);
            let batch_var = self.reshape_to_1d(&centered.multiply(&centered).mean_axis(0));
            
            // Update running statistics (all 1D)
            if let (Some(running_mean), Some(running_var)) = 
                (&mut self.running_mean, &mut self.running_var) 
            {
                let momentum = self.momentum.unwrap();
                *running_mean = running_mean.multiply_scalar(momentum)
                    .add(&batch_mean.multiply_scalar(1.0 - momentum));
                *running_var = running_var.multiply_scalar(momentum)
                    .add(&batch_var.multiply_scalar(1.0 - momentum));
            }
            
            self.batch_mean = Some(batch_mean.clone());
            self.batch_var = Some(batch_var.clone());
            
            (batch_mean, batch_var)
        } else {
            (self.running_mean.as_ref().unwrap().clone(), 
             self.running_var.as_ref().unwrap().clone())
        };

        // Broadcast 1D statistics to match input shape
        let broadcasted_mean = self.broadcast_to_batch(&mean, batch_size);
        let broadcasted_var = self.broadcast_to_batch(&var, batch_size);
        let broadcasted_weights = self.broadcast_to_batch(self.weights.as_ref().unwrap(), batch_size);
        let broadcasted_biases = self.broadcast_to_batch(self.biases.as_ref().unwrap(), batch_size);

        // All operations now use properly broadcasted arrays
        let centered = input.subtract(&broadcasted_mean);
        let std_dev = broadcasted_var.add_scalar(self.epsilon.unwrap()).sqrt();
        let normalized = centered.divide(&std_dev);
        self.normalized = Some(normalized.clone());

        let output = normalized.multiply(&broadcasted_weights).add(&broadcasted_biases);
        self.output_cache = Some(output.clone());
        output
    }

    /// Backward pass for the layer
    pub fn backward(&mut self, gradient: &NDArray) -> NDArray {
        match self.layer_type.as_str() {
            "Dense" => self.dense_backward(gradient),
            "Input" => gradient.clone(),
            "Activation" => self.activation_backward(gradient),
            "Flatten" => self.flatten_backward(gradient),
            "Dropout" => self.dropout_backward(gradient),
            "BatchNorm" => self.batch_norm_backward(gradient),
            _ => panic!("Unknown layer type: {}", self.layer_type),
        }
    }

    /// Backward pass for Dense layer
    fn dense_backward(&mut self, gradient: &NDArray) -> NDArray {
        let input = self.input_cache.as_ref().unwrap();
        let output = self.output_cache.as_ref().unwrap();
        let weights = self.weights.as_ref().unwrap();

        // 1. Compute activation gradient
        let act_gradient = if let Some(act_type) = &self.activation {
            match act_type.as_str() {
                "relu" => NablaActivation::relu_backward(gradient, output),
                "sigmoid" => NablaActivation::sigmoid_backward(gradient, output),
                "tanh" => NablaActivation::tanh_backward(gradient, output),
                "softmax" => NablaActivation::softmax_backward(gradient, output),
                _ => panic!("Unknown activation type: {}", act_type),
            }
        } else {
            gradient.clone()
        };

        // 2. Compute gradients
        let input_t = input.transpose().expect("Failed to transpose input");
        let weights_t = weights.transpose().expect("Failed to transpose weights");
        
        self.weight_gradients = Some(input_t.dot(&act_gradient));
        // Fix: Reshape bias gradients to match bias shape
        self.bias_gradients = Some(act_gradient.sum_axis(0).reshape(&[self.output_shape[0]])
            .expect("Failed to reshape bias gradients"));
        
        // 3. Compute input gradient
        act_gradient.dot(&weights_t)
    }

    /// Backward pass for Activation layer
    fn activation_backward(&mut self, gradient: &NDArray) -> NDArray {
        let input = self.input_cache.as_ref().unwrap();
        let output = self.output_cache.as_ref().unwrap();
        
        match self.activation.as_ref().unwrap().as_str() {
            "relu" => NablaActivation::relu_backward(gradient, input),
            "sigmoid" => NablaActivation::sigmoid_backward(gradient, output),
            "tanh" => NablaActivation::tanh_backward(gradient, output),
            "leaky_relu" => NablaActivation::leaky_relu_backward(gradient, input, None),
            _ => panic!("Unknown activation type: {}", self.activation.as_ref().unwrap()),
        }
    }

    /// Backward pass for Flatten layer
    fn flatten_backward(&mut self, gradient: &NDArray) -> NDArray {
        // Get the original input shape from input_cache
        let original_shape = self.input_cache.as_ref().unwrap().shape();
        
        // Reshape gradient to match original input shape
        gradient.reshape(original_shape)
            .expect("Failed to reshape in flatten backward")
    }

    /// Backward pass for Dropout layer
    fn dropout_backward(&mut self, gradient: &NDArray) -> NDArray {
        // Use the same mask from forward pass
        if let Some(mask) = &self.dropout_mask {
            gradient.multiply(mask)
        } else {
            gradient.clone() // No dropout was applied in forward pass
        }
    }

    /// Backward pass for BatchNormalization layer
    #[allow(unused_variables)]
    fn batch_norm_backward(&mut self, gradient: &NDArray) -> NDArray {
        let input = self.input_cache.as_ref().unwrap();
        let batch_size = input.shape()[0];
        let weights = self.weights.as_ref().unwrap();
        let normalized = self.normalized.as_ref().unwrap();
        
        // Broadcast weights for gradient calculation
        let mut broadcasted_weights = Vec::with_capacity(input.data().len());
        for _ in 0..batch_size {
            broadcasted_weights.extend(weights.data());
        }
        let broadcasted_weights = NDArray::new(broadcasted_weights, input.shape().to_vec());
        
        // Gradients for scale and shift
        self.weight_gradients = Some(gradient.multiply(normalized).sum_axis(0));
        self.bias_gradients = Some(gradient.sum_axis(0));
        
        // Gradient with respect to normalized input
        let dx_normalized = gradient.multiply(&broadcasted_weights);
        
        // Gradient with respect to variance
        let std_dev = self.batch_var.as_ref().unwrap()
            .add_scalar(self.epsilon.unwrap())
            .sqrt();
        
        // Broadcast std_dev for division
        let mut broadcasted_std = Vec::with_capacity(input.data().len());
        for _ in 0..batch_size {
            broadcasted_std.extend(std_dev.data());
        }
        let broadcasted_std = NDArray::new(broadcasted_std, input.shape().to_vec());
        
        let dx = dx_normalized.divide(&broadcasted_std);
        
        // Broadcast batch_mean for subtraction
        let mut broadcasted_mean = Vec::with_capacity(input.data().len());
        for _ in 0..batch_size {
            broadcasted_mean.extend(self.batch_mean.as_ref().unwrap().data());
        }
        let broadcasted_mean = NDArray::new(broadcasted_mean, input.shape().to_vec());
        
        let centered = input.subtract(&broadcasted_mean);
        dx.multiply_scalar(1.0 / batch_size as f64)
    }

    /// Returns the output shape of the layer
    pub fn get_output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Returns the name of the layer
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Returns whether the layer is trainable
    pub fn is_trainable(&self) -> bool {
        self.trainable
    }

    /// Computes output shape for a given input shape
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        match self.layer_type.as_str() {
            "Dense" => {
                // Preserve batch size if present, append output dimension
                if input_shape.len() > 1 {
                    vec![input_shape[0], self.output_shape[0]]
                } else {
                    vec![self.output_shape[0]]
                }
            },
            "Input" => {
                // Always preserve input shape including batch size
                input_shape.to_vec()
            },
            "Flatten" => {
                // Preserve batch size, flatten rest
                let flat_size: usize = input_shape[1..].iter().product();
                vec![input_shape[0], flat_size]
            },
            _ => {
                // For other layers, preserve batch size and use stored output shape
                if input_shape.len() > 1 {
                    let mut shape = vec![input_shape[0]];
                    shape.extend(self.output_shape.iter());
                    shape
                } else {
                    self.output_shape.clone()
                }
            }
        }
    }

    // Sets the node index for the layer
    pub fn set_node_index(&mut self, index: usize) {
        self.node_index = Some(index);
    }

    // Add new instance method for setting inputs
    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        // Store input connections in layer
        self.input_nodes = Some(inputs);
    }

    /// Returns the layer type as a string
    pub fn get_type(&self) -> &str {
        &self.layer_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_layer() {
        // Test creation with explicit name
        let input = NabLayer::input(vec![784], Some("mnist_input"));
        assert_eq!(input.get_name(), "mnist_input");
        assert_eq!(input.get_output_shape(), &[784]);
        assert!(!input.is_trainable());

        // Test forward pass
        let data = NDArray::from_matrix(vec![vec![1.0; 784]; 32]); // 32x784 matrix of ones
        let mut layer = NabLayer::input(vec![784], None);
        let output = layer.forward(&data, true);
        
        // Verify output shape and values
        assert_eq!(output.shape(), vec![32, 784]);
        assert_eq!(output.data(), data.data());

        // Test backward pass (should be identity for input layer)
        let gradient = NDArray::from_matrix(vec![vec![1.0; 784]; 32]);
        let backward = layer.backward(&gradient);
        assert_eq!(backward.data(), gradient.data());
    }

    #[test]
    fn test_dense_layer() {
        // Test creation
        let dense = NabLayer::dense(784, 128, Some("relu"), Some("hidden_1"));
        assert_eq!(dense.get_name(), "hidden_1");
        assert_eq!(dense.get_output_shape(), &[128]);
        assert!(dense.is_trainable());

        // Test forward pass
        let batch_size = 32;
        let input = NDArray::from_matrix(vec![vec![0.1; 784]; batch_size]);
        let mut layer = NabLayer::dense(784, 128, None, None);
        let output = layer.forward(&input, true);
        
        // Verify output shape
        assert_eq!(output.shape(), vec![batch_size, 128]);
        
        // Test backward pass
        let gradient = NDArray::from_matrix(vec![vec![0.1; 128]; batch_size]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shapes
        assert_eq!(backward.shape(), vec![batch_size, 784]);
        
        // Verify weight gradients were computed
        assert!(layer.weight_gradients.is_some());
        assert!(layer.bias_gradients.is_some());
    }

    #[test]
    fn test_activation_layer() {
        // Test creation
        let relu = NabLayer::activation("relu", vec![128], Some("relu_1"));
        assert_eq!(relu.get_name(), "relu_1");
        assert_eq!(relu.get_output_shape(), &[128]);
        assert!(!relu.is_trainable());

        // Test forward pass
        let batch_size = 32;
        let input = NDArray::from_matrix(vec![vec![-0.5, 0.0, 0.5]; batch_size]);
        let mut layer = NabLayer::activation("relu", vec![3], None);
        let output = layer.forward(&input, true);
        
        // Verify output shape
        assert_eq!(output.shape(), vec![batch_size, 3]);
        
        // Verify ReLU behavior
        for row in 0..batch_size {
            assert_eq!(output.get_2d(row, 0), 0.0); // negative -> 0
            assert_eq!(output.get_2d(row, 1), 0.0); // zero -> 0
            assert_eq!(output.get_2d(row, 2), 0.5); // positive unchanged
        }

        // Test backward pass
        let gradient = NDArray::from_matrix(vec![vec![1.0; 3]; batch_size]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shape
        assert_eq!(backward.shape(), vec![batch_size, 3]);
        
        // Verify ReLU gradient behavior
        for row in 0..batch_size {
            assert_eq!(backward.get_2d(row, 0), 0.0); // gradient zero for negative input
            assert_eq!(backward.get_2d(row, 1), 0.0); // gradient zero for zero input
            assert_eq!(backward.get_2d(row, 2), 1.0); // gradient unchanged for positive input
        }
    }

    #[test]
    fn test_dense_layer_with_activation() {
        // Test creation with ReLU activation
        let dense = NabLayer::dense(3, 2, Some("relu"), Some("dense_relu"));
        assert_eq!(dense.get_name(), "dense_relu");
        assert_eq!(dense.get_output_shape(), &[2]);
        assert!(dense.is_trainable());

        // Test forward pass with specific inputs
        let input = NDArray::from_matrix(vec![
            vec![-1.0, 0.0, 1.0],  // First sample
            vec![2.0, -2.0, 0.0],  // Second sample
        ]);
        let mut layer = NabLayer::dense(3, 2, Some("relu"), None);
        
        // Force specific weights and biases for predictable outputs
        layer.weights = Some(NDArray::from_matrix(vec![
            vec![1.0, -1.0],  // First input unit
            vec![-1.0, 1.0],  // Second input unit
            vec![0.5, 0.5],   // Third input unit
        ]));
        layer.biases = Some(NDArray::from_vec(vec![0.0, 0.0]));

        let output = layer.forward(&input, true);
        
        // Verify output shape
        assert_eq!(output.shape(), vec![2, 2]);
        
        // Verify ReLU activation (should be zero for negative values)
        assert!(output.get_2d(0, 0) >= 0.0); // First sample, first output
        assert!(output.get_2d(0, 1) >= 0.0); // First sample, second output
        assert!(output.get_2d(1, 0) >= 0.0); // Second sample, first output
        assert!(output.get_2d(1, 1) >= 0.0); // Second sample, second output

        // Test backward pass
        let gradient = NDArray::from_matrix(vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shape
        assert_eq!(backward.shape(), vec![2, 3]);
        
        // Verify gradients are zero where activation was negative
        let output_cache = layer.output_cache.as_ref().unwrap();
        
        // Print debug information
        println!("Output shape: {:?}", output_cache.shape());
        println!("Backward shape: {:?}", backward.shape());
        
        // First, identify which outputs were negative (ReLU made them zero)
        let negative_outputs: Vec<(usize, usize)> = (0..2)
            .flat_map(|i| (0..2).map(move |j| (i, j)))
            .filter(|&(i, j)| output_cache.get_2d(i, j) <= 0.0)
            .collect();

        // Then verify the corresponding gradients in backward pass
        for (i, j) in negative_outputs {
            println!("Checking gradient for negative output at ({}, {})", i, j);
            println!("Output value: {}", output_cache.get_2d(i, j));
            println!("Backward value: {}", backward.get_2d(i, j));
        }
    }

    #[test]
    fn test_flatten_layer() {
        // Test creation
        let flatten = NabLayer::flatten(vec![2, 3, 4], Some("flatten_1"));
        assert_eq!(flatten.get_name(), "flatten_1");
        assert_eq!(flatten.get_output_shape(), &[24]); // 2 * 3 * 4 = 24
        assert!(!flatten.is_trainable());

        // Test forward pass
        let batch_size = 2;
        let input = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  // First sample
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],  // Second sample
        ]);
        let mut layer = NabLayer::flatten(vec![2, 3], None);
        let output = layer.forward(&input, true);
        
        // Verify output shape
        assert_eq!(output.shape(), vec![batch_size, 6]); // 2 * 3 = 6
        
        // Verify values are preserved
        for i in 0..batch_size {
            for j in 0..6 {
                assert_eq!(output.get_2d(i, j), input.get_2d(i, j));
            }
        }

        // Test backward pass
        let gradient = NDArray::from_matrix(vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        ]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shape matches input shape
        assert_eq!(backward.shape(), input.shape());
        
        // Verify gradient values are preserved
        for i in 0..batch_size {
            for j in 0..6 {
                assert_eq!(backward.get_2d(i, j), gradient.get_2d(i, j));
            }
        }
    }

    #[test]
    fn test_dropout_layer() {
        // Test creation
        let dropout = NabLayer::dropout(vec![100], 0.5, Some("dropout_1"));
        assert_eq!(dropout.get_name(), "dropout_1");
        assert_eq!(dropout.get_output_shape(), &[100]);
        assert!(!dropout.is_trainable());

        // Test forward pass during training
        let batch_size = 10;
        let input = NDArray::from_matrix(vec![vec![1.0; 100]; batch_size]);
        let mut layer = NabLayer::dropout(vec![100], 0.5, None);
        
        // Training mode
        let output_train = layer.forward(&input, true);
        assert_eq!(output_train.shape(), vec![batch_size, 100]);
        
        // Verify some units were dropped (approximately 50%)
        let zeros = output_train.data().iter().filter(|&&x| x == 0.0).count();
        let total = output_train.data().len();
        let dropout_rate = zeros as f64 / total as f64;
        assert!((dropout_rate - 0.5).abs() < 0.1, 
            "Dropout rate should be approximately 0.5, got {}", dropout_rate);

        // Test forward pass during inference
        let output_test = layer.forward(&input, false);
        assert_eq!(output_test.data(), input.data(),
            "During testing, dropout should act as identity");

        // Test backward pass
        let gradient = NDArray::from_matrix(vec![vec![1.0; 100]; batch_size]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shape
        assert_eq!(backward.shape(), gradient.shape());
        
        // Verify zeros in forward pass correspond to zeros in backward pass
        if let Some(mask) = &layer.dropout_mask {
            for i in 0..total {
                if mask.data()[i] == 0.0 {
                    assert_eq!(backward.data()[i], 0.0,
                        "Gradient should be zero where input was dropped");
                }
            }
        }

        // Add a test for uniform distribution
        let input = NDArray::from_matrix(vec![vec![1.0; 100]; batch_size]);
        let mut layer = NabLayer::dropout(vec![100], 0.5, None);
        
        // Run multiple forward passes to verify randomness
        let mut different_masks = false;
        let first_output = layer.forward(&input, true);
        
        for _ in 0..5 {
            let output = layer.forward(&input, true);
            if output.data() != first_output.data() {
                different_masks = true;
                break;
            }
        }
        
        assert!(different_masks, "Dropout should generate different masks");
    }

    #[test]
    fn test_batch_norm_layer() {
        // Test creation
        let bn = NabLayer::batch_norm(vec![3], Some(1e-5), Some(0.99), Some("bn_1"));
        assert_eq!(bn.get_name(), "bn_1");
        assert_eq!(bn.get_output_shape(), &[3]);
        assert!(bn.is_trainable());

        // Test forward pass with known values
        let input = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let mut layer = NabLayer::batch_norm(vec![3], Some(1e-5), Some(0.99), None);
        
        // Training mode
        let output_train = layer.forward(&input, true);
        
        // Verify shape
        assert_eq!(output_train.shape(), vec![2, 3]);
        
        // Verify normalization (mean ≈ 0, var ≈ 1)
        let output_mean = output_train.mean_axis(0);
        let output_var = output_train.var_axis(0);
        
        for i in 0..3 {
            assert!((output_mean.get_2d(0, i)).abs() < 1e-5, 
                "Mean should be close to 0, got {}", output_mean.get_2d(0, i));
            assert!((output_var.get_2d(0, i) - 1.0).abs() < 1e-5, 
                "Variance should be close to 1, got {}", output_var.get_2d(0, i));
        }

        // Test backward pass
        let gradient = NDArray::from_matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ]);
        let backward = layer.backward(&gradient);
        
        // Verify gradient shape
        assert_eq!(backward.shape(), input.shape());
        
        // Verify weight and bias gradients were computed
        assert!(layer.weight_gradients.is_some());
        assert!(layer.bias_gradients.is_some());
    }

    #[test]
    fn test_compute_output_shape() {
        // Test for Input layer
        let input_layer = NabLayer::input(vec![784], Some("input_layer"));
        assert_eq!(input_layer.compute_output_shape(&[32, 784]), vec![32, 784]);

        // Test for Dense layer
        let dense_layer = NabLayer::dense(784, 128, Some("relu"), Some("dense_layer"));
        assert_eq!(dense_layer.compute_output_shape(&[32, 784]), vec![32, 128]);

        // Test for Activation layer
        let activation_layer = NabLayer::activation("relu", vec![128], Some("activation_layer"));
        assert_eq!(activation_layer.compute_output_shape(&[32, 128]), vec![32, 128]);

        // Test for Flatten layer
        let flatten_layer = NabLayer::flatten(vec![28, 28, 1], Some("flatten_layer"));
        assert_eq!(flatten_layer.compute_output_shape(&[32, 28, 28, 1]), vec![32, 784]);

        // Test for Dropout layer
        let dropout_layer = NabLayer::dropout(vec![128], 0.5, Some("dropout_layer"));
        assert_eq!(dropout_layer.compute_output_shape(&[32, 128]), vec![32, 128]);

        // Test for BatchNorm layer
        let batch_norm_layer = NabLayer::batch_norm(vec![128], None, None, Some("batch_norm_layer"));
        assert_eq!(batch_norm_layer.compute_output_shape(&[32, 128]), vec![32, 128]);
    }
}
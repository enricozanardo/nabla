use crate::nab_embeddings::EmbeddingLayer;
use crate::nab_stacked_transformer_layer::StackedTransformerLayer;
use crate::nab_output_head::OutputHead;
use crate::nab_distil_transformer_block::DistilTransformerBlock;
use crate::nab_sa::NabAttention;
use crate::nab_model::FeedForwardNetwork;
use crate::nab_array::NDArray;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json;

/// TinyLLMModel represents a tiny language model by combining an embedding layer,
/// a stacked transformer, and an output head that predicts token probabilities.
///
/// Italian: TinyLLMModel rappresenta un piccolo modello linguistico combinando un embedding layer,
/// un trasformatore impilato e un output head che prevede le probabilità dei token.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TinyLLMModel {
    /// Embedding layer: maps token IDs to dense embeddings and adds positional encoding.
    pub embedding_layer: EmbeddingLayer,
    /// Stacked transformer: processes embeddings through multiple transformer blocks with residual connections.
    pub transformer: StackedTransformerLayer,
    /// Output head: projects hidden states to vocabulary probabilities using a linear layer and softmax.
    pub output_head: OutputHead,
}

impl TinyLLMModel {
    /// Constructs a new TinyLLMModel.
    /// TODO: Investigate if any improvements are needed for real model initialization.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary (number of tokens).
    /// * `embedding_dim` - Dimension of the embeddings (and model hidden dimension).
    /// * `n_layers` - Number of transformer layers to stack.
    ///
    /// # Returns
    /// A new instance of TinyLLMModel.
    ///
    /// Italian: Costruisce un nuovo TinyLLMModel.
    /// Argomenti:
    /// - `vocab_size`: dimensione del vocabolario.
    /// - `embedding_dim`: dimensione degli embedding (e del modello).
    /// - `n_layers`: numero di layer transformer da impilare.
    pub fn new(vocab_size: usize, embedding_dim: usize, n_layers: usize) -> Self {
        let embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim);
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let attention = NabAttention::dummy(embedding_dim);
            let ffn = FeedForwardNetwork::new(embedding_dim, embedding_dim, embedding_dim);
            let block = DistilTransformerBlock { attention, ffn };
            blocks.push(block);
        }
        let transformer = StackedTransformerLayer::new(blocks);
        let output_head = OutputHead::new(embedding_dim, vocab_size);
        TinyLLMModel {
            embedding_layer,
            transformer,
            output_head,
        }
    }

    /// Performs a forward pass through the TinyLLMModel.
    /// TODO: Replace with real forward pass implementation if necessary.
    ///
    /// # Arguments
    /// * `tokens` - An NDArray of token IDs.
    ///
    /// # Returns
    /// An NDArray of shape [sequence_length, vocab_size] containing token probabilities.
    ///
    /// Italian: Esegue il forward pass attraverso TinyLLMModel. Converte gli ID dei token in embedding,
    /// li processa con il trasformatore, ed applica l'output head per ottenere le probabilità dei token.
    pub fn forward(&self, tokens: &NDArray) -> NDArray {
        let embeddings = self.embedding_layer.forward(tokens);
        let hidden = self.transformer.forward(&embeddings);
        self.output_head.forward(&hidden)
    }

    /// Trains the language model.
    /// TODO: Replace dummy gradient computations and parameter updates with real backpropagation and optimizer logic.
    /// 
    /// The training loop iterates over epochs and mini-batches. For each mini-batch, it:
    /// - Performs a forward pass through the model.
    /// - Computes the loss using the cross_entropy_loss_lm function from NabLoss.
    /// - Simulates a backward pass by generating dummy gradients (constant values) for each parameter.
    /// - Updates the model parameters using the sgd_optimizer_step function.
    /// - Tracks and returns the average loss and (optional) accuracy per epoch.
    /// 
    /// Note: This is a simplified training loop for demonstration purposes and uses dummy gradients.
    ///
    /// # Arguments
    ///
    /// * `input` - NDArray of input token IDs (1D).
    /// * `target` - NDArray of target token IDs (1D), shifted by one relative to input.
    /// * `batch_size` - The size of each mini-batch.
    /// * `epochs` - Number of training epochs.
    /// * `learning_rate` - Learning rate for SGD updates.
    ///
    /// # Returns
    ///
    /// A tuple (loss_history, accuracy_history) where each is a Vec<f64> containing the metric for each epoch.
    pub fn train_language_model(&mut self, input: &crate::nab_array::NDArray, target: &crate::nab_array::NDArray, batch_size: usize, epochs: usize, learning_rate: f64) -> (Vec<f64>, Vec<f64>) {
        use crate::nab_loss::NabLoss;
        use indicatif::{ProgressBar, ProgressStyle};

        let fixed_input = if input.ndim() == 1 {
            input.reshape(&[input.size(), 1]).unwrap()
        } else {
            input.clone()
        };
        let num_samples = fixed_input.shape()[0];
        let feature_dim = fixed_input.shape()[1];
        let mut epoch_losses = Vec::new();
        let mut epoch_accuracies = Vec::new();

        for epoch in 0..epochs {
            let pb = ProgressBar::new(num_samples as u64);
            let template_string = format!("Epoch {{msg}}/{} [{{{{bar:40.cyan/blue}}}}] {{pos}}/{{len}}", epochs);
            // Leak the string to obtain a &'static str
            let template_static: &'static str = Box::leak(template_string.into_boxed_str());
            let mut style = ProgressStyle::default_bar();
            style = style.template(template_static).unwrap();
            style = style.progress_chars("##-");
            pb.set_style(style);

            let mut total_loss = 0.0;
            let mut total_correct = 0;
            let mut batch_count = 0;

            // Iterate over batches
            let mut i = 0;
            while i < num_samples {
                let current_batch = if i + batch_size <= num_samples { batch_size } else { num_samples - i };
                // Extract batch for input (2D) and target (1D)
                let start_idx = i * feature_dim;
                let end_idx = (i + current_batch) * feature_dim;
                let batch_input = crate::nab_array::NDArray::new(fixed_input.data()[start_idx..end_idx].to_vec(), vec![current_batch, feature_dim]);
                let batch_target = crate::nab_array::NDArray::new(target.data()[i..i+current_batch].to_vec(), vec![current_batch]);

                // Forward pass: get hidden representation and logits
                let (hidden, logits) = self.forward_with_hidden(&batch_input);
                let predictions = logits; // predictions are softmax probabilities from output head

                // Compute loss using cross-entropy for language modeling
                let loss = crate::nab_loss::NabLoss::cross_entropy_loss_lm(&predictions, &batch_target);
                total_loss += loss;

                // Compute accuracy: for each example, check if argmax equals target token
                let mut correct = 0;
                let vocab = predictions.shape()[1];
                for j in 0..current_batch {
                    let row_start = j * vocab;
                    let row_end = row_start + vocab;
                    let row = &predictions.data()[row_start..row_end];
                    let row_nd = crate::nab_array::NDArray::new(row.to_vec(), vec![1, vocab]);
                    let (predicted_idx, _) = Self::argmax(&row_nd);
                    if (batch_target.data()[j] as usize) == predicted_idx {
                        correct += 1;
                    }
                }
                total_correct += correct;

                // Backward pass: compute gradient of loss with respect to logits
                // For softmax cross-entropy: grad = predictions - one_hot(target) / batch_size
                let mut grad_logits = predictions.data().to_vec();
                for j in 0..current_batch {
                    let target_idx = batch_target.data()[j] as usize;
                    grad_logits[j * vocab + target_idx] -= 1.0;
                }
                // Scale gradients by 1/current_batch
                for val in grad_logits.iter_mut() {
                    *val /= current_batch as f64;
                }
                let grad_logits_nd = crate::nab_array::NDArray::new(grad_logits, vec![current_batch, vocab]);

                // Compute gradients w.r.t output head parameters
                // Let hidden be the activations before output head: shape [current_batch, hidden_dim]
                // dW = hidden^T dot grad_logits; db = sum over rows of grad_logits
                let hidden_T = hidden.transpose().unwrap();
                let grad_w = hidden_T.dot(&grad_logits_nd);

                // Compute bias gradient by summing grad_logits over batch (row-wise sum)
                let mut grad_b = vec![0.0; vocab];
                for j in 0..current_batch {
                    for k in 0..vocab {
                        grad_b[k] += grad_logits_nd.data()[j * vocab + k];
                    }
                }
                let grad_b_nd = crate::nab_array::NDArray::new(grad_b, vec![1, vocab]);

                // Update output head parameters using SGD
                // new_param = param - learning_rate * gradient
                // Assuming NDArray has element-wise subtraction and multiplication
                let updated_weight = self.output_head.weight.subtract(&grad_w.multiply_scalar(learning_rate));
                let updated_bias = self.output_head.bias.subtract(&grad_b_nd.multiply_scalar(learning_rate));
                self.output_head.weight = updated_weight;
                self.output_head.bias = updated_bias;

                batch_count += 1;
                pb.inc(current_batch as u64);
                i += current_batch;
            }
            let finish_msg = format!("Epoch {} complete", epoch + 1);
            let finish_msg_static: &'static str = Box::leak(finish_msg.into_boxed_str());
            pb.finish_with_message(finish_msg_static);
            epoch_losses.push(total_loss / batch_count as f64);
            epoch_accuracies.push(total_correct as f64 / num_samples as f64);
            println!("Epoch {}: Loss = {:.6}, Accuracy = {:.2}%", epoch + 1, epoch_losses.last().unwrap(), epoch_accuracies.last().unwrap() * 100.0);
        }

        (epoch_losses, epoch_accuracies)
    }

    /// Evaluates the model's performance on a validation set.
    /// TODO: Replace with real evaluation logic if needed.
    ///
    /// It iterates over the validation data in mini-batches, computes the average cross-entropy loss
    /// and accuracy, and returns them as a tuple (avg_loss, accuracy).
    ///
    /// # Arguments
    ///
    /// * `val_input` - NDArray of validation input token IDs (1D).
    /// * `val_target` - NDArray of validation target token IDs (1D), shifted by one relative to input.
    /// * `batch_size` - Batch size to use for evaluation.
    ///
    /// # Returns
    ///
    /// A tuple (avg_loss, accuracy) as f64 values.
    pub fn evaluate(&self, val_input: &crate::nab_array::NDArray, val_target: &crate::nab_array::NDArray, batch_size: usize) -> (f64, f64) {
        use crate::nab_loss::NabLoss;
        let n_tokens = val_input.size();
        let num_batches = (n_tokens as f64 / batch_size as f64).ceil() as usize;
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        for i in 0..num_batches {
            let start = i * batch_size;
            let end = ((i + 1) * batch_size).min(n_tokens);
            let batch_input = crate::nab_array::NDArray::from_vec(val_input.data()[start..end].to_vec());
            let batch_target = crate::nab_array::NDArray::from_vec(val_target.data()[start..end].to_vec());
            let output = self.forward(&batch_input);
            let loss = NabLoss::cross_entropy_loss_lm(&output, &batch_target);
            total_loss += loss;
            if end - start == 1 {
                let output_data = output.data();
                let (predicted_idx, _) = output_data.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();
                let target_idx = batch_target.data()[0] as usize;
                if predicted_idx == target_idx {
                    correct += 1;
                }
                total += 1;
            }
        }
        let avg_loss = total_loss / num_batches as f64;
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        (avg_loss, accuracy)
    }

    /// Samples a token sequence from the model given a prompt.
    /// TODO: Enhance sampling strategy if required.
    ///
    /// Starting from the given prompt (NDArray of token IDs), this function repeatedly runs forward passes
    /// and selects the token with the maximum probability to append to the sequence. It runs for the specified
    /// sample_length steps and returns the generated sequence as an NDArray.
    ///
    /// # Arguments
    ///
    /// * `prompt` - NDArray of token IDs to start the generation (1D).
    /// * `sample_length` - The number of tokens to generate.
    ///
    /// # Returns
    ///
    /// An NDArray of token IDs representing the generated sequence (concatenation of the prompt and generated tokens).
    pub fn sample(&self, prompt: &crate::nab_array::NDArray, sample_length: usize) -> crate::nab_array::NDArray {
        let mut generated = prompt.data().to_vec();
        let mut current_prompt = prompt.clone();
        for _ in 0..sample_length {
            let output = self.forward(&current_prompt);
            let output_data = output.data();
            let vocab = output.shape()[1];
            let seq_len = output.shape()[0];
            let last_token_probs = &output_data[(seq_len - 1) * vocab..seq_len * vocab];
            let (predicted_idx, _) = last_token_probs.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            generated.push(predicted_idx as f64);
            current_prompt = NDArray::from_vec(generated.clone());
        }
        NDArray::from_vec(generated)
    }

    /// Performs a forward pass and returns both hidden representation and logits.
    /// TODO: Replace with a more sophisticated mechanism if required.
    pub fn forward_with_hidden(&self, tokens: &crate::nab_array::NDArray) -> (crate::nab_array::NDArray, crate::nab_array::NDArray) {
        let embeddings = self.embedding_layer.forward(tokens);
        let hidden = self.transformer.forward(&embeddings);
        let logits = self.output_head.forward(&hidden);
        (hidden, logits)
    }

    /// Converts a 1D NDArray of token IDs into a one-hot encoded NDArray.
    /// TODO: Check edge cases and improve if needed.
    pub fn one_hot(target: &crate::nab_array::NDArray, vocab_size: usize) -> crate::nab_array::NDArray {
        let num_samples = target.shape()[0];
        let mut data = vec![0.0; num_samples * vocab_size];
        for (i, &val) in target.data().iter().enumerate() {
            let idx = val as usize;
            if idx < vocab_size {
                data[i * vocab_size + idx] = 1.0;
            }
        }
        crate::nab_array::NDArray::new(data, vec![num_samples, vocab_size])
    }

    /// Returns the index and maximum value of a 1-row NDArray (shape [1, vocab]).
    /// TODO: Validate input dimensions and improve if needed.
    pub fn argmax(array: &crate::nab_array::NDArray) -> (usize, f64) {
        let vocab = array.shape()[1];
        let row = &array.data()[0..vocab];
        let mut max_idx = 0;
        let mut max_val = row[0];
        for (i, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        (max_idx, max_val)
    }

    /// Saves the model in GGUF format for use with AnythingLLM.
    /// The model parameters (from the embedding layer, transformer, and output head) are extracted,
    /// structured into a GGUFModel, serialized to JSON, and written to the specified file path.
    /// TODO: Adjust the GGUF schema as necessary to fully comply with AnythingLLM requirements.
    pub fn save_to_gguf(&self, path: &str) -> std::io::Result<()> {
        // Construct a GGUF model representation (this is a simplified example)
        let gguf_model = GGUFModel {
            embedding_weights: self.embedding_layer.embedding_matrix.data().to_vec(),
            embedding_shape: self.embedding_layer.embedding_matrix.shape().to_vec(),
            transformer_weights: self.transformer.blocks.iter().flat_map(|block| {
                let mut v = Vec::new();
                v.extend_from_slice(block.attention.query.data());
                v.extend_from_slice(block.attention.key.data());
                v.extend_from_slice(block.attention.value.data());
                v
            }).collect(),
            output_head_weight: self.output_head.weight.data().to_vec(),
            output_head_bias: self.output_head.bias.data().to_vec(),
        };
        // Serialize the GGUF model to JSON (for demonstration purposes)
        let serialized = serde_json::to_string(&gguf_model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, serialized)
    }
}

// New struct representing the model in GGUF format.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GGUFModel {
    pub embedding_weights: Vec<f64>,
    pub embedding_shape: Vec<usize>,
    pub transformer_weights: Vec<f64>,
    pub output_head_weight: Vec<f64>,
    pub output_head_bias: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;

    // Remove the previous dummy() implementation for TinyLLMModel that used transmute.
    // Instead, define a dedicated dummy struct for testing the forward_with_hidden function.
    struct DummyTinyLLMModel;
    impl DummyTinyLLMModel {
        // This dummy forward_with_hidden simply returns the input as both hidden and logits.
        fn forward_with_hidden(&self, tokens: &NDArray) -> (NDArray, NDArray) {
            (tokens.clone(), tokens.clone())
        }
    }

    #[test]
    fn test_one_hot() {
        let input = NDArray::new(vec![1.0, 0.0, 2.0], vec![3]);
        let one_hot = TinyLLMModel::one_hot(&input, 3);
        let expected = NDArray::new(
            vec![0.0, 1.0, 0.0,
                 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0],
            vec![3, 3]
        );
        assert_eq!(one_hot.data(), expected.data());
        assert_eq!(one_hot.shape(), expected.shape());
    }

    #[test]
    fn test_argmax() {
        let array = NDArray::new(vec![0.1, 0.5, 0.3, 0.2], vec![1, 4]);
        let (index, value) = TinyLLMModel::argmax(&array);
        assert_eq!(index, 1);
        assert!((value - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_forward_with_hidden() {
        let input = NDArray::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let dummy_model = DummyTinyLLMModel;
        let (hidden, logits) = dummy_model.forward_with_hidden(&input);
        // Since DummyTinyLLMModel acts as the identity, both hidden and logits should equal input
        assert_eq!(hidden.data(), input.data());
        assert_eq!(logits.data(), input.data());
    }
}

#[cfg(test)]
mod training_loop_tests {
    use super::*;
    use crate::nab_array::NDArray;

    #[test]
    fn test_train_language_model() {
        // Create a dummy training sequence of 20 tokens
        // For simplicity, we use token IDs in range [0, 9]
        let tokens: Vec<f64> = (0..20).map(|i| (i % 10) as f64).collect();
        let input = NDArray::from_vec(tokens[..19].to_vec());
        let target = NDArray::from_vec(tokens[1..20].to_vec());

        // Create a tiny LLM model with small dimensions
        let vocab_size = 10;
        let embedding_dim = 4;
        let n_layers = 1;
        let mut model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);

        // Train for 2 epochs with batch_size 5 and learning_rate 0.1
        let (loss_history, acc_history) = model.train_language_model(&input, &target, 5, 2, 0.1);

        // Check that we have 2 entries in loss_history and acc_history
        assert_eq!(loss_history.len(), 2);
        assert_eq!(acc_history.len(), 2);

        // Loss values should be finite
        for loss in loss_history {
            assert!(loss.is_finite());
        }
    }
}

#[cfg(test)]
mod evaluation_tests {
    use super::*;
    use crate::nab_array::NDArray;

    #[test]
    fn test_evaluate() {
        // Create a dummy validation set of 10 tokens
        let tokens: Vec<f64> = (0..10).map(|i| (i % 5) as f64).collect();
        let val_input = NDArray::from_vec(tokens[..9].to_vec());
        let val_target = NDArray::from_vec(tokens[1..10].to_vec());
        
        // Create a tiny LLM model
        let vocab_size = 5;
        let embedding_dim = 4;
        let n_layers = 1;
        let model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);
        
        let (avg_loss, accuracy) = model.evaluate(&val_input, &val_target, 3);
        // Check that loss is finite and accuracy is between 0 and 1
        assert!(avg_loss.is_finite());
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_sample() {
        // Create a dummy prompt of 3 tokens
        let prompt = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        // Create a tiny LLM model with limited vocab
        let vocab_size = 10;
        let embedding_dim = 4;
        let n_layers = 1;
        let model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);
        
        let generated = model.sample(&prompt, 5);
        // The generated sequence should have length equal to prompt length + sample_length
        assert_eq!(generated.data().len(), 3 + 5);
    }
}

// ---- New helper functions for sampling strategies ----

// Function: sample_from_distribution
// This function samples a token index from a given probability distribution, which is an NDArray with shape [1, vocab].
// It iterates over the distribution, computes the cumulative probability, and returns the first index where the cumulative probability exceeds a random threshold.
// Returns: usize token index.
pub fn sample_from_distribution(probs: &NDArray) -> usize {
    let shape = probs.shape();
    assert_eq!(shape.len(), 2, "Expected a 2D array for distribution");
    let vocab = shape[1];
    let mut cumulative = 0.0;
    let r: f64 = rand::random(); // random number in [0, 1)
    for j in 0..vocab {
        cumulative += probs.data()[j];
        if r < cumulative {
            return j;
        }
    }
    vocab - 1 // fallback in case of numerical issues
}

// Function: concatenate
// This function concatenates a token (represented as a usize) to an existing token sequence (an NDArray assumed to be 1D).
// It returns a new NDArray with the new token appended to the end of the sequence.
pub fn concatenate(token_seq: &NDArray, token: usize) -> NDArray {
    let mut new_data = token_seq.data().to_vec();
    new_data.push(token as f64);
    NDArray::from_vec(new_data)
}

// ---- End of new helper functions ----

// ---- New unit tests for the sampling helper functions ----
#[cfg(test)]
mod sampling_tests {
    use super::*;
    
    #[test]
    fn test_sample_from_distribution() {
        // Create a simple probability distribution: [0.1, 0.3, 0.6]
        // The NDArray should be 2D with one row and 3 columns
        let probs = NDArray::from_vec(vec![0.1, 0.3, 0.6]).reshape(&[1, 3]).unwrap();
        // Sample a token index
        let idx = sample_from_distribution(&probs);
        // The returned index should be between 0 and 2
        assert!(idx < 3, "Sampled index {} is out of expected range", idx);
    }
    
    #[test]
    fn test_concatenate() {
        // Create an NDArray representing a token sequence [1.0, 2.0, 3.0]
        let token_seq = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        // Concatenate token '4'
        let new_seq = concatenate(&token_seq, 4);
        // The expected sequence is [1.0, 2.0, 3.0, 4.0]
        assert_eq!(new_seq.data(), &vec![1.0, 2.0, 3.0, 4.0]);
    }
}

// ---- End of new unit tests ----

// ---- New helper function for temperature scaling ----

/// adjust_with_temperature applies temperature scaling to a probability distribution represented as an NDArray.
/// It computes the natural logarithm of each probability, divides by the temperature, exponentiates, and then normalizes the result.
/// This can be used to control the randomness of the sampling process (lower temperature means a sharper distribution).
pub fn adjust_with_temperature(probs: &NDArray, temperature: f64) -> NDArray {
    // Convert probabilities to logits using natural logarithm
    let logits: Vec<f64> = probs.data().iter().map(|&p| p.ln()).collect();
    // Divide logits by temperature and exponentiate
    let scaled: Vec<f64> = logits.iter().map(|&l| (l / temperature).exp()).collect();
    // Create an NDArray with the scaled values
    let scaled_nd = NDArray::from_vec(scaled).reshape(probs.shape()).unwrap();
    // Normalize each row of the NDArray
    let shape = scaled_nd.shape();
    let batch = shape[0];
    let vocab = shape[1];
    let mut normalized = Vec::with_capacity(scaled_nd.data().len());
    for i in 0..batch {
        let start = i * vocab;
        let end = start + vocab;
        let row = &scaled_nd.data()[start..end];
        let sum: f64 = row.iter().sum();
        for &val in row {
            normalized.push(val / sum);
        }
    }
    NDArray::from_vec(normalized).reshape(probs.shape()).unwrap()
}

// ---- End of new helper function ----

// ---- New unit tests for temperature scaling ----
#[cfg(test)]
mod temperature_tests {
    use super::*;

    #[test]
    fn test_adjust_with_temperature() {
        // Create a simple probability distribution in 2D: one row, 3 columns
        let probs = NDArray::from_vec(vec![0.2, 0.3, 0.5]).reshape(&[1, 3]).unwrap();
        let temperature = 0.8;
        let adjusted = adjust_with_temperature(&probs, temperature);
        // Check that the output shape matches the input shape
        assert_eq!(adjusted.shape(), probs.shape());

        // The sum of probabilities in each row should be approximately 1
        let sum: f64 = adjusted.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Row sum is not normalized, got {}", sum);
    }
}
// ---- End of new unit tests ---- 
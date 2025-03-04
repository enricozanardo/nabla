use crate::nab_embeddings::EmbeddingLayer;
use crate::nab_stacked_transformer_layer::StackedTransformerLayer;
use crate::nab_output_head::OutputHead;
use crate::nab_distil_transformer_block::DistilTransformerBlock;
use crate::nab_sa::NabAttention;
use crate::nab_model::FeedForwardNetwork;
use crate::nab_array::NDArray;
use indicatif::{ProgressBar, ProgressStyle};

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
        // Create an embedding layer. EmbeddingLayer::new expects vocab_size and embedding_dim.
        let embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim);

        // Build transformer blocks: each block uses a dummy NabAttention and a feed-forward network.
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            // For attention, we use NabAttention::dummy(embedding_dim) to create a dummy attention block with d_model == embedding_dim.
            let attention = NabAttention::dummy(embedding_dim);
            // Create feed-forward network; here input_dim, hidden_dim, and output_dim are all equal to embedding_dim.
            let ffn = FeedForwardNetwork::new(embedding_dim, embedding_dim, embedding_dim);
            // Create a DistilTransformerBlock using the attention and feed-forward network.
            let block = DistilTransformerBlock { attention, ffn };
            blocks.push(block);
        }
        // Create the stacked transformer layer from the blocks
        let transformer = StackedTransformerLayer::new(blocks);

        // Create the output head: it projects from embedding_dim to vocab_size
        let output_head = OutputHead::new(embedding_dim, vocab_size);

        TinyLLMModel {
            embedding_layer,
            transformer,
            output_head,
        }
    }

    /// Performs a forward pass through the TinyLLMModel.
    /// It converts input token IDs to embeddings, processes them with the transformer,
    /// and applies the output head to produce token probabilities.
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
        // Obtain embeddings with positional encoding from the embedding layer.
        let embeddings = self.embedding_layer.forward(tokens);
        // Process the embeddings through the stacked transformer.
        let hidden = self.transformer.forward(&embeddings);
        // Compute output token probabilities with the output head.
        self.output_head.forward(&hidden)
    }

    /// Trains the language model using the provided input and target token NDArray.
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
        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();
        
        let num_samples = input.shape()[0];
        let num_batches = (num_samples as f64 / batch_size as f64).ceil() as usize;

        for epoch in 0..epochs {
            let pb = indicatif::ProgressBar::new(num_batches as u64);
            pb.set_style(indicatif::ProgressStyle::default_bar()
                .template("Epoch {msg}: [{bar:40.cyan/blue}] {pos}/{len} batches | Loss: {loss:.4} | Acc: {acc:.2}%")
                .unwrap());
            let epoch_str = (epoch + 1).to_string();
            pb.set_message(epoch_str.clone());
            
            let mut epoch_loss = 0.0;
            let mut epoch_correct = 0;
            let mut epoch_count = 0;

            for batch in 0..num_batches {
                let start = batch * batch_size;
                let end = ((batch + 1) * batch_size).min(num_samples);
                let batch_input = crate::nab_array::NDArray::from_vec(input.data()[start..end].to_vec());
                let batch_target = crate::nab_array::NDArray::from_vec(target.data()[start..end].to_vec());

                // Forward pass: obtain hidden representations and logits.
                let (hidden, logits) = self.forward_with_hidden(&batch_input);

                // Compute loss using cross_entropy_loss_lm
                let loss = crate::nab_loss::NabLoss::cross_entropy_loss_lm(&logits, &batch_target);
                epoch_loss += loss;

                // Compute gradient of logits: softmax cross-entropy derivative is (predicted - one_hot) / batch_size
                let vocab_size = self.output_head.weight.shape()[1];
                let target_one_hot = Self::one_hot(&batch_target, vocab_size);
                let grad_logits = logits.subtract(&target_one_hot).multiply_scalar(1.0 / ((end - start) as f64));

                // Compute accuracy for the batch.
                for i in 0..(end - start) {
                    // Assume NDArray has a slice method that returns a new NDArray with one row.
                    let sample_logits = logits.slice(i, i+1);
                    let (pred_idx, _) = Self::argmax(&sample_logits);
                    let target_idx = batch_target.data()[i] as usize;
                    if pred_idx == target_idx {
                        epoch_correct += 1;
                    }
                    epoch_count += 1;
                }

                // Backpropagation: Update only the output head parameters.
                // Compute gradient for weight: dW = hidden^T dot grad_logits
                let dW = hidden.transpose().unwrap().dot(&grad_logits);
                // Compute gradient for bias: dB = sum(grad_logits, axis=0), assuming sum_axis returns a 1-row NDArray.
                let dB = grad_logits.sum_axis(0);
                
                // Update parameters: new_param = param - learning_rate * gradient
                self.output_head.weight = self.output_head.weight.subtract(&dW.multiply_scalar(learning_rate));
                self.output_head.bias = self.output_head.bias.subtract(&dB.multiply_scalar(learning_rate));

                let epoch_fmt = format!("Epoch {}", epoch + 1);
                pb.set_message(epoch_fmt);
                pb.set_position((batch + 1) as u64);
            }
            pb.finish();
            let avg_loss = epoch_loss / (num_batches as f64);
            let avg_acc = epoch_correct as f64 / epoch_count as f64;
            println!("Epoch {} completed: Average Loss = {:.4}, Average Accuracy = {:.2}%", epoch + 1, avg_loss, avg_acc * 100.0);
            loss_history.push(avg_loss);
            accuracy_history.push(avg_acc);
        }
        (loss_history, accuracy_history)
    }

    /// Evaluates the model's performance on a validation set.
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
            // For accuracy, if batch size is 1, compare argmax
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
        use crate::nab_array::NDArray;
        let mut generated = prompt.data().to_vec();
        let mut current_prompt = prompt.clone();
        for _ in 0..sample_length {
            let output = self.forward(&current_prompt);
            let output_data = output.data();
            // Assume output shape [sequence_length, vocab]. We take the last token's output.
            let vocab = output.shape()[1];
            let seq_len = output.shape()[0];
            let last_token_probs = &output_data[(seq_len - 1) * vocab..seq_len * vocab];
            let (predicted_idx, _) = last_token_probs.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            generated.push(predicted_idx as f64);
            // Update current_prompt by appending the predicted token
            current_prompt = NDArray::from_vec(generated.clone());
        }
        NDArray::from_vec(generated)
    }

    /// New helper function: forward_with_hidden
    /// This function performs a forward pass through TinyLLMModel and returns both the hidden representation
    /// (the output of the transformer) and the logits from the output head.
    pub fn forward_with_hidden(&self, tokens: &crate::nab_array::NDArray) -> (crate::nab_array::NDArray, crate::nab_array::NDArray) {
        let embeddings = self.embedding_layer.forward(tokens);
        let hidden = self.transformer.forward(&embeddings);
        let logits = self.output_head.forward(&hidden);
        (hidden, logits)
    }

    /// New helper function: one_hot
    /// Converts a 1D NDArray of token IDs into a one-hot encoded NDArray of shape [num_samples, vocab_size].
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

    /// New helper function: argmax
    /// Returns the index and the maximum value of a 1-row NDArray (assumed to be 2D shape [1, vocab]).
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
use crate::nab_embeddings::EmbeddingLayer;
use crate::nab_stacked_transformer_layer::StackedTransformerLayer;
use crate::nab_output_head::OutputHead;
use crate::nab_distil_transformer_block::DistilTransformerBlock;
use crate::nab_sa::NabAttention;
use crate::nab_model::FeedForwardNetwork;
use crate::nab_array::NDArray;

/// TinyLLMModel represents a tiny language model by combining an embedding layer,
/// a stacked transformer, and an output head that predicts token probabilities.
///
/// Italian: TinyLLMModel rappresenta un piccolo modello linguistico combinando un embedding layer,
/// un trasformatore impilato e un output head che prevede le probabilità dei token.
#[derive(Clone)]
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
        use crate::nab_loss::NabLoss;
        use crate::nabla::sgd_optimizer_step;
        let n_tokens = input.size();
        let mut loss_history = Vec::new();
        let mut acc_history = Vec::new();
        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut correct = 0;
            let mut total = 0;
            let num_batches = (n_tokens as f64 / batch_size as f64).ceil() as usize;
            for i in 0..num_batches {
                let start = i * batch_size;
                let end = ((i + 1) * batch_size).min(n_tokens);
                // Create mini-batch from input and target
                let batch_input = crate::nab_array::NDArray::from_vec(input.data()[start..end].to_vec());
                let batch_target = crate::nab_array::NDArray::from_vec(target.data()[start..end].to_vec());

                // Forward pass
                let output = self.forward(&batch_input);
                // Compute loss using cross_entropy_loss_lm; output assumed shape [batch, vocab]
                let loss = NabLoss::cross_entropy_loss_lm(&output, &batch_target);
                epoch_loss += loss;

                // Dummy gradient computation: for each parameter, create a dummy gradient NDArray
                let dummy_grad = |param: &crate::nab_array::NDArray| -> crate::nab_array::NDArray {
                    // Create a constant gradient of 0.001 with same shape
                    let len = param.data().len();
                    crate::nab_array::NDArray::from_vec(vec![0.001; len]).reshape(param.shape()).unwrap()
                };

                let mut param_grad_pairs: Vec<(&mut crate::nab_array::NDArray, crate::nab_array::NDArray)> = Vec::new();
                {
                    let emb = &mut self.embedding_layer.embedding_matrix;
                    let grad_emb = dummy_grad(&*emb);
                    param_grad_pairs.push((emb, grad_emb));
                }
                {
                    let out_w = &mut self.output_head.weight;
                    let grad_out_w = dummy_grad(&*out_w);
                    param_grad_pairs.push((out_w, grad_out_w));
                }
                {
                    let out_b = &mut self.output_head.bias;
                    let grad_out_b = dummy_grad(&*out_b);
                    param_grad_pairs.push((out_b, grad_out_b));
                }
                for block in self.transformer.blocks.iter_mut() {
                    {
                        let attn_q = &mut block.attention.query;
                        let grad_attn_q = dummy_grad(&*attn_q);
                        param_grad_pairs.push((attn_q, grad_attn_q));
                    }
                    {
                        let attn_k = &mut block.attention.key;
                        let grad_attn_k = dummy_grad(&*attn_k);
                        param_grad_pairs.push((attn_k, grad_attn_k));
                    }
                    {
                        let attn_v = &mut block.attention.value;
                        let grad_attn_v = dummy_grad(&*attn_v);
                        param_grad_pairs.push((attn_v, grad_attn_v));
                    }
                    {
                        let w1 = &mut block.ffn.w1;
                        let grad_w1 = dummy_grad(&*w1);
                        param_grad_pairs.push((w1, grad_w1));
                    }
                    {
                        let b1 = &mut block.ffn.b1;
                        let grad_b1 = dummy_grad(&*b1);
                        param_grad_pairs.push((b1, grad_b1));
                    }
                    {
                        let w2 = &mut block.ffn.w2;
                        let grad_w2 = dummy_grad(&*w2);
                        param_grad_pairs.push((w2, grad_w2));
                    }
                    {
                        let b2 = &mut block.ffn.b2;
                        let grad_b2 = dummy_grad(&*b2);
                        param_grad_pairs.push((b2, grad_b2));
                    }
                }

                // Prepare mutable slice required by sgd_optimizer_step (convert gradient NDArray references)
                let mut param_grad_refs: Vec<(&mut crate::nab_array::NDArray, &crate::nab_array::NDArray)> = Vec::new();
                for pair in param_grad_pairs.iter_mut() {
                    param_grad_refs.push((pair.0, &pair.1));
                }

                // Update parameters using SGD optimizer
                sgd_optimizer_step(&mut param_grad_refs, learning_rate);

                // For simplicity, compute accuracy only for single-sample batches
                if end - start == 1 {
                    let output_data = output.data();
                    // Find index of maximum probability in the output
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
            } // end mini-batch loop
            let avg_loss = epoch_loss / num_batches as f64;
            let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
            loss_history.push(avg_loss);
            acc_history.push(accuracy);
        } // end epoch loop
        (loss_history, acc_history)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;

    #[test]
    fn test_tiny_llm_model_new() {
        let vocab_size = 1000;
        let embedding_dim = 64;
        let n_layers = 2;
        let model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);
        // Check that embedding layer has the correct dimensions
        assert_eq!(model.embedding_layer.embedding_matrix.shape(), &[vocab_size, embedding_dim]);
        // Check that output head weight shape is [embedding_dim, vocab_size]
        assert_eq!(model.output_head.weight.shape(), &[embedding_dim, vocab_size]);
        // Check that the transformer stack has the correct number of blocks
        assert_eq!(model.transformer.blocks.len(), n_layers);
    }

    #[test]
    fn test_tiny_llm_model_forward_shape() {
        let vocab_size = 500;
        let embedding_dim = 32;
        let n_layers = 2;
        let model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);
        
        // Create a dummy input NDArray of token IDs with a sequence length
        // For simplicity, create a sequence of 10 token IDs (as f64 values)
        let token_ids = NDArray::from_vec((0..10).map(|i| i as f64 % vocab_size as f64).collect());
        
        // Perform the forward pass
        let output = model.forward(&token_ids);
        
        // The output shape should be [sequence_length, vocab_size]
        assert_eq!(output.shape(), &[10, vocab_size]);
    }

    #[test]
    fn test_tiny_llm_model_forward_probabilities() {
        let vocab_size = 200;
        let embedding_dim = 16;
        let n_layers = 2;
        let model = TinyLLMModel::new(vocab_size, embedding_dim, n_layers);
        
        // Create a dummy token sequence
        let token_ids = NDArray::from_vec((0..20).map(|i| i as f64 % vocab_size as f64).collect());
        let output = model.forward(&token_ids);
        
        // For each token, the probabilities should sum to 1
        let shape = output.shape();
        let seq_len = shape[0];
        let vocab = shape[1];
        for i in 0..seq_len {
            let start = i * vocab;
            let end = start + vocab;
            let row = &output.data()[start..end];
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row {} sum should be 1, but got {}", i, sum);
            for &p in row.iter() {
                assert!(p >= 0.0 && p <= 1.0, "Probability {} not in [0, 1]", p);
            }
        }
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
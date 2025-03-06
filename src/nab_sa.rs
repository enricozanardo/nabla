use crate::nab_array::NDArray;
use crate::nab_tokenizer::NabTokenizer;
use serde;
// use std::fmt;

/// NabAttention is responsible for computing attention weights for the transformer.
///
/// Italian: NabAttention calcola i pesi di attenzione per il trasformatore.
#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct NabAttention {
    pub query: NDArray,     // Query weight matrix [d_model, d_model]
    pub key: NDArray,       // Key weight matrix [d_model, d_model]
    pub value: NDArray,     // Value weight matrix [d_model, d_model]
    pub num_heads: usize,   // Number of attention heads
    pub d_model: usize,     // Model dimension
}

impl NabAttention {
    /// Creates a new attention module with the given weight matrices
    /// 
    /// # Arguments
    /// * `query` - Query weight matrix of shape [d_model, d_model]
    /// * `key` - Key weight matrix of shape [d_model, d_model]
    /// * `value` - Value weight matrix of shape [d_model, d_model]
    /// * `num_heads` - Number of attention heads
    /// * `d_model` - Model dimension
    /// 
    /// # Returns
    /// * `Self` - A new NabAttention instance
    pub fn new(query: NDArray, key: NDArray, value: NDArray, num_heads: usize, d_model: usize) -> Self {
        // Verify inputs are 2D matrices
        assert_eq!(query.ndim(), 2, "Query must be a 2D matrix");
        assert_eq!(key.ndim(), 2, "Key must be a 2D matrix"); 
        assert_eq!(value.ndim(), 2, "Value must be a 2D matrix");

        // Verify compatible dimensions
        assert_eq!(key.shape()[0], query.shape()[0], "Key and query must have same input dimension (d_model)");
        assert_eq!(key.shape()[1], query.shape()[1], "Key and query must have same projection dimension");
        assert_eq!(value.shape()[0], query.shape()[0], "Value must have same input dimension as query (d_model)");
        
        // Verify multi-head compatibility
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        // Now, expect weight matrices to be of shape [d_model, d_model]
        assert_eq!(query.shape()[0], d_model, "Query input dimension must be d_model");
        assert_eq!(query.shape()[1], d_model, "Query projection dimension must be d_model");

        Self { query, key, value, num_heads, d_model }
    }

    /// Calculates scaled dot-product attention for a single head
    fn attention_head(&self, embeddings: &NDArray, head_idx: usize) -> (NDArray, NDArray) {
        let head_dim = self.d_model / self.num_heads;
        let start_idx = head_idx * head_dim;
        let end_idx = start_idx + head_dim;

        // To select the projection for the given head, we want to slice the projection dimension (columns).
        // We can achieve this by transposing the weight matrix (so columns become rows), then slicing, and then transposing back.
        let q_weights = self.query.transpose().unwrap().slice(start_idx, end_idx).transpose().unwrap();
        let k_weights = self.key.transpose().unwrap().slice(start_idx, end_idx).transpose().unwrap();
        let v_weights = self.value.transpose().unwrap().slice(start_idx, end_idx).transpose().unwrap();

        // Project inputs: embeddings [seq_len, d_model] dot weight [d_model, head_dim] gives [seq_len, head_dim]
        let query_matrix = embeddings.dot(&q_weights);
        let key_matrix = embeddings.dot(&k_weights);
        let value_matrix = embeddings.dot(&v_weights);

        // Calculate attention scores and scale
        let attention_scores = query_matrix.dot(&key_matrix.transpose().unwrap()); // [seq_len, seq_len]
        let scaling_factor = (head_dim as f64).sqrt();
        let scaled_scores = attention_scores.divide_scalar(scaling_factor);

        // Compute softmax with numerical stability: subtract row max, exponentiate, then normalize row-wise
        let rows = scaled_scores.shape()[0];
        let cols = scaled_scores.shape()[1];
        // Compute row-wise maximums
        let mut max_vals = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row = &scaled_scores.data()[start..end];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            max_vals.push(max_val);
        }
        
        // Compute exponentials with shifted scores
        let mut exp_data = Vec::with_capacity(scaled_scores.data().len());
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row = &scaled_scores.data()[start..end];
            let max_val = max_vals[i];
            for &val in row {
                exp_data.push((val - max_val).exp());
            }
        }
        let exp_scores = NDArray::new(exp_data, scaled_scores.shape().to_vec());
        
        // Row-wise normalization of exponentials
        let mut probs_data = Vec::with_capacity(exp_scores.data().len());
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row = &exp_scores.data()[start..end];
            let row_sum: f64 = row.iter().sum();
            for &val in row {
                probs_data.push(val / row_sum);
            }
        }
        // Correction: re-normalize each row to mitigate floating-point errors
        let mut corrected_probs_data = probs_data.clone();
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row_sum: f64 = corrected_probs_data[start..end].iter().sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                for j in start..end {
                    corrected_probs_data[j] /= row_sum;
                }
            }
            // println!("Debug: row {} corrected sum: {}", i, corrected_probs_data[start..end].iter().sum::<f64>());
        }
        let attention_probs = NDArray::new(corrected_probs_data, exp_scores.shape().to_vec());

        // Calculate weighted values
        let output = attention_probs.dot(&value_matrix); // [seq_len, head_dim]
        
        (attention_probs, output)
    }

    /// Calculates scaled dot-product attention
    /// 
    /// Implements the attention mechanism as defined in "Attention Is All You Need":
    /// Attention(Q, K, V) = softmax(QK^T/√d_k)V
    /// 
    /// # Arguments
    /// * `embeddings` - Input token embeddings of shape [seq_length, d_model]
    /// 
    /// # Returns
    /// * `(NDArray, NDArray)` - Tuple containing:
    ///   - Attention probabilities of shape [seq_length, seq_length]
    ///   - Output tensor of shape [seq_length, d_v]
    pub fn forward(&self, embeddings: &NDArray) -> (NDArray, NDArray) {
        // For single-head attention, just use the first head
        self.attention_head(embeddings, 0)
    }

    /// Calculates multi-head attention
    /// 
    /// # Arguments
    /// * `embeddings` - Input token embeddings of shape [seq_length, d_model]
    /// 
    /// # Returns
    /// * `(Vec<NDArray>, NDArray)` - Tuple containing:
    ///   - Vector of attention probabilities for each head
    ///   - Concatenated and projected output tensor
    pub fn forward_multihead(&self, embeddings: &NDArray) -> (Vec<NDArray>, NDArray) {
        let mut attention_probs = Vec::with_capacity(self.num_heads);
        let mut head_outputs = Vec::with_capacity(self.num_heads);

        // Calculate attention for each head
        for head_idx in 0..self.num_heads {
            let (probs, output) = self.attention_head(embeddings, head_idx);
            attention_probs.push(probs);
            head_outputs.push(output);
        }

        // Concatenate head outputs
        let mut concatenated_data = Vec::new();
        for i in 0..embeddings.shape()[0] {
            for head in &head_outputs {
                concatenated_data.extend_from_slice(&head.data()[i * (self.d_model / self.num_heads)..
                                                               (i + 1) * (self.d_model / self.num_heads)]);
            }
        }

        let concatenated = NDArray::new(concatenated_data, 
                                      vec![embeddings.shape()[0], self.d_model]);

        (attention_probs, concatenated)
    }

    /// Calculates the sine component of positional encoding
    /// 
    /// # Arguments
    /// * `pos` - Position in the sequence
    /// * `i` - Dimension index
    /// * `d_model` - Model dimension
    fn sin_positional_encoding(pos: usize, i: usize, d_model: usize) -> f64 {
        let denominator = f64::powf(10000.0, (2.0 * i as f64) / d_model as f64);
        f64::sin(pos as f64 / denominator)
    }

    /// Calculates the cosine component of positional encoding
    /// 
    /// # Arguments
    /// * `pos` - Position in the sequence
    /// * `i` - Dimension index
    /// * `d_model` - Model dimension
    fn cos_positional_encoding(pos: usize, i: usize, d_model: usize) -> f64 {
        let denominator = f64::powf(10000.0, (2.0 * i as f64) / d_model as f64);
        f64::cos(pos as f64 / denominator)
    }

    /// Generates positional encoding matrix for the input sequence
    /// 
    /// # Arguments
    /// * `seq_length` - Length of the input sequence
    /// * `d_model` - Model dimension
    /// 
    /// # Returns
    /// * NDArray - Positional encoding matrix of shape [seq_length, d_model]
    pub fn generate_positional_encoding(seq_length: usize, d_model: usize) -> NDArray {
        let mut encoding = vec![0.0; seq_length * d_model];
        
        for pos in 0..seq_length {
            for i in 0..(d_model/2) {
                let sin_val = Self::sin_positional_encoding(pos, i, d_model);
                let cos_val = Self::cos_positional_encoding(pos, i, d_model);
                
                encoding[pos * d_model + 2*i] = sin_val;
                encoding[pos * d_model + 2*i + 1] = cos_val;
            }
            
            // Handle odd d_model case
            if d_model % 2 != 0 {
                let last_idx = d_model - 1;
                encoding[pos * d_model + last_idx] = Self::sin_positional_encoding(pos, last_idx, d_model);
            }
        }
        
        NDArray::new(encoding, vec![seq_length, d_model])
    }

    /// Adds positional encoding to the input embeddings
    /// 
    /// # Arguments
    /// * `embeddings` - Input token embeddings of shape [seq_length, d_model]
    /// 
    /// # Returns
    /// * NDArray - Embeddings with positional encoding added
    pub fn add_positional_encoding(embeddings: &NDArray) -> NDArray {
        let seq_length = embeddings.shape()[0];
        let d_model = embeddings.shape()[1];
        
        let positional_encoding = Self::generate_positional_encoding(seq_length, d_model);
        embeddings + &positional_encoding
    }

    /// Prints detailed information about tokens and their embeddings
    /// 
    /// # Arguments
    /// * `tokens` - NDArray containing token IDs
    /// * `embeddings` - NDArray containing token embeddings
    /// * `tokenizer` - Reference to the NabTokenizer
    pub fn print_token_info(tokens: &NDArray, embeddings: &NDArray, tokenizer: &NabTokenizer) {
        println!("\nToken Information:");
        println!("Total tokens: {}", tokens.shape()[0]);
        println!("Embedding dimension: {}", embeddings.shape()[1]);
        println!("\nDetailed token breakdown:");
        println!("{:-<60}", "");  // Print separator line
        
        for i in 0..tokens.shape()[0] {
            let token_id = tokens.data()[i];
            let token_text = tokenizer.decode(&NDArray::from_vec(vec![token_id]), false);
            let token_emb = embeddings.slice(i, i + 1);
            
            println!("Position: {}", i);
            println!("Token text: '{}'", token_text.trim());
            println!("Token ID: {}", token_id);
            println!("Embedding: {:?}", token_emb.data());
            
            // Calculate and print key vector if attention weights are available
            // if let Some(attention) = self.get_attention() {
            //     let key_vector = token_emb.dot(&attention.key.transpose().unwrap());
            //     println!("Key vector: {:?}", key_vector.data());
            // }
            
            // println!("{:-<60}", "");  // Print separator line
        }
    }

    /// Creates a dummy NabAttention instance with random weights.
    /// TODO: Replace with a full attention mechanism if needed.
    pub fn dummy(d_model: usize) -> Self {
        // Initialize query, key, and value weight matrices with random normal values
        let query = NDArray::randn_2d(d_model, d_model);
        let key = NDArray::randn_2d(d_model, d_model);
        let value = NDArray::randn_2d(d_model, d_model);
        NabAttention {
            query,
            key,
            value,
            num_heads: 1, // For simplicity, using single head; can be extended later
            d_model,
        }
    }
}

/// TransformerBlock wraps a multi-head self-attention layer.
///
/// # Fields
/// * `attention` - The NabAttention instance for multi-head attention.
/// * `d_model` - The model dimension.
pub struct TransformerBlock {
    pub attention: NabAttention,
    pub d_model: usize,
}

impl TransformerBlock {
    /// Creates a new TransformerBlock with the given NabAttention instance.
    ///
    /// # Arguments
    /// * `attention` - The NabAttention instance for multi-head attention.
    ///
    /// # Returns
    /// A new TransformerBlock instance.
    pub fn new(attention: NabAttention) -> Self {
        let d_model = attention.d_model;
        Self { attention, d_model }
    }

    /// Processes input embeddings through the multi-head self-attention layer.
    ///
    /// # Arguments
    /// * `embeddings` - Input token embeddings of shape [seq_length, d_model].
    ///
    /// # Returns
    /// A tuple containing:
    /// - Attention probabilities for each head.
    /// - The output tensor after attention.
    pub fn forward(&self, embeddings: &NDArray) -> (Vec<NDArray>, NDArray) {
        self.attention.forward_multihead(embeddings)
    }
}

/*
   NabAttention Module Tests
   Test module for NabAttention with detailed bilingual comments
   Test del modulo NabAttention con commenti bilingue (inglese e italiano)
*/

#[cfg(test)]
mod tests {
    use super::*;

    /// Test for single-head attention.
    /// Test per l'attention a testa singola.
    #[test]
    fn test_single_head_attention() {
        // Set parameters: embedding dimension 6, one head (thus head_dim = 6).
        // Impostiamo: dimensione embedding 6, una testa (head_dim = 6).
        let embedding_dim = 6;
        let num_heads = 1;
        let _head_dim = embedding_dim / num_heads; // head_dim = 6

        // Generate sample embeddings with shape [4, 6].
        // Genera embeddings di esempio con forma [4, 6].
        let embeddings = NDArray::rand_uniform(&[4, embedding_dim]).multiply_scalar(0.1);

        // Create weight matrices with the correct dimensions [6, 6].
        // Crea matrici di pesi con dimensioni [6, 6].
        let query_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let key_weights   = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let value_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);

        // Create the NabAttention module.
        // Crea il modulo NabAttention.
        let attention = NabAttention::new(
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            embedding_dim
        );

        // Execute the forward pass for single-head attention.
        // Esegui il forward pass per l'attention a testa singola.
        let (attention_probs, output) = attention.forward(&embeddings);

        // Verify that the attention probability matrix has shape [seq_len, seq_len] = [4, 4].
        // Verifica che la matrice delle probabilità sia di forma [4, 4].
        assert_eq!(attention_probs.shape(), &[4, 4], "Attention matrix shape should be [4, 4]");

        // Verify that the output has shape [seq_len, head_dim] = [4, 6].
        // Verifica che l'output abbia forma [4, 6].
        assert_eq!(output.shape(), &[4, _head_dim], "Output shape should be [4, head_dim]");

        // Check that each row of the attention probabilities sums to approximately 1 (tolerance 1e-3).
        let row_sums = attention_probs.sum_axis(1);
        for (i, &sum) in row_sums.data().iter().enumerate() {
            assert!((sum - 1.0).abs() < 1e-3, "Row {} sum {} should be approximately 1", i, sum);
        }

        // Ensure all attention probabilities are in the range [0, 1].
        // Verifica che tutte le probabilità siano comprese tra 0 e 1.
        for &prob in attention_probs.data() {
            assert!(prob >= 0.0, "Attention probabilities should be non-negative");
            assert!(prob <= 1.0, "Attention probabilities should be <= 1");
        }
    }

    /// Test for multi-head attention.
    /// Test per l'attention multi-testa.
    #[test]
    fn test_multi_head_attention() {
        // Set parameters: embedding dimension 6, two heads (thus head_dim = 3).
        // Impostiamo: dimensione embedding 6, due teste (head_dim = 3).
        let embedding_dim = 6;
        let num_heads = 2;
        let _head_dim = embedding_dim / num_heads; // head_dim = 3

        // Generate sample embeddings with shape [4, 6].
        // Genera embeddings di esempio con forma [4, 6].
        let embeddings = NDArray::rand_uniform(&[4, embedding_dim]).multiply_scalar(0.1);

        // Create weight matrices with the correct dimensions [6, 6].
        // Crea matrici di pesi con forma [6, 6].
        let query_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let key_weights   = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let value_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);

        // Create the NabAttention module for multi-head attention.
        // Crea il modulo NabAttention per l'attention multi-testa.
        let attention = NabAttention::new(
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            embedding_dim
        );

        // Execute the forward pass for multi-head attention.
        // Esegui il forward pass per l'attention multi-testa.
        let (attention_probs, output) = attention.forward_multihead(&embeddings);

        // Verify that the number of attention probability matrices equals the number of heads.
        // Verifica che il numero di matrici di probabilità corrisponda al numero di teste.
        assert_eq!(attention_probs.len(), num_heads, "Should have {} attention probability matrices", num_heads);

        // For each head, verify the probability matrix has shape [4, 4] and each row sums to approximately 1 (tolerance 1e-3).
        // Per ogni testa, verifica che la matrice delle probabilità abbia forma [4, 4] e che ogni riga sommi circa 1 (tolleranza 1e-3).
        for (i, head_probs) in attention_probs.iter().enumerate() {
            assert_eq!(head_probs.shape(), &[4, 4], "Head {} should have attention matrix shape [4, 4]", i);
            let row_sums = head_probs.sum_axis(1);
            for (j, &sum) in row_sums.data().iter().enumerate() {
                assert!((sum - 1.0).abs() < 1e-3, "Head {} row {} sum {} should be approximately 1", i, j, sum);
            }
            // Check that probabilities are in the range [0, 1].
            for &prob in head_probs.data() {
                assert!(prob >= 0.0, "Head {} attention probabilities should be non-negative", i);
                assert!(prob <= 1.0, "Head {} attention probabilities should be <= 1", i);
            }
        }

        // Verify that the concatenated output has shape [4, 6] (original embedding dimension).
        // Verifica che l'output concatenato abbia forma [4, 6] (dimensione embedding originale).
        assert_eq!(output.shape(), &[4, embedding_dim], "Final concatenated output should have shape [4, {}]", embedding_dim);
    }

    /// Test for positional encoding properties.
    /// Test per le proprietà dell'encoding posizionale.
    #[test]
    fn test_positional_encoding_properties() {
        // Set parameters: sequence length 10, model dimension 8.
        // Impostiamo: lunghezza sequenza 10, dimensione modello 8.
        let seq_length = 10;
        let d_model = 8;

        // Generate the positional encoding matrix.
        // Genera la matrice di encoding posizionale.
        let pos_encoding = NabAttention::generate_positional_encoding(seq_length, d_model);

        // Verify the shape is [10, 8].
        // Verifica che la forma sia [10, 8].
        assert_eq!(pos_encoding.shape(), &[seq_length, d_model], "Positional encoding shape should be [{} , {}]", seq_length, d_model);

        // Check periodicity: sine and cosine values should be between -1 and 1.
        // Verifica che i valori di sin e cos siano compresi tra -1 e 1.
        for pos in 0..seq_length {
            for i in 0..(d_model / 2) {
                let sin_val = pos_encoding.get_2d(pos, 2 * i);
                let cos_val = pos_encoding.get_2d(pos, 2 * i + 1);
                assert!(sin_val >= -1.0 && sin_val <= 1.0, "Sine value at position {} index {} should be between -1 and 1", pos, 2 * i);
                assert!(cos_val >= -1.0 && cos_val <= 1.0, "Cosine value at position {} index {} should be between -1 and 1", pos, 2 * i + 1);
            }
        }
    }

    /// Test for attention with positional encoding.
    /// Test per l'attention con encoding posizionale.
    #[test]
    fn test_attention_with_positional_encoding() {
        // Set parameters: embedding dimension 6, 2 heads (head_dim = 3), sequence length 4.
        // Impostiamo: dimensione embedding 6, 2 teste (head_dim = 3), lunghezza sequenza 4.
        let embedding_dim = 6;
        let num_heads = 2;
        let _head_dim = embedding_dim / num_heads; // head_dim = 3
        let seq_length = 4;

        // Generate sample embeddings with shape [4, 6].
        // Genera embeddings di esempio con forma [4, 6].
        let embeddings = NDArray::rand_uniform(&[seq_length, embedding_dim]).multiply_scalar(0.1);

        // Add positional encoding to the embeddings.
        // Aggiunge l'encoding posizionale agli embeddings.
        let encoded_embeddings = NabAttention::add_positional_encoding(&embeddings);

        // Verify that encoded embeddings have the same shape as the original embeddings.
        // Verifica che gli embeddings con encoding abbiano la stessa forma degli originali.
        assert_eq!(encoded_embeddings.shape(), embeddings.shape(), "Encoded embeddings should have the same shape as the original embeddings");

        // Create weight matrices with correct dimensions [6, 6].
        // Crea matrici di pesi con forma corretta [6, 6].
        let query_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let key_weights   = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let value_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);

        // Create the NabAttention module.
        // Crea il modulo NabAttention.
        let attention = NabAttention::new(
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            embedding_dim
        );

        // Execute forward pass for both single-head and multi-head attention using the encoded embeddings.
        // Esegui il forward pass per single-head e multi-head usando gli embeddings con encoding.
        let (single_probs, single_output) = attention.forward(&encoded_embeddings);
        let (multi_probs, multi_output) = attention.forward_multihead(&encoded_embeddings);

        // Verify output shapes:
        // For single-head, output should have shape [seq_length, head_dim] = [4, 3].
        // For multi-head, concatenated output should have shape [seq_length, embedding_dim] = [4, 6].
        // Verifica le forme degli output.
        assert_eq!(single_output.shape()[0], seq_length, "Single-head output should have {} rows", seq_length);
        assert_eq!(multi_output.shape()[0], seq_length, "Multi-head output should have {} rows", seq_length);
        assert_eq!(multi_output.shape()[1], embedding_dim, "Multi-head concatenated output should have {} columns", embedding_dim);

        // Verify that each row in the single-head attention probabilities sums approximately to 1 (tolerance 1e-2).
        // Verifica che ogni riga delle probabilità in single-head sommi circa 1 (tolleranza 1e-2).
        let single_row_sums = single_probs.sum_axis(1);
        for (i, &sum) in single_row_sums.data().iter().enumerate() {
            assert!((sum - 1.0).abs() < 1e-2, "Single-head row {} sum {} should be approximately 1", i, sum);
        }

        // For multi-head attention, verify each head's row sums are approximately 1 (tolerance 1e-2) and probabilities are valid.
        // Per l'attention multi-testa, verifica che ogni riga per ogni testa sommi circa 1 (tolleranza 1e-2) e che le probabilità siano valide.
        for (i, head_probs) in multi_probs.iter().enumerate() {
            let row_sums = head_probs.sum_axis(1);
            for (j, &sum) in row_sums.data().iter().enumerate() {
                assert!((sum - 1.0).abs() < 1e-2, "Multi-head {} row {} sum {} should be approximately 1", i, j, sum);
            }
            for &prob in head_probs.data() {
                assert!(prob >= 0.0, "Head {} attention probabilities should be non-negative", i);
                assert!(prob <= 1.0, "Head {} attention probabilities should be <= 1", i);
            }
        }
    }

    /// Test full pipeline using NabTokenizer and NabAttention.
    /// 
    /// This test tokenizes the sentence "The bank of the river was flooded!" using NabTokenizer,
    /// ensuring that special tokens [CLS] and [SEP] are added. Then it obtains token embeddings,
    /// applies positional encoding, and passes the embeddings through a NabAttention module.
    /// The test prints out intermediate values including the token embeddings, positional encodings,
    /// projected Q, K, V matrices (via debug prints in attention_head), final attention probabilities,
    /// and the concatenated output. //
    ///
    /// Test completo che utilizza NabTokenizer e NabAttention.
    #[test]
    fn test_full_pipeline_with_tokenizer() {
        use crate::nab_tokenizer::NabTokenizer;
        
        // Define the sentence and embedding dimension (using a small dimension for clarity, e.g., 8).
        let sentence = "The bank of the river was flooded!";
        let embedding_dim = 8;
        
        // Create a NabTokenizer with embedding dimension 8.
        let mut tokenizer = NabTokenizer::new(embedding_dim);
        
        // Tokenize the sentence with special tokens.
        let tokens = tokenizer.encode(sentence, true);
        println!("Token IDs: {:?}", tokens.data());
        
        // Decode back to check that [CLS] and [SEP] are present.
        let decoded = tokenizer.decode(&tokens, false);
        println!("Decoded text (with special tokens): {}", decoded);
        
        // Get token embeddings.
        let embeddings = tokenizer.encode_with_embeddings(sentence, true);
        println!("Initial Embeddings (shape {:?}): {:?}", embeddings.shape(), embeddings.data());
        
        // Add positional encoding to the embeddings.
        let pos_embeddings = NabAttention::add_positional_encoding(&embeddings);
        println!("Positional Encoded Embeddings (shape {:?}): {:?}", pos_embeddings.shape(), pos_embeddings.data());
        
        // Create a NabAttention module. We'll use 2 heads.
        let num_heads = 2;
        // Initialize weight matrices for query, key, value of shape [embedding_dim, embedding_dim].
        let query_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let key_weights   = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let value_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        
        let attention = NabAttention::new(query_weights, key_weights, value_weights, num_heads, embedding_dim);
        
        // Run the forward pass for single head and multi-head attention.
        println!("\n--- Single-head Attention ---");
        let (single_probs, single_output) = attention.forward(&pos_embeddings);
        println!("Single-head Attention Probabilities (shape {:?}): {:?}", single_probs.shape(), single_probs.data());
        println!("Single-head Output (shape {:?}): {:?}", single_output.shape(), single_output.data());
        
        println!("\n--- Multi-head Attention ---");
        let (multi_probs, multi_output) = attention.forward_multihead(&pos_embeddings);
        println!("Multi-head Attention Probabilities for each head:");
        for (i, head_probs) in multi_probs.iter().enumerate() {
            println!("  Head {}: shape {:?} -> {:?}", i, head_probs.shape(), head_probs.data());
        }
        println!("Multi-head Concatenated Output (shape {:?}): {:?}", multi_output.shape(), multi_output.data());
    }

    /// Test the creation and forward pass of the TransformerBlock.
    #[test]
    fn test_transformer_block_forward() {
        let embedding_dim = 8;
        let num_heads = 2;
        let seq_length = 4;

        // Generate sample embeddings with shape [4, 8].
        let embeddings = NDArray::rand_uniform(&[seq_length, embedding_dim]).multiply_scalar(0.1);

        // Create weight matrices with correct dimensions [8, 8].
        let query_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let key_weights   = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);
        let value_weights = NDArray::rand_uniform(&[embedding_dim, embedding_dim]).multiply_scalar(0.1);

        // Create the NabAttention module.
        let attention = NabAttention::new(
            query_weights,
            key_weights,
            value_weights,
            num_heads,
            embedding_dim
        );

        // Create the TransformerBlock.
        let transformer_block = TransformerBlock::new(attention);

        // Execute the forward pass.
        let (attention_probs, output) = transformer_block.forward(&embeddings);

        // Verify that the number of attention probability matrices equals the number of heads.
        assert_eq!(attention_probs.len(), num_heads, "Should have {} attention probability matrices", num_heads);

        // Verify that the concatenated output has shape [4, 8] (original embedding dimension).
        assert_eq!(output.shape(), &[seq_length, embedding_dim], "Final concatenated output should have shape [4, {}]", embedding_dim);
    }
}




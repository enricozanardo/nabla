use crate::nab_array::NDArray;
use crate::nab_sa::NabAttention;
use rand_distr::{Normal, Distribution};
use rand::thread_rng;

/// EmbeddingLayer maps token IDs to dense vectors and adds positional encoding.
///
/// # Fields
/// * `embedding_matrix` - NDArray of shape [vocab_size, embedding_dim] representing the embedding weights.
/// * `vocab_size` - The size of the vocabulary.
/// * `embedding_dim` - The dimension of each embedding vector.
pub struct EmbeddingLayer {
    pub embedding_matrix: NDArray, // shape: [vocab_size, embedding_dim]
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl EmbeddingLayer {
    /// Creates a new EmbeddingLayer with randomly initialized embeddings.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary.
    /// * `embedding_dim` - Dimension of embeddings.
    ///
    /// # Returns
    /// A new EmbeddingLayer instance.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let total = vocab_size * embedding_dim;
        let data: Vec<f64> = (0..total).map(|_| normal.sample(&mut rng)).collect();
        // The embedding matrix is stored in row-major order with shape [vocab_size, embedding_dim]
        let embedding_matrix = NDArray::new(data, vec![vocab_size, embedding_dim]);
        Self { embedding_matrix, vocab_size, embedding_dim }
    }

    /// Performs a forward pass: maps token IDs to embeddings and adds positional encoding.
    ///
    /// # Arguments
    /// * `token_ids` - NDArray of shape [seq_len] with token IDs as f64 (should be integer values).
    ///
    /// # Returns
    /// An NDArray of shape [seq_len, embedding_dim] representing the token embeddings with positional encoding added.
    pub fn forward(&self, token_ids: &NDArray) -> NDArray {
        let seq_len = token_ids.shape()[0];
        let mut output_data = Vec::with_capacity(seq_len * self.embedding_dim);
        // For each token id, retrieve the corresponding row from the embedding matrix.
        for &id_val in token_ids.data().iter() {
            let id = id_val as usize;
            // Ensure the token id is within the range of the vocabulary.
            assert!(id < self.vocab_size, "Token id {} out of range", id);
            // Each row's data is at position [id * embedding_dim, (id+1) * embedding_dim)
            let start = id * self.embedding_dim;
            let end = start + self.embedding_dim;
            let row = &self.embedding_matrix.data()[start..end];
            output_data.extend_from_slice(row);
        }
        let token_embeddings = NDArray::new(output_data, vec![seq_len, self.embedding_dim]);

        // Generate positional encoding for the given sequence length and embedding dimension using NabAttention.
        let pos_encoding = NabAttention::generate_positional_encoding(seq_len, self.embedding_dim);
        // Add token embeddings with positional encoding element-wise.
        token_embeddings + &pos_encoding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;

    /// Test the creation of the EmbeddingLayer and check dimensions.
    #[test]
    fn test_embedding_layer_creation() {
        let vocab_size = 100;
        let embedding_dim = 16;
        let layer = EmbeddingLayer::new(vocab_size, embedding_dim);
        assert_eq!(layer.embedding_matrix.shape(), &[vocab_size, embedding_dim]);
    }

    /// Test the forward pass of the EmbeddingLayer by verifying shape and approximate sum of first row.
    #[test]
    fn test_embedding_layer_forward() {
        let vocab_size = 50;
        let embedding_dim = 8;
        let layer = EmbeddingLayer::new(vocab_size, embedding_dim);
        // Create a dummy token_ids NDArray of shape [seq_len]
        let token_ids = NDArray::from_vec(vec![2.0, 10.0, 3.0]); // Sequence length 3
        let output = layer.forward(&token_ids);
        // Check that the output shape is [3, embedding_dim]
        assert_eq!(output.shape(), &[3, embedding_dim]);

        // Manually compute expected sum for the first token row (token id 2) plus its positional encoding.
        let start = 2 * embedding_dim;
        let end = start + embedding_dim;
        let token_sum: f64 = layer.embedding_matrix.data()[start..end].iter().sum();
        let pos_encoding = NabAttention::generate_positional_encoding(3, embedding_dim);
        let pos_sum: f64 = pos_encoding.data()[0..embedding_dim].iter().sum();
        let expected_sum = token_sum + pos_sum;

        let output_first_sum: f64 = output.data()[0..embedding_dim].iter().sum();
        assert!((output_first_sum - expected_sum).abs() < 1e-6, "First row sum mismatch");
    }
} 
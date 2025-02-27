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
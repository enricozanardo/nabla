/*
Module: nab_transformer.rs
This module implements a basic Transformer Block.
Il modulo implementa un blocco Transformer di base.

A TransformerBlock is composed of a multi-head self-attention layer and a feed-forward network,
wrapped with residual connections and layer normalization.
Un TransformerBlock è composto da un layer di self-attention multi-head e una rete feed-forward,
avvolti da connessioni residuali e normalizzazione del layer.
*/

use crate::nab_array::NDArray;
use crate::nab_sa::NabAttention;
use crate::nab_model::FeedForwardNetwork;
use crate::nab_model::layer_normalize;

/// TransformerBlock struct
///
/// English: Represents a transformer block which contains a self-attention layer and a feed-forward network.
/// Italian: Rappresenta un blocco Transformer che contiene un layer di self-attention e una rete feed-forward.
#[derive(Clone)]
pub struct TransformerBlock {
    /// Self-attention module
    /// Modulo di self-attention
    pub attention: NabAttention,
    /// Feed-forward network
    /// Rete feed-forward
    pub ffn: FeedForwardNetwork,
}

impl TransformerBlock {
    /// Creates a new TransformerBlock.
    ///
    /// English: Initializes a new TransformerBlock with the provided self-attention and feed-forward modules.
    /// Italian: Inizializza un nuovo TransformerBlock con i moduli di self-attention e feed-forward forniti.
    ///
    /// # Arguments
    /// - `attention`: NabAttention module for self-attention.
    /// - `ffn`: FeedForwardNetwork module for the feed-forward sublayer.
    ///
    /// # Returns
    /// A new TransformerBlock.
    pub fn new(attention: NabAttention, ffn: FeedForwardNetwork) -> Self {
        TransformerBlock { attention, ffn }
    }

    /// Forward pass of the TransformerBlock.
    ///
    /// English: Applies self-attention, adds a residual connection and layer normalization, then applies the feed-forward network,
    /// adds another residual connection and layer normalization.
    /// Italian: Applica il self-attention, aggiunge una connessione residuale e la normalizzazione del layer, quindi applica la rete feed-forward,
    /// aggiungendo un'altra connessione residuale e la normalizzazione del layer.
    ///
    /// # Arguments
    /// - `x`: Input NDArray with shape [batch_size, sequence_length, d_model] (for simplicity, handled as 2D [batch_size, d_model]).
    ///
    /// # Returns
    /// A normalized NDArray of the same shape as input.
    pub fn forward(&self, x: &NDArray) -> NDArray {
        // Self-Attention sublayer: compute attention output
        // Sottolayer di self-attention: calcola l'output dell'attenzione
        let (_, attn_out) = self.attention.forward(x);
        
        // First residual connection: add input and attention output
        // Prima connessione residuale: somma dell'input e dell'output di attenzione
        let x_attn = x.clone().add(&attn_out);
        // Apply layer normalization
        // Applica la normalizzazione del layer
        let x_norm = layer_normalize(&x_attn);
        
        // Feed-Forward Network sublayer: process the normalized output
        // Sottolayer della rete feed-forward: processa l'output normalizzato
        let ffn_out = self.ffn.forward(&x_norm);
        
        // Second residual connection: add the normalized input and feed-forward output
        // Seconda connessione residuale: somma dell'input normalizzato e dell'output della rete feed-forward
        let x_ffn = x_norm.clone().add(&ffn_out);
        // Apply layer normalization
        // Applica la normalizzazione del layer
        let out = layer_normalize(&x_ffn);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;
    use crate::nab_sa::NabAttention;
    use crate::nab_model::FeedForwardNetwork;

    #[test]
    fn test_transformer_block_new() {
        // Create a dummy attention module and a feed-forward network.
        // Crea un modulo di attenzione dummy e una rete feed-forward.
        let attention = NabAttention::dummy();
        let ffn = FeedForwardNetwork::new(4, 4, 4);
        let block = TransformerBlock::new(attention, ffn);
        // Check that the block was created with the correct dimensions
        // Verifica che il blocco sia stato creato con le dimensioni corrette
        assert_eq!(block.attention.d_model, 4);
        assert_eq!(block.ffn.hidden_dim, 4);
    }

    #[test]
    fn test_transformer_block_forward() {
        // Create a transformer block with dummy attention and a basic feed-forward network.
        // Crea un blocco Transformer con attenzione dummy e una rete feed-forward di base.
        let attention = NabAttention::dummy();
        let ffn = FeedForwardNetwork::new(4, 4, 4);
        let block = TransformerBlock::new(attention, ffn);
        
        // Create a dummy input NDArray of shape [batch_size, d_model]. For simplicity, we use 2D.
        // Crea un NDArray di input dummy di forma [batch_size, d_model]. Per semplicità, usiamo 2D.
        let input = NDArray::rand_uniform(&[2, 4]).multiply_scalar(0.1);
        
        // Perform the forward pass
        // Esegui il forward pass
        let output = block.forward(&input);
        
        // Check that the output shape matches the input shape
        // Verifica che la forma dell'output corrisponda a quella dell'input
        assert_eq!(output.shape(), input.shape());
    }
} 
use crate::nab_array::NDArray;
use crate::nab_distil_transformer_block::DistilTransformerBlock;

/// StackedTransformerLayer stacks multiple DistilTransformerBlock layers to form a small transformer encoder
/// with residual connections across layers to help with gradient flow.
#[derive(Clone)]
pub struct StackedTransformerLayer {
    pub blocks: Vec<DistilTransformerBlock>,
}

impl StackedTransformerLayer {
    /// Creates a new StackedTransformerLayer with the given transformer blocks.
    ///
    /// # Arguments
    /// * `blocks` - A vector of DistilTransformerBlock instances.
    ///
    /// # Returns
    /// A new instance of StackedTransformerLayer.
    pub fn new(blocks: Vec<DistilTransformerBlock>) -> Self {
        StackedTransformerLayer { blocks }
    }

    /// Performs a forward pass through all stacked transformer blocks in sequence with residual connections.
    /// For each block, it updates the representation using a residual connection: x = x + block.forward(x).
    ///
    /// # Arguments
    /// * `x` - An input NDArray, usually of shape [batch_size, d_model].
    ///
    /// # Returns
    /// The final output NDArray after all blocks have been applied, with residual connections.
    pub fn forward(&self, x: &NDArray) -> NDArray {
        let mut output = x.clone();
        for block in &self.blocks {
            // Compute the block's output
            let block_output = block.forward(&output);
            // Add residual connection: output = output + block_output
            output = output.clone().add(&block_output);
        }
        output
    }
}

// Dedicated unit tests for StackedTransformerLayer with residual connections
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;
    use crate::nab_sa::NabAttention;
    use crate::nab_model::FeedForwardNetwork;
    use crate::nab_distil_transformer_block::DistilTransformerBlock;

    // Helper function to create a dummy DistilTransformerBlock using identity weights for FeedForwardNetwork
    fn dummy_distil_block(d: usize) -> DistilTransformerBlock {
        // Create identity data for a square matrix of dimension d
        let identity_data: Vec<f64> = (0..(d * d))
            .map(|i| if i % (d + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let zero_data: Vec<f64> = vec![0.0; d];
        let w_ffn1 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn1 = NDArray::new(zero_data.clone(), vec![1, d]);
        let w_ffn2 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn2 = NDArray::new(zero_data.clone(), vec![1, d]);
        let ffn = FeedForwardNetwork { w1: w_ffn1, b1: b_ffn1, w2: w_ffn2, b2: b_ffn2, hidden_dim: d, output_dim: d };

        // Create dummy attention module using its dummy() method
        let attention = NabAttention::dummy(d);
        DistilTransformerBlock { attention, ffn }
    }

    #[test]
    fn test_stacked_transformer_layer_new() {
        let d = 4;
        let block1 = dummy_distil_block(d);
        let block2 = dummy_distil_block(d);
        let blocks = vec![block1, block2];
        let layer = StackedTransformerLayer::new(blocks);
        // Verify that the layer contains the correct number of blocks
        assert_eq!(layer.blocks.len(), 2);
    }

    #[test]
    fn test_stacked_transformer_layer_forward_shape() {
        let d = 4;
        let block1 = dummy_distil_block(d);
        let block2 = dummy_distil_block(d);
        let layer = StackedTransformerLayer::new(vec![block1, block2]);
        let input = NDArray::rand_uniform(&[2, d]).multiply_scalar(0.1);
        let output = layer.forward(&input);
        // The output shape should match the input shape since addition is element-wise
        assert_eq!(output.shape(), input.shape(), "Output shape should match input shape");
    }

    #[test]
    fn test_stacked_transformer_layer_forward_values() {
        let d = 4;
        let block1 = dummy_distil_block(d);
        let block2 = dummy_distil_block(d);
        let layer = StackedTransformerLayer::new(vec![block1, block2]);
        let input = NDArray::from_vec(vec![1.0, -1.0, 2.0, -2.0]).reshape(&[1, d]).unwrap();
        let output = layer.forward(&input);
        // Check that all output values are finite
        for &v in output.data().iter() {
            assert!(v.is_finite(), "Output value should be finite");
        }
    }
} 
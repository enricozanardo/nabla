use crate::nab_array::NDArray;
use crate::nab_sa::NabAttention;
use crate::nab_model::FeedForwardNetwork;
use crate::nab_model::layer_normalize;

//// DistilTransformerBlock implementation following DistilBERT pre-layer norm design

// DistilTransformerBlock applies pre-layer normalization before attention and MLP layers.
#[derive(Clone)]
pub struct DistilTransformerBlock {
    pub attention: NabAttention,
    pub ffn: FeedForwardNetwork,
}

impl DistilTransformerBlock {
    /// Performs a forward pass through the DistilBERT-style transformer block.
    /// Steps:
    /// 1. Pre-attention layer normalization
    /// 2. Self-attention and add residual connection
    /// 3. Pre-MLP layer normalization
    /// 4. MLP forward and add residual connection
    pub fn forward(&self, x: &NDArray) -> NDArray {
        let norm1 = layer_normalize(x);
        let (_, attn_out) = self.attention.forward(&norm1);
        let x_attn = x.clone().add(&attn_out);
        let norm2 = layer_normalize(&x_attn);
        let mlp_out = self.ffn.forward(&norm2);
        let output = x_attn.clone().add(&mlp_out);
        output
    }
}

//// NEW: Dedicatd unit tests for DistilTransformerBlock
#[allow(unused_imports)]
#[cfg(test)]
mod distil_transformer_tests {
    use super::*;
    use crate::nab_array::NDArray;
    use crate::nab_sa::NabAttention;
    use crate::nab_model::FeedForwardNetwork;
    use crate::nab_model::layer_normalize;

    #[test]
    fn test_distil_transformer_block_forward_shape() {
        let d = 4;
        let identity_data: Vec<f64> = (0..(d*d))
            .map(|i| if i % (d+1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let zero_data: Vec<f64> = vec![0.0; d];
        let w_ffn1 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn1 = NDArray::new(zero_data.clone(), vec![1, d]);
        let w_ffn2 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn2 = NDArray::new(zero_data.clone(), vec![1, d]);
        let ffn = FeedForwardNetwork { w1: w_ffn1, b1: b_ffn1, w2: w_ffn2, b2: b_ffn2, hidden_dim: d, output_dim: d };

        let attention = NabAttention::dummy();
        let block = DistilTransformerBlock { attention, ffn };
        let input = NDArray::rand_uniform(&[2, d]).multiply_scalar(0.1);
        let output = block.forward(&input);
        assert_eq!(output.shape(), input.shape(), "Output shape should match input shape");
    }

    #[test]
    fn test_distil_transformer_block_forward_values() {
        let d = 4;
        let identity_data: Vec<f64> = (0..(d*d))
            .map(|i| if i % (d+1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let zero_data: Vec<f64> = vec![0.0; d];
        let w_ffn1 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn1 = NDArray::new(zero_data.clone(), vec![1, d]);
        let w_ffn2 = NDArray::new(identity_data.clone(), vec![d, d]);
        let b_ffn2 = NDArray::new(zero_data.clone(), vec![1, d]);
        let ffn = FeedForwardNetwork { w1: w_ffn1, b1: b_ffn1, w2: w_ffn2, b2: b_ffn2, hidden_dim: d, output_dim: d };

        let attention = NabAttention::dummy();
        let block = DistilTransformerBlock { attention, ffn };
        let input = NDArray::from_vec(vec![1.0, -1.0, 2.0, -2.0]).reshape(&[1, d]).unwrap();
        let output = block.forward(&input);
        for &v in output.data().iter() {
            assert!(v.is_finite(), "Output value should be finite");
        }
    }
} 
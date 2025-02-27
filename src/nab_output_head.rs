use crate::nab_array::NDArray;

/// OutputHead represents the output layer for a language model, mapping hidden states to token probabilities using a linear transformation followed by a softmax.
///
/// Italian: OutputHead rappresenta il layer di output per un modello linguistico, mappando lo stato nascosto alle probabilità dei token usando una trasformazione lineare seguita da softmax.
#[derive(Clone)]
pub struct OutputHead {
    /// Weight matrix of shape [input_dim, vocab_size]
    pub weight: NDArray,
    /// Bias vector of shape [1, vocab_size]
    pub bias: NDArray,
}

impl OutputHead {
    /// Creates a new OutputHead with random weights and zero biases.
    ///
    /// # Arguments
    /// * `input_dim` - The dimension of the input features (hidden dimension).
    /// * `vocab_size` - The size of the vocabulary.
    ///
    /// # Returns
    /// A new instance of OutputHead.
    ///
    /// Italian: Crea un nuovo OutputHead con pesi casuali e bias zero.
    pub fn new(input_dim: usize, vocab_size: usize) -> Self {
        // Initialize weight with uniform random values in range [-0.1, 0.1]
        let total_elements = input_dim * vocab_size;
        let weight_data: Vec<f64> = (0..total_elements)
            .map(|_| (rand::random::<f64>() - 0.5) * 0.2)  // random number in [-0.1, 0.1]
            .collect();
        let weight = NDArray::new(weight_data, vec![input_dim, vocab_size]);

        // Initialize bias as zeros with shape [1, vocab_size]
        let bias = NDArray::zeros(vec![1, vocab_size]);

        OutputHead { weight, bias }
    }

    /// Performs a forward pass through the output head.
    /// It applies a linear transformation to the input and then computes the row-wise softmax to obtain token probabilities.
    ///
    /// # Arguments
    /// * `x` - An input NDArray of shape [batch_size, input_dim].
    ///
    /// # Returns
    /// An NDArray of shape [batch_size, vocab_size] containing token probabilities.
    ///
    /// Italian: Esegue il forward pass attraverso l'output head, applicando una trasformazione lineare seguita da softmax row-wise per ottenere le probabilità dei token.
    pub fn forward(&self, x: &NDArray) -> NDArray {
        // Compute linear transformation: logits = x.dot(weight) + broadcasted_bias
        let dot_product = x.dot(&self.weight);
        let batch_size = x.shape()[0];
        let bias_broadcasted = broadcast_bias(&self.bias, batch_size);
        let logits = dot_product.add(&bias_broadcasted);
        // Apply row-wise softmax
        row_softmax(&logits)
    }
}

/// Broadcasts a bias NDArray (of shape [1, vocab_size]) along the batch dimension to shape [batch_size, vocab_size].
///
/// Italian: Effettua il broadcast di un NDArray bias (di forma [1, vocab_size]) lungo la dimensione del batch, ottenendo una forma [batch_size, vocab_size].
fn broadcast_bias(bias: &NDArray, batch_size: usize) -> NDArray {
    // Expect bias to have shape [1, vocab_size]
    let shape = bias.shape();
    assert!(shape.len() == 2, "Bias must be a 2D array");
    assert!(shape[0] == 1, "Bias must have shape [1, vocab_size]");
    let vocab_size = shape[1];
    let bias_data = bias.data();
    // Repeat the single row for each batch element
    let mut new_data = Vec::with_capacity(batch_size * vocab_size);
    for _ in 0..batch_size {
        new_data.extend_from_slice(bias_data);
    }
    NDArray::new(new_data, vec![batch_size, vocab_size])
}

/// Applies row-wise softmax on the input NDArray. Assumes the input is 2D with shape [batch_size, vocab_size].
///
/// For each row, it subtracts the maximum value for numerical stability, applies the exponential function,
/// computes the sum, and normalizes each element.
///
/// Italian: Applica la softmax per riga all'NDArray di input. Assume che l'input sia 2D con forma [batch_size, vocab_size].
fn row_softmax(logits: &NDArray) -> NDArray {
    // Retrieve shape information: batch_size and vocab_size
    let shape = logits.shape();
    assert!(shape.len() == 2, "logits must be a 2D array");
    let batch_size = shape[0];
    let vocab_size = shape[1];

    let mut result = Vec::with_capacity(batch_size * vocab_size);

    // Iterate over each row
    for i in 0..batch_size {
        // Extract the row
        let start = i * vocab_size;
        let end = start + vocab_size;
        let row = &logits.data()[start..end];
        
        // Compute max value for numerical stability
        let row_max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Compute exponentials and sum
        let exp_row: Vec<f64> = row.iter().map(|&x| (x - row_max).exp()).collect();
        let sum_exp: f64 = exp_row.iter().sum();
        
        // Normalize each element
        for &val in exp_row.iter() {
            result.push(val / sum_exp);
        }
    }

    NDArray::new(result, vec![batch_size, vocab_size])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nab_array::NDArray;

    #[test]
    fn test_output_head_new() {
        let input_dim = 8;
        let vocab_size = 100;
        let output_head = OutputHead::new(input_dim, vocab_size);
        // Check weight shape
        assert_eq!(output_head.weight.shape(), &[input_dim, vocab_size]);
        // Check bias shape
        assert_eq!(output_head.bias.shape(), &[1, vocab_size]);
    }

    #[test]
    fn test_broadcast_bias() {
        let vocab_size = 10;
        let bias = NDArray::zeros(vec![1, vocab_size]);
        let batch_size = 3;
        let broadcasted = broadcast_bias(&bias, batch_size);
        assert_eq!(broadcasted.shape(), &[batch_size, vocab_size]);
        // Check that every element is 0
        for &v in broadcasted.data().iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_row_softmax_normalization() {
        // Create a simple 2D NDArray with known values
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // 2 rows, 3 columns
        let logits = NDArray::new(data, vec![2, 3]);
        let probabilities = row_softmax(&logits);
        // For each row, sum of probabilities should be approximately 1
        let shape = probabilities.shape();
        let batch_size = shape[0];
        let vocab_size = shape[1];
        for i in 0..batch_size {
            let start = i * vocab_size;
            let end = start + vocab_size;
            let row = &probabilities.data()[start..end];
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row sum should be 1, but got {}", sum);
        }
    }

    #[test]
    fn test_output_head_forward_shape() {
        let input_dim = 8;
        let vocab_size = 100;
        let output_head = OutputHead::new(input_dim, vocab_size);
        // Create a dummy input NDArray of shape [batch_size, input_dim]
        let batch_size = 4;
        let input = NDArray::rand_uniform(&[batch_size, input_dim]);
        let probs = output_head.forward(&input);
        // Check that the output shape is [batch_size, vocab_size]
        assert_eq!(probs.shape(), &[batch_size, vocab_size]);
    }

    #[test]
    fn test_output_head_forward_probabilities() {
        let input_dim = 8;
        let vocab_size = 10; // small vocab for testing
        let output_head = OutputHead::new(input_dim, vocab_size);
        let batch_size = 2;
        let input = NDArray::rand_uniform(&[batch_size, input_dim]);
        let probs = output_head.forward(&input);
        // Check that each probability is between 0 and 1 and rows sum to 1
        let shape = probs.shape();
        let bs = shape[0];
        let vs = shape[1];
        for i in 0..bs {
            let start = i * vs;
            let end = start + vs;
            let row = &probs.data()[start..end];
            let sum: f64 = row.iter().sum();
            for &p in row.iter() {
                assert!(p >= 0.0 && p <= 1.0, "Probability {} is not in [0, 1]", p);
            }
            assert!((sum - 1.0).abs() < 1e-6, "Row sum should be 1, but got {}", sum);
        }
    }
} 
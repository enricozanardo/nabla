use crate::nab_array::NDArray;
use crate::nab_tokenizer::NabTokenizer;

pub struct NabAttention {
    pub query: NDArray,  // Matrix of shape [n_queries, d_model]
    pub key: NDArray,    // Matrix of shape [n_keys, d_model] 
    pub value: NDArray,  // Matrix of shape [n_keys, d_value]
}

impl NabAttention {
    pub fn new(query: NDArray, key: NDArray, value: NDArray) -> Self {
        // Verify inputs are 2D matrices
        assert_eq!(query.ndim(), 2, "Query must be a 2D matrix");
        assert_eq!(key.ndim(), 2, "Key must be a 2D matrix"); 
        assert_eq!(value.ndim(), 2, "Value must be a 2D matrix");

        // Verify compatible dimensions
        assert_eq!(key.shape()[1], query.shape()[1], "Key and query must have same feature dimension");
        assert_eq!(key.shape()[0], value.shape()[0], "Key and value must have same sequence length");

        Self { query, key, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nab_attention_creation() {
        // Create test matrices with compatible dimensions
        let query = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]); // 3 queries, d_model = 3
        let key = NDArray::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);   // 2 keys, d_model = 2
        let value = NDArray::new(vec![9.0, 10.0, 11.0, 12.0], vec![2, 2]); // 2 values, d_value = 2

        // Test successful creation
        let attention = NabAttention::new(query.clone(), key.clone(), value.clone());
        assert_eq!(attention.query.shape(), &[2, 2]);
        assert_eq!(attention.key.shape(), &[2, 2]);
        assert_eq!(attention.value.shape(), &[2, 2]);

        // Test incompatible dimensions
        let wrong_key = NDArray::new(vec![1.0, 2.0, 3.0], vec![3, 1]); // Wrong shape
        let result = std::panic::catch_unwind(|| {
            NabAttention::new(query.clone(), wrong_key, value.clone())
        });
        assert!(result.is_err());

        // Test non-2D input
        let wrong_dim = NDArray::new(vec![1.0, 2.0, 3.0], vec![3]); // 1D array
        let result = std::panic::catch_unwind(|| {
            NabAttention::new(query.clone(), key.clone(), wrong_dim)
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_nab_attention_calculation() {
        let embedding_dim = 3;

        
        let test_string = "[CLS] The bank of the river was flooded [SEP]";
        let mut tokenizer = NabTokenizer::new(embedding_dim);  // Create tokenizer with embedding dim = 3
        
        // Get both tokens and embeddings
        let tokens = tokenizer.encode(test_string, true);
        let embeddings = tokenizer.encode_with_embeddings(test_string, true);

        println!("Original text: {}", test_string);
        println!("Decoded text: {}", tokenizer.decode(&tokens, false));
        println!("\nTokens and their embeddings:");
        
        for i in 0..tokens.shape()[0] {
            let token_id = tokens.data()[i];
            let token_text = tokenizer.decode(&NDArray::from_vec(vec![token_id]), false);
            let token_emb = embeddings.slice(i, i + 1);
            println!("Token {}: '{}' (ID: {}) -> {:?}", 
                    i, token_text.trim(), token_id, token_emb.data());
        }

        println!("\nEmbeddings shape: {:?}", embeddings.shape());
        println!("Number of tokens: {}", embeddings.shape()[0]);
        println!("Embedding dimension: {}", embeddings.shape()[1]);

        // Verify we have exactly 9 tokens:
        // [CLS] (1 token)
        // the, bank, of, the, river, was, flooded (7 tokens)
        // [SEP] (1 token)
        assert_eq!(embeddings.shape()[0], 9, "Should have exactly 9 tokens (2 special + 7 words)");
        assert_eq!(embeddings.shape()[1], embedding_dim, "Each token should have embedding dimension 3");

        // Create attention matrices with proper dimensions
        let query = embeddings.clone(); // Use embeddings as query
        let key = embeddings.clone();   // Use embeddings as key
        let value = embeddings.clone(); // Use embeddings as value

        // Test attention creation with embeddings
        let attention = NabAttention::new(query.clone(), key.clone(), value.clone());
        assert_eq!(attention.query.shape()[1], embedding_dim, "Query dimension should be 3");
        assert_eq!(attention.key.shape()[1], embedding_dim, "Key dimension should be 3");
        assert_eq!(attention.value.shape()[1], embedding_dim, "Value dimension should be 3");
    }
}




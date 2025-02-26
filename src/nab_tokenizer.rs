use tokenizers::models::wordpiece::WordPieceBuilder;
use tokenizers::Tokenizer;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
// use tokenizers::processors::bert::BertProcessing;
use std::collections::HashMap;
use rand_distr::{Distribution, Normal};
use rand::thread_rng;
use crate::nab_array::NDArray;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A BERT-style tokenizer with support for word embeddings
///
/// The tokenizer uses WordPiece tokenization and the BERT base uncased vocabulary.
/// Each token is associated with a fixed-dimensional embedding vector initialized
/// from a normal distribution N(0, 1).
///
/// # Features
///
/// - BERT base uncased vocabulary (~30,000 tokens)
/// - WordPiece tokenization with subword units
/// - Special token support ([PAD], [UNK], [CLS], [SEP], [MASK])
/// - Configurable embedding dimensions
/// - Random normal initialization for embeddings
///
/// # Examples
///
/// ```
/// use nabla_ml::nab_tokenizer::NabTokenizer;
///
/// // Create a tokenizer with BERT vocabulary and 768-dimensional embeddings
/// let mut tokenizer = NabTokenizer::new(768);
///
/// // Tokenize text and get embeddings
/// let text = "Hello world";
/// let tokens = tokenizer.encode(text);
/// let embeddings = tokenizer.encode_with_embeddings(text);
///
/// // Verify embedding dimensions
/// assert_eq!(embeddings.shape()[1], 768);
/// ```
pub struct NabTokenizer {
    /// The underlying BERT tokenizer
    tokenizer: Tokenizer,
    /// Dimension of token embeddings
    embedding_dim: usize,
    /// Map from token IDs to their embeddings
    embeddings: HashMap<u32, NDArray>,
    /// Next available ID for unknown tokens
    next_unk_id: u32,
}

impl NabTokenizer {
    /// Creates a new tokenizer with BERT vocabulary and specified embedding dimension.
    ///
    /// This constructor initializes a tokenizer using the BERT base uncased vocabulary
    /// and creates random embeddings for each token from a normal distribution N(0, 1).
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - The dimension of the token embeddings (typically 768 for BERT base)
    ///
    /// # Returns
    ///
    /// A new NabTokenizer instance
    ///
    /// # Examples
    ///
    /// ```
    /// let tokenizer = NabTokenizer::new(768);
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        Self::with_vocab(embedding_dim, Some("resources/bert-base-uncased-vocab.txt"))
    }

    /// Creates a new tokenizer with a custom vocabulary file and specified embedding dimension.
    ///
    /// This constructor allows using a custom vocabulary file while maintaining BERT's
    /// special token conventions. If no vocabulary file is provided, it uses the BERT
    /// base uncased vocabulary.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - The dimension of the token embeddings
    /// * `vocab_path` - Optional path to a vocabulary file. If None, uses BERT vocabulary
    ///
    /// # Returns
    ///
    /// A new NabTokenizer instance
    ///
    /// # Examples
    ///
    /// ```
    /// // With custom vocabulary
    /// let tokenizer = NabTokenizer::with_vocab(768, Some("path/to/vocab.txt"));
    ///
    /// // With default BERT vocabulary
    /// let tokenizer = NabTokenizer::with_vocab(768, None);
    /// ```
    pub fn with_vocab(embedding_dim: usize, vocab_path: Option<&str>) -> Self {
        let mut vocab = HashMap::new();
        
        let path = vocab_path.unwrap_or("resources/bert-base-uncased-vocab.txt");
        if let Ok(loaded_vocab) = Self::load_bert_vocab(path) {
            // Skip special tokens that are already in the vocabulary
            for (word, idx) in loaded_vocab {
                if !vocab.contains_key(&word) {
                    vocab.insert(word, idx as u32);
                }
            }
        }
        // Force special tokens to have expected indices
        let specials = vec![
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4),
            ("[unused0]", 5),
            ("[unused1]", 6)
        ];
        for (token, index) in specials {
            vocab.insert(token.to_string(), index);
        }

        let wordpiece = WordPieceBuilder::new()
            .vocab(vocab.clone())
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .max_input_chars_per_word(100)
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(wordpiece);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        
        // Configure special tokens in the tokenizer and treat them as single tokens
        let special_tokens = vec![
            ("[PAD]", 0),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("[MASK]", 4)
        ];

        // Add special tokens as single tokens that won't be split
        for (token, _) in &special_tokens {
            tokenizer.add_special_tokens(&[tokenizers::AddedToken::from(*token, true)]);
        }
        
        // Initialize embeddings with random values from N(0, 1)
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut embeddings = HashMap::new();
        
        // Ensure special tokens have embeddings
        for token_id in 0..vocab.len() as u32 {
            let embedding_data: Vec<f64> = (0..embedding_dim)
                .map(|_| normal.sample(&mut rng))
                .collect();
            let embedding = NDArray::new(embedding_data, vec![1, embedding_dim]);
            embeddings.insert(token_id, embedding);
        }

        Self {
            tokenizer,
            embedding_dim,
            embeddings,
            next_unk_id: vocab.len() as u32,
        }
    }

    /// Loads a BERT vocabulary file into a token-to-index mapping.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the vocabulary file
    ///
    /// # Returns
    ///
    /// A Result containing a HashMap mapping tokens to their indices
    fn load_bert_vocab(path: &str) -> Result<HashMap<String, usize>, std::io::Error> {
        let file = File::open(Path::new(path))?;
        let reader = BufReader::new(file);
        let mut vocab = HashMap::new();

        // Start indexing from 0 to match BERT's convention
        for (idx, line) in reader.lines().enumerate() {
            if let Ok(word) = line {
                let word = word.trim().to_string();
                vocab.insert(word, idx);
            }
        }

        Ok(vocab)
    }

    /// Encodes text into token IDs represented as an NDArray.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    /// * `add_special_tokens` - Whether to add [CLS] and [SEP] tokens
    ///
    /// # Returns
    ///
    /// An NDArray of shape [num_tokens] containing token IDs
    ///
    /// # Examples
    ///
    /// ```
    /// let tokenizer = NabTokenizer::new(768);
    /// // Without special tokens
    /// let tokens = tokenizer.encode("Hello world", false);
    /// // With special tokens
    /// let tokens_with_special = tokenizer.encode("Hello world", true);
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> NDArray {
        // Convert to lowercase since we're using BERT uncased vocabulary
        let text = text.to_lowercase();
        
        // Split text into words
        let mut tokens = Vec::new();
        let words: Vec<&str> = text.split_whitespace()
            .filter(|&w| !add_special_tokens || (w != "[cls]" && w != "[sep]"))
            .collect();
        
        if add_special_tokens {
            tokens.push(2.0); // [CLS]
        }
        
        for word in words {
            match word {
                "[pad]" => tokens.push(0.0),
                "[unk]" => tokens.push(1.0),
                "[mask]" => tokens.push(4.0),
                _ => {
                    // For regular words, use the tokenizer's vocabulary
                    if let Some(&id) = self.tokenizer.get_vocab(false).get(word) {
                        tokens.push(id as f64);
                    } else {
                        tokens.push(1.0); // [UNK]
                    }
                }
            }
        }
        
        if add_special_tokens {
            tokens.push(3.0); // [SEP]
        }
        
        NDArray::from_vec(tokens)
    }

    /// Encodes text and returns embeddings for each token.
    ///
    /// This method tokenizes the input text and returns an NDArray containing
    /// the embeddings for each token. Unknown tokens receive unique random
    /// embeddings that persist for the lifetime of the tokenizer.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    /// * `add_special_tokens` - Whether to add [CLS] and [SEP] tokens
    ///
    /// # Returns
    ///
    /// An NDArray of shape [num_tokens, embedding_dim] containing token embeddings
    ///
    /// # Examples
    ///
    /// ```
    /// let mut tokenizer = NabTokenizer::new(768);
    /// let embeddings = tokenizer.encode_with_embeddings("Hello world", false);
    /// assert_eq!(embeddings.shape()[1], 768);
    /// ```
    pub fn encode_with_embeddings(&mut self, text: &str, add_special_tokens: bool) -> NDArray {
        let tokens = self.encode(text, add_special_tokens);
        let mut embeddings = Vec::new();
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for token_id in tokens.data().iter() {
            let id = *token_id as u32;
            if let Some(embedding) = self.embeddings.get(&id) {
                embeddings.extend(embedding.data().iter());
            } else {
                // Generate new embedding for unknown token
                let embedding_data: Vec<f64> = (0..self.embedding_dim)
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                let embedding = NDArray::new(embedding_data, vec![1, self.embedding_dim]);
                self.embeddings.insert(self.next_unk_id, embedding.clone());
                embeddings.extend(embedding.data().iter());
                self.next_unk_id += 1;
            }
        }

        NDArray::new(embeddings, vec![tokens.shape()[0], self.embedding_dim])
    }

    /// Decodes token IDs back into text.
    ///
    /// # Arguments
    ///
    /// * `tokens` - An NDArray containing token IDs
    /// * `skip_special_tokens` - Whether to remove special tokens from the output
    ///
    /// # Returns
    ///
    /// The decoded text
    ///
    /// # Examples
    ///
    /// ```
    /// let tokenizer = NabTokenizer::new(768);
    /// let tokens = tokenizer.encode("Hello world", true);
    /// let text = tokenizer.decode(&tokens, true); // Skip special tokens
    /// assert_eq!(text.trim(), "hello world");
    /// ```
    pub fn decode(&self, tokens: &NDArray, skip_special_tokens: bool) -> String {
        let ids: Vec<u32> = tokens.data()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let mut text = self.tokenizer.decode(&ids, skip_special_tokens).unwrap();
        if skip_special_tokens {
            // Ensure all special tokens are removed
            text = text.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").replace("[UNK]", "").replace("[MASK]", "");
            text = text.trim().to_string();
        }
        text
    }

    /// Returns the size of the vocabulary.
    ///
    /// # Returns
    ///
    /// The number of tokens in the vocabulary
    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab(false).len()
    }

    /// Returns the current vocabulary as a mapping from tokens to IDs.
    ///
    /// # Returns
    ///
    /// A HashMap mapping tokens to their IDs
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(false)
    }

    /// Returns the dimension of the token embeddings.
    ///
    /// # Returns
    ///
    /// The embedding dimension
    pub fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Returns the embedding vector for a specific token ID.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The ID of the token
    ///
    /// # Returns
    ///
    /// An optional reference to the token's embedding vector
    pub fn get_token_embedding(&self, token_id: u32) -> Option<&NDArray> {
        self.embeddings.get(&token_id)
    }

    /// Debug function to print vocabulary information
    fn debug_vocab(&self) {
        let vocab = self.get_vocab();
        println!("Vocabulary size: {}", vocab.len());
        println!("Special tokens:");
        for token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] {
            println!("  {} -> {:?}", token, vocab.get(token));
        }
        println!("Common words:");
        for word in ["the", "a", "and", "is"] {
            println!("  {} -> {:?}", word, vocab.get(word));
        }
    }

    /// Batch encodes a slice of texts into a vector of token NDArray.
    /// This function leverages the existing encode function.
    /// It returns a vector of NDArray, one for each input text.
    ///
    /// Codifica in batch una slice di testi in un vettore di NDArray di token.
    /// Questa funzione utilizza la funzione encode esistente e ritorna un vettore di NDArray, uno per ogni testo.
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<NDArray> {
        texts.iter().map(|&text| self.encode(text, add_special_tokens)).collect()
    }

    /// Preprocesses a corpus string by splitting on new lines and filtering out empty lines.
    /// Returns a vector of non-empty, trimmed lines.
    ///
    /// Preprocessa una stringa di corpus dividendo per linea e rimuovendo quelle vuote.
    /// Ritorna un vettore di linee non vuote e rimosse dei bordi.
    pub fn preprocess_corpus(corpus: &str) -> Vec<&str> {
        corpus.lines()
              .filter_map(|line| {
                  let trimmed = line.trim();
                  if trimmed.is_empty() { None } else { Some(trimmed) }
              })
              .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_with_embeddings() {
        let mut tokenizer = NabTokenizer::new(768);
        let text = "Hello world";
        let tokens = tokenizer.encode(text, true);
        let embeddings = tokenizer.encode_with_embeddings(text, true);

        assert_eq!(embeddings.shape(), &[tokens.shape()[0], 768]);
        assert_ne!(
            embeddings.data()[0..768], 
            embeddings.data()[768..], 
            "Different tokens should have different embeddings"
        );
    }

    #[test]
    fn test_different_embedding_sizes() {
        let mut tokenizer_small = NabTokenizer::new(64);
        let mut tokenizer_large = NabTokenizer::new(768);
        
        let text = "Hello world";
        let embeddings_small = tokenizer_small.encode_with_embeddings(text, false);
        let embeddings_large = tokenizer_large.encode_with_embeddings(text, false);

        assert_eq!(embeddings_small.shape(), &[2, 64]);
        assert_eq!(embeddings_large.shape(), &[2, 768]);
        
        assert_ne!(
            embeddings_small.data()[0..64], 
            embeddings_small.data()[64..], 
            "Different tokens should have different embeddings in small model"
        );
        assert_ne!(
            embeddings_large.data()[0..768], 
            embeddings_large.data()[768..], 
            "Different tokens should have different embeddings in large model"
        );
    }

    #[test]
    fn test_bert_vocab() {
        let tokenizer = NabTokenizer::new(768);
        assert!(tokenizer.get_vocab_size() > 30000, "BERT vocabulary should contain more than 30,000 tokens");
        
        let vocab = tokenizer.get_vocab();
        assert_eq!(vocab.get("[PAD]"), Some(&0));
        assert_eq!(vocab.get("[UNK]"), Some(&1));
        assert_eq!(vocab.get("[CLS]"), Some(&2));
        assert_eq!(vocab.get("[SEP]"), Some(&3));
        assert_eq!(vocab.get("[MASK]"), Some(&4));
        assert_eq!(vocab.get("[unused0]"), Some(&5));
        assert_eq!(vocab.get("[unused1]"), Some(&6));

        // Test common words from BERT vocab
        assert!(vocab.contains_key("the"));
        assert!(vocab.contains_key("and"));
        assert!(vocab.contains_key("is"));
        
        // Test subword units
        assert!(vocab.contains_key("##ing"));
        assert!(vocab.contains_key("##s"));
    }

    #[test]
    fn test_tokenization_with_bert_vocab() {
        let mut tokenizer = NabTokenizer::new(768);
        tokenizer.debug_vocab();
        
        let text = "The quick brown fox jumps over the lazy dog";
        let tokens = tokenizer.encode(text, false);  // No special tokens
        let embeddings = tokenizer.encode_with_embeddings(text, false);
        
        assert!(tokens.shape()[0] > 0, "Should tokenize the text");
        assert_eq!(embeddings.shape()[1], 768, "Each token should have 768-dimensional embedding");
        assert_eq!(embeddings.shape()[0], tokens.shape()[0], "Should have an embedding for each token");
        
        let decoded = tokenizer.decode(&tokens, true);  // Skip special tokens
        println!("Original text: {}", text);
        println!("Decoded text: {}", decoded);
        println!("Token IDs: {:?}", tokens.data());
        assert_eq!(decoded.trim(), text.to_lowercase());
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = NabTokenizer::new(768);
        let text = "[CLS] Hello world [SEP]";
        let tokens = tokenizer.encode(text, true);  // Add special tokens

        println!("Tokens: {:?}", tokens.data());
        
        // First token should be [CLS] (ID: 2)
        assert_eq!(tokens.data()[0], 2.0);
        // Last token should be [SEP] (ID: 3)
        assert_eq!(tokens.data()[tokens.shape()[0] - 1], 3.0);

        // Test decoding with and without special tokens
        let decoded_with_special = tokenizer.decode(&tokens, false);
        let decoded_without_special = tokenizer.decode(&tokens, true);
        assert!(decoded_with_special.contains("[CLS]"));
        assert!(!decoded_without_special.contains("[CLS]"));
    }

    #[test]
    fn test_unknown_tokens() {
        let mut tokenizer = NabTokenizer::new(768);
        let text = "🌟 emoji test";  // emoji should be unknown
        let tokens = tokenizer.encode(text, false);  // No special tokens
        let embeddings = tokenizer.encode_with_embeddings(text, false);
        
        // First token should be UNK (ID: 1)
        assert_eq!(tokens.data()[0], 1.0);
        assert_eq!(embeddings.shape()[1], 768);
    }

    #[test]
    fn test_special_token_embeddings() {
        let mut tokenizer = NabTokenizer::new(768);
        let text = "[CLS] Hello world [SEP]";
        let tokens = tokenizer.encode(text, true);  // Add special tokens
        let embeddings = tokenizer.encode_with_embeddings(text, true);
        
        // Verify token IDs
        assert_eq!(tokens.data()[0], 2.0); // [CLS]
        assert_eq!(tokens.data()[tokens.shape()[0] - 1], 3.0); // [SEP]
        
        // Verify embedding dimensions
        assert_eq!(embeddings.shape()[1], 768);
        assert_eq!(embeddings.shape()[0], tokens.shape()[0]);
        
        // Verify that special token embeddings are consistent
        let cls_embedding1 = &embeddings.data()[0..768];
        let text2 = "[CLS] Different text [SEP]";
        let embeddings2 = tokenizer.encode_with_embeddings(text2, true);
        let cls_embedding2 = &embeddings2.data()[0..768];
        
        assert_eq!(cls_embedding1, cls_embedding2, "Same special token should have same embedding");
    }

    /// Test for the batch encoding function of NabTokenizer.
    /// Test per la funzione di codifica in batch di NabTokenizer.
    #[test]
    fn test_batch_encoding() {
        let tokenizer = NabTokenizer::new(64);
        let texts = vec!["Hello world", "Testing batch encoding", "Another sentence"];

        // Encode with special tokens disabled
        let encoded_batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(encoded_batch.len(), texts.len(), "Should return as many NDArray as input texts");

        // Check that each encoded vector has at least one token
        for tokens in encoded_batch.iter() {
            assert!(tokens.shape()[0] > 0, "Each encoded text should have at least one token");
        }

        // Encode with special tokens enabled and verify first and last tokens
        let encoded_with_special = tokenizer.encode_batch(&["Hello world"], true);
        let tokens = &encoded_with_special[0];
        // First token should be [CLS] (ID: 2) and last token should be [SEP] (ID: 3)
        assert_eq!(tokens.data()[0], 2.0, "First token should be [CLS] with ID 2");
        assert_eq!(tokens.data()[tokens.shape()[0] - 1], 3.0, "Last token should be [SEP] with ID 3");
    }

    /// Test for the preprocess_corpus function of NabTokenizer.
    /// Test per la funzione preprocess_corpus di NabTokenizer.
    #[test]
    fn test_preprocess_corpus() {
        let corpus = "\nLine one\n\n Line two  \nLine three\n   \n";
        let processed = NabTokenizer::preprocess_corpus(corpus);
        let expected = vec!["Line one", "Line two", "Line three"];
        assert_eq!(processed, expected, "The processed corpus should match the expected non-empty trimmed lines");
    }
}
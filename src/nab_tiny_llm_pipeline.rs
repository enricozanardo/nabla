use std::fs::File;
use std::io::{BufReader, Read};
use crate::nab_array::NDArray;
use crate::nab_tokenizer::NabTokenizer;

/// TinyLLMPipeline builds a training pipeline for a tiny language model.
/// It loads a dataset from a file, tokenizes the text, and prepares training inputs and targets.
/// 
/// Italian: TinyLLMPipeline costruisce una pipeline di training per un piccolo modello linguistico.
/// Carica un dataset da un file, tokenizza il testo e prepara input e target per l'addestramento.
pub struct TinyLLMPipeline {
    /// The tokenizer used for tokenizing the text
    pub tokenizer: NabTokenizer,
    /// The raw text loaded from dataset
    pub raw_text: String,
    /// The tokenized sequence as token IDs in an NDArray
    pub tokens: NDArray,
    /// Training input: token sequence except the last token
    pub input: NDArray,
    /// Training target: token sequence shifted by one (i.e., all tokens except the first)
    pub target: NDArray,
}

impl TinyLLMPipeline {
    /// Loads the dataset from a given file path.
    /// 
    /// # Arguments
    /// * `dataset_path` - Path to the dataset file, e.g., "datasets/alice_in_wonderland.txt".
    /// 
    /// # Returns
    /// A String containing the dataset content.
    /// 
    /// Italian: Carica il dataset da un percorso file dato.
    pub fn load_dataset(dataset_path: &str) -> Result<String, std::io::Error> {
        let file = File::open(dataset_path)?;
        let mut reader = BufReader::new(file);
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        Ok(content)
    }

    /// Tokenizes the entire corpus using the provided NabTokenizer.
    /// 
    /// # Arguments
    /// * `text` - The raw text corpus to tokenize.
    /// * `add_special_tokens` - Whether to add special tokens (for LM, typically false to have a continuous sequence).
    /// 
    /// # Returns
    /// An NDArray of token IDs.
    /// 
    /// Italian: Tokenizza l'intero corpus usando il NabTokenizer fornito.
    pub fn tokenize_corpus(tokenizer: &NabTokenizer, text: &str, add_special_tokens: bool) -> NDArray {
        // For language modeling, we want a continuous sequence of token IDs
        // We'll convert the text to lowercase (as tokenizer does) and then encode it.
        tokenizer.encode(text, add_special_tokens)
    }

    /// Prepares training data for language modeling by creating input and target sequences.
    /// Given a token sequence [t0, t1, ..., tN], the input is [t0, ..., t_(N-1)] and the target is [t1, ..., tN].
    /// 
    /// # Arguments
    /// * `tokens` - An NDArray of token IDs.
    /// 
    /// # Returns
    /// A tuple (input, target) where each is an NDArray.
    /// 
    /// Italian: Prepara i dati di training per il language modeling creando sequenze input e target.
    pub fn prepare_data(tokens: &NDArray) -> (NDArray, NDArray) {
        let total = tokens.size();
        assert!(total > 1, "Token sequence must contain more than one token");
        // Extract input tokens from index 0 to total-2 (inclusive), and target tokens from index 1 to total-1
        let input_data = tokens.data()[0..(total - 1)].to_vec();
        let target_data = tokens.data()[1..total].to_vec();
        // Shape is [length] (1D sequence). For training, we can treat them as 1D NDArray.
        let input = NDArray::from_vec(input_data);
        let target = NDArray::from_vec(target_data);
        (input, target)
    }

    /// Constructs a new TinyLLMPipeline from a dataset file and a given embedding dimension for the tokenizer.
    /// 
    /// # Arguments
    /// * `dataset_path` - Path to the dataset file.
    /// * `embedding_dim` - Embedding dimension for initializing the tokenizer.
    /// * `add_special_tokens` - Whether to add special tokens during tokenization.
    /// 
    /// # Returns
    /// A new TinyLLMPipeline instance with raw text, tokenized sequence, input and target.
    /// 
    /// Italian: Costruisce un nuovo TinyLLMPipeline a partire da un file di dataset e una dimensione di embedding.
    pub fn new(dataset_path: &str, embedding_dim: usize, add_special_tokens: bool) -> Result<Self, std::io::Error> {
        // Load dataset from file
        let raw_text = Self::load_dataset(dataset_path)?;

        // Initialize tokenizer (using default BERT vocab with given embedding_dim)
        let tokenizer = NabTokenizer::new(embedding_dim);

        // Tokenize the entire corpus (for LM we typically disable special tokens to have continuous text)
        let tokens = Self::tokenize_corpus(&tokenizer, &raw_text, add_special_tokens);
        
        // Prepare training data: input tokens and target tokens are shifted by one
        let (input, target) = Self::prepare_data(&tokens);

        Ok(TinyLLMPipeline {
            tokenizer,
            raw_text,
            tokens,
            input,
            target,
        })
    }

    /// Trains a tiny language model using a training pipeline.
    ///
    /// Steps:
    /// 1. Loads the dataset from the provided file path.
    /// 2. Tokenizes the corpus and prepares training data (input and target sequences).
    /// 3. Computes the vocabulary size as (max token ID + 1).
    /// 4. Initializes a TinyLLMModel with the computed vocabulary size, given embedding dimension, and number of transformer layers.
    /// 5. Trains the model using the train_language_model function.
    ///
    /// # Arguments
    ///
    /// * `dataset_path` - The file path to the dataset (e.g., "datasets/alice_in_wonderland.txt").
    /// * `embedding_dim` - Dimension of token embeddings and model hidden dimension.
    /// * `n_layers` - Number of transformer layers to stack.
    /// * `batch_size` - Mini-batch size for training.
    /// * `epochs` - Number of training epochs.
    /// * `learning_rate` - Learning rate for SGD updates.
    ///
    /// # Returns
    ///
    /// A Result containing a tuple (TinyLLMPipeline, TinyLLMModel, (loss_history, accuracy_history)) on success, or an error string.
    pub fn train_pipeline(dataset_path: &str, embedding_dim: usize, n_layers: usize, batch_size: usize, epochs: usize, learning_rate: f64) -> Result<(TinyLLMPipeline, crate::nab_tiny_llm_model::TinyLLMModel, (Vec<f64>, Vec<f64>)), String> {
        // Load dataset from file
        let raw_text = Self::load_dataset(dataset_path).map_err(|e| e.to_string())?;

        // Initialize tokenizer using NabTokenizer with given embedding_dim
        let tokenizer = crate::nab_tokenizer::NabTokenizer::new(embedding_dim);

        // Tokenize the entire corpus; for language modeling, we disable special tokens
        let tokens = Self::tokenize_corpus(&tokenizer, &raw_text, false);

        // Prepare training data: input tokens (all except last) and target tokens (all except first)
        let (input, target) = Self::prepare_data(&tokens);

        // Compute vocabulary size as (maximum token id + 1)
        let vocab_size = tokens.data().iter().cloned().fold(0.0, f64::max) as usize + 1;

        // Initialize a TinyLLMModel with the computed vocab_size, embedding_dim, and n_layers
        let mut model = crate::nab_tiny_llm_model::TinyLLMModel::new(vocab_size, embedding_dim, n_layers);

        // Train the model using the training loop
        let metrics = model.train_language_model(&input, &target, batch_size, epochs, learning_rate);

        // Construct a TinyLLMPipeline instance for inspection
        let pipeline = TinyLLMPipeline {
            tokenizer,
            raw_text,
            tokens,
            input,
            target,
        };

        Ok((pipeline, model, metrics))
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use std::fs::write;
    use std::path::Path;
    
    // Create a temporary dataset file for testing
    fn create_temp_dataset() -> String {
        let temp_path = "temp_alice.txt";
        let sample_text = "Alice was beginning to get very tired of sitting by her sister on the bank.";
        write(temp_path, sample_text).unwrap();
        temp_path.to_string()
    }

    #[test]
    fn test_load_dataset() {
        let temp_file = create_temp_dataset();
        let content = TinyLLMPipeline::load_dataset(&temp_file).unwrap();
        assert!(content.contains("Alice was beginning"), "Dataset should contain the sample text");
        // Cleanup temp file
        std::fs::remove_file(&temp_file).unwrap();
    }

    #[test]
    fn test_tokenize_corpus() {
        // Use a small sample text
        let sample_text = "Alice was beginning to get very tired.";
        let tokenizer = NabTokenizer::new(64);
        let tokens = TinyLLMPipeline::tokenize_corpus(&tokenizer, sample_text, false);
        // Check that the token NDArray has at least one token
        assert!(tokens.size() > 0, "Tokenized output should not be empty");
    }

    #[test]
    fn test_prepare_data() {
        // Create a dummy NDArray of token IDs
        let token_ids = NDArray::from_vec(vec![2.0, 5.0, 7.0, 9.0]);
        let (input, target) = TinyLLMPipeline::prepare_data(&token_ids);
        // Input should be [2, 5, 7] and target should be [5, 7, 9]
        assert_eq!(input.data(), &vec![2.0, 5.0, 7.0]);
        assert_eq!(target.data(), &vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tiny_llm_pipeline_new() {
        let temp_file = create_temp_dataset();
        // Use a small embedding dimension for testing
        let pipeline = TinyLLMPipeline::new(&temp_file, 64, false).unwrap();
        // Ensure raw text is loaded
        assert!(pipeline.raw_text.len() > 0);
        // Ensure tokens are created
        assert!(pipeline.tokens.size() > 0);
        // Ensure input and target sizes are correct (they should be one less than total tokens)
        assert_eq!(pipeline.input.size() + 1, pipeline.tokens.size());
        assert_eq!(pipeline.target.size() + 1, pipeline.tokens.size());
        // Cleanup temp file
        std::fs::remove_file(&temp_file).unwrap();
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod training_pipeline_tests {
    use super::*;
    use crate::nab_array::NDArray;
    
    #[test]
    fn test_train_pipeline() {
        // Use a small dataset file. For testing purposes, we assume the file exists.
        // In a real test, we might create a temporary file. Here we'll use a sample text similar to Alice.
        let sample_text = "Alice was beginning to get very tired of sitting by her sister on the bank.\nAnd so she began her adventures.";
        let temp_path = "temp_alice.txt";
        std::fs::write(temp_path, sample_text).unwrap();
        
        // Set parameters for the training pipeline
        let embedding_dim = 8;
        let n_layers = 1;
        let batch_size = 2;
        let epochs = 2;
        let learning_rate = 0.1;
        
        // Run the training pipeline
        let result = TinyLLMPipeline::train_pipeline(temp_path, embedding_dim, n_layers, batch_size, epochs, learning_rate);
        assert!(result.is_ok(), "Train pipeline should succeed");
        let (pipeline, model, metrics) = result.unwrap();
        
        // Check that the pipeline contains non-empty raw_text and tokens
        assert!(!pipeline.raw_text.is_empty(), "Raw text should not be empty");
        assert!(pipeline.tokens.size() > 0, "Tokens should not be empty");
        
        // Check that the model's vocab size matches the expected dimension from tokens
        let expected_vocab_size = pipeline.tokens.data().iter().cloned().fold(0.0, f64::max) as usize + 1;
        // The embedding layer of the model should have shape [vocab_size, embedding_dim]
        assert_eq!(model.embedding_layer.embedding_matrix.shape()[0], expected_vocab_size);
        
        // Check that training metrics (loss and accuracy histories) have length equal to epochs
        assert_eq!(metrics.0.len(), epochs, "Loss history length should equal number of epochs");
        assert_eq!(metrics.1.len(), epochs, "Accuracy history length should equal number of epochs");
        
        // Cleanup temporary file
        std::fs::remove_file(temp_path).unwrap();
    }
} 
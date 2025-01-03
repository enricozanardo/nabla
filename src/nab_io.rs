use std::fs::File;
use std::io::{self, Read, Write};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};
use serde::{Serialize, Deserialize};
use crate::NDArray;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct SerializableNDArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

/// Saves an NDArray to a .nab file with compression
pub fn save_nab(filename: &str, array: &NDArray) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut encoder = GzEncoder::new(file, Compression::default());
    let serializable_array = SerializableNDArray {
        data: array.data.clone(),
        shape: array.shape.clone(),
    };
    let serialized_data = bincode::serialize(&serializable_array).unwrap();
    encoder.write_all(&serialized_data)?;
    encoder.finish()?;
    Ok(())
}

/// Loads an NDArray from a compressed .nab file
pub fn load_nab(filename: &str) -> io::Result<NDArray> {
    let file = File::open(filename)?;
    let mut decoder = GzDecoder::new(file);
    let mut serialized_data = Vec::new();
    decoder.read_to_end(&mut serialized_data)?;
    let serializable_array: SerializableNDArray = bincode::deserialize(&serialized_data).unwrap();
    Ok(NDArray::new(serializable_array.data, serializable_array.shape))
}

/// Saves multiple NDArrays to a .nab file
///
/// # Arguments
///
/// * `filename` - The name of the file to save the arrays to.
/// * `arrays` - A vector of tuples containing the name and NDArray to save.
pub fn savez(filename: &str, arrays: Vec<(&str, &NDArray)>) -> io::Result<()> {
    let mut file = File::create(filename)?;
    for (name, array) in arrays {
        let shape_str = array.shape().iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
        let data_str = array.data().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        writeln!(file, "{}:{};{}", name, shape_str, data_str)?;
    }
    Ok(())
} 
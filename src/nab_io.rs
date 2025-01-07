use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use crate::nab_array::NDArray;

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
        data: array.data().to_vec(),
        shape: array.shape().to_vec(),
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
#[allow(dead_code)]
pub fn savez_nab(filename: &str, arrays: Vec<(&str, &NDArray)>) -> io::Result<()> {
    let mut file = File::create(filename)?;
    for (name, array) in arrays {
        let shape_str = array.shape().iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
        let data_str = array.data().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        writeln!(file, "{}:{};{}", name, shape_str, data_str)?;
    }
    Ok(())
}

#[allow(dead_code)]
pub fn loadz_nab(filename: &str) -> io::Result<HashMap<String, NDArray>> {
    let file = File::open(filename)?;
    let mut arrays = HashMap::new();
    
    // Read the file line by line
    for line in io::BufReader::new(file).lines() {
        let line = line?;
        // Split the line into name, shape, and data parts
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue;
        }
        
        let name = parts[0].to_string();
        let shape_and_data: Vec<&str> = parts[1].split(';').collect();
        if shape_and_data.len() != 2 {
            continue;
        }
        
        // Parse shape
        let shape: Vec<usize> = shape_and_data[0]
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
            
        // Parse data
        let data: Vec<f64> = shape_and_data[1]
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
            
        arrays.insert(name, NDArray::new(data, shape));
    }
    
    Ok(arrays)
} 
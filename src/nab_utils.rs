use crate::nab_array::NDArray;
use crate::nab_io::{save_nab, load_nab};


impl NDArray {


    /// Normalizes the array values to range [0, 1] using min-max normalization
    pub fn normalize(&mut self) {
        let min_val = self.data().iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = self.data().iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Avoid division by zero if all values are the same
        let range = max_val - min_val;
        if range != 0.0 {
            self.data.iter_mut().for_each(|x| {
                *x = (*x - min_val) / range;
            });
        }
    }

    /// Normalizes the array values using specified min and max values
    pub fn normalize_with_range(&mut self, min_val: f64, max_val: f64) {
        let range = max_val - min_val;
        if range != 0.0 {
            self.data.iter_mut().for_each(|x| {
                *x = (*x - min_val) / range;
            });
        }
    }

    pub fn data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }

    /// Loads a dataset from .nab files and splits it into training and testing sets
    ///
    /// # Arguments
    ///
    /// * `path` - Base path for the .nab files (e.g., "mnist")
    /// * `train_percent` - Percentage of data to use for training (e.g., 80 for 80%)
    ///
    /// # Returns
    ///
    /// A tuple containing ((train_images, train_labels), (test_images, test_labels))
    #[allow(dead_code)]
    pub fn load_and_split_dataset(path: &str, train_percent: f64) -> std::io::Result<((NDArray, NDArray), (NDArray, NDArray))> {
        let images = load_nab(&format!("{}_images.nab", path))?;
        let labels = load_nab(&format!("{}_labels.nab", path))?;

        let num_samples = images.shape()[0];
        let train_size = ((train_percent / 100.0) * num_samples as f64).round() as usize;

        let train_images = NDArray::new(
            images.data()[..train_size * images.shape()[1] * images.shape()[2]].to_vec(),
            vec![train_size, images.shape()[1], images.shape()[2]],
        );

        let test_images = NDArray::new(
            images.data()[train_size * images.shape()[1] * images.shape()[2]..].to_vec(),
            vec![num_samples - train_size, images.shape()[1], images.shape()[2]],
        );

        let train_labels = NDArray::new(
            labels.data()[..train_size].to_vec(),
            vec![train_size],
        );

        let test_labels = NDArray::new(
            labels.data()[train_size..].to_vec(),
            vec![num_samples - train_size],
        );

        Ok(((train_images, train_labels), (test_images, test_labels)))
    }

    /// Converts CSV data to .nab format
    /// 
    /// # Arguments
    /// 
    /// * `csv_path` - Path to the CSV file
    /// * `output_path` - Path where to save the .nab file
    /// * `shape` - Shape of the resulting array (e.g., [60000, 28, 28] for MNIST images)
    #[allow(dead_code)]
    #[allow(unused_variables)]
    pub fn csv_to_nab(csv_path: &str, output_path: &str, shape: Vec<usize>, skip_first_column: bool) -> std::io::Result<()> {
        let mut rdr = csv::Reader::from_path(csv_path)?;
        let mut data = Vec::new();
        let mut row_count = 0;

        for result in rdr.records() {
            row_count += 1;
            let record = result?;
            let start_index = if skip_first_column { 1 } else { 0 };
            
            for value in record.iter().skip(start_index) {
                let num: f64 = value.parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                data.push(num);
            }
        }

        let expected_size: usize = shape.iter().product();

        if data.len() != expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Data length ({}) does not match expected size from shape ({:?}): {}",
                    data.len(), shape, expected_size)
            ));
        }

        let array = NDArray::from_vec_reshape(data, shape);
        save_nab(output_path, &array)?;

        Ok(())
    }

    /// Displays the array in a formatted way similar to numpy's print format
    ///
    /// # Returns
    ///
    /// A string representation of the array
    pub fn display(&self) -> String {
        match self.ndim() {
            1 => self.format_1d(),
            2 => self.format_2d(),
            3 => self.format_3d(),
            _ => format!("Array of shape {:?}", self.shape()),
        }
    }

    fn format_1d(&self) -> String {
        format!("[{}]", self.data().iter()
            .map(|x| format!("{:3.0}", x))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn format_2d(&self) -> String {
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        
        let mut result = String::from("[\n");
        for i in 0..rows {
            result.push_str(" [");
            for j in 0..cols {
                let value = self.get_2d(i, j);
                result.push_str(&format!("{:3.0}", value));
                if j < cols - 1 {
                    result.push_str(" ");
                }
            }
            result.push_str("]");
            if i < rows - 1 {
                result.push_str("\n");
            }
        }
        result.push_str("\n]");
        result
    }

    fn format_3d(&self) -> String {
        let depth = self.shape()[0];
        let mut result = String::new();
        
        for d in 0..depth {
            let slice = self.sub_matrix(d, d+1, 0, self.shape()[2]);
            if d > 0 {
                result.push_str("\n\n");
            }
            result.push_str(&format!("Layer {}:\n{}", d, slice.format_2d()));
        }
        result
    }

}


#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use std::io;
    use crate::nab_io::{savez_nab, loadz_nab};

    #[test]
    fn test_save_and_load_nab() -> std::io::Result<()> {
        let array = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        save_nab("test.nab", &array)?;
        let loaded_array = load_nab("test.nab")?;
        assert_eq!(array.data(), loaded_array.data());
        assert_eq!(array.shape(), loaded_array.shape());

        // Clean up test file
        std::fs::remove_file("test.nab")?;
        Ok(())
    }

    #[test]
    fn test_savez_and_loadz_nab() -> std::io::Result<()> {
        let array1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let array2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        savez_nab("test_multiple.nab", vec![("x", &array1), ("y", &array2)])?;

        let arrays = loadz_nab("test_multiple.nab")?;
        assert_eq!(arrays.get("x").unwrap().data(), array1.data());
        assert_eq!(arrays.get("y").unwrap().data(), array2.data());

        // Clean up test file
        std::fs::remove_file("test_multiple.nab")?;
        Ok(())
    }
}
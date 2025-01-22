use crate::nab_array::NDArray;
use crate::nab_io::save_nab;
use crate::nab_utils::NabUtils;

pub struct NabMnist;


impl NabMnist {
        /// Converts MNIST CSV data to image and label .nab files
    /// 
    /// # Arguments
    /// 
    /// * `csv_path` - Path to the CSV file
    /// * `images_path` - Path where to save the images .nab file
    /// * `labels_path` - Path where to save the labels .nab file
    /// * `image_shape` - Shape of a single image (e.g., [28, 28])
    #[allow(dead_code)]
    pub fn mnist_csv_to_nab(
        csv_path: &str,
        images_path: &str,
        labels_path: &str,
        image_shape: Vec<usize>
    ) -> std::io::Result<()> {
        let mut rdr = csv::Reader::from_path(csv_path)?;
        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut sample_count = 0;

        for result in rdr.records() {
            let record = result?;
            sample_count += 1;

            if let Some(label) = record.get(0) {
                labels.push(label.parse::<f64>().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?);
            }

            for value in record.iter().skip(1) {
                let pixel: f64 = value.parse().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                images.push(pixel);
            }
        }

        let mut full_image_shape = vec![sample_count];
        full_image_shape.extend(image_shape);
        let images_array = NDArray::new(images, full_image_shape);
        save_nab(images_path, &images_array)?;

        let labels_array = NDArray::new(labels, vec![sample_count]);
        save_nab(labels_path, &labels_array)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use crate::nab_io::load_nab;

    #[test]
    fn test_mnist_load_and_split_dataset() -> std::io::Result<()> {
        std::fs::create_dir_all("datasets")?;

        NabMnist::mnist_csv_to_nab(
            "csv/mnist_test.csv",
            "datasets/mnist_test_images.nab",
            "datasets/mnist_test_labels.nab",
            vec![28, 28]
        )?;

        let ((train_images, train_labels), (test_images, test_labels)) = 
            NabUtils::load_and_split_dataset("datasets/mnist_test", 80.0)?;

        assert_eq!(train_images.shape()[0] + test_images.shape()[0], 999);
        assert_eq!(train_labels.shape()[0] + test_labels.shape()[0], 999);

        // std::fs::remove_file("datasets/mnist_test_images.nab")?;
        // std::fs::remove_file("datasets/mnist_test_labels.nab")?;

        Ok(())
    }

    #[test]
    fn test_mnist_csv_to_nab_conversion() -> io::Result<()> {
        // Define paths for the CSV and .nab files
        let csv_path = "csv/mnist_test.csv";
        let nab_path = "datasets/mnist_test";
        let expected_shape = vec![999, 28, 28];
        
        println!("Starting test with CSV: {}", csv_path);

        // Convert CSV to .nab, skipping the first column
        NabUtils::csv_to_nab(csv_path, nab_path, expected_shape.clone(), true)?;

        // Load the .nab file
        let images = load_nab(nab_path)?;
        println!("Loaded NAB file with shape: {:?}", images.shape());

        // Verify the shape of the data
        assert_eq!(images.shape(), &expected_shape, 
            "Shape mismatch: expected {:?}, got {:?}", expected_shape, images.shape());

        // Clean up the .nab file
        // std::fs::remove_file(nab_path)?;
        // println!("Test cleanup complete");

        Ok(())
    }
    

    #[test]
    fn test_extract_and_print_sample() -> io::Result<()> {
        // Ensure the datasets directory exists
        std::fs::create_dir_all("datasets")?;

        // Convert CSV to .nab files if not already done
        NabMnist::mnist_csv_to_nab(
            "csv/mnist_test.csv",
            "datasets/mnist_test_images.nab",
            "datasets/mnist_test_labels.nab",
            vec![28, 28]
        )?;

        // Load the dataset
        let ((train_images, train_labels), _) = 
            NabUtils::load_and_split_dataset("datasets/mnist_test", 80.0)?;

        // Extract and print the 42nd entry
        println!("Label of 42nd entry: {}", train_labels.get(42));
        println!("Image of 42nd entry:");
        let image_42: NDArray = train_images.extract_sample(42);
        image_42.pretty_print(0);

        // Clean up
        // std::fs::remove_file("datasets/mnist_test_images.nab")?;
        // std::fs::remove_file("datasets/mnist_test_labels.nab")?;

        Ok(())
    }

    #[test]
    fn test_mnist_normalize() -> std::io::Result<()> {
        let ((mut train_images, _), _) = 
            NabUtils::load_and_split_dataset("datasets/mnist_test", 80.0)?;
        
        NabUtils::normalize_with_range(&mut train_images, 0.0, 255.0);
        
        // Add this to check actual values
        let gray_image_42 = train_images.extract_sample(42);
        println!("First few raw values: {:?}", &gray_image_42.data()[..5]);
        gray_image_42.pretty_print(4); // Add precision parameter to show 3 decimal places

        Ok(())
    }
} 


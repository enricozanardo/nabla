# Nabla-ML

Nabla-ML is a Rust library inspired by NumPy, providing a multi-dimensional array implementation with various mathematical and array manipulation functionalities.

## Features

- **Array Creation**: Create 1D and 2D arrays using vectors and matrices.
- **Random Arrays**: Generate arrays with random numbers, including uniform and normal distributions.
- **Arithmetic Operations**: Perform element-wise addition, subtraction, multiplication, and division.
- **Mathematical Functions**: Apply functions like square root, exponential, sine, cosine, logarithm, hyperbolic tangent, ReLU, Leaky ReLU, and Sigmoid to arrays.
- **Array Reshaping**: Change the shape of arrays while maintaining data integrity.
- **Indexing and Slicing**: Access and modify elements using indices and slices.
- **Conditional Selection**: Filter arrays based on conditions.
- **Array Attributes**: Retrieve attributes like number of dimensions, shape, size, and data type.
- **Axis Manipulation**: Add new axes to arrays to increase dimensionality.
- **File I/O**: Save and load arrays using a custom `.nab` file format with compression.
- **Loss Functions**: Calculate Mean Squared Error (MSE) and Cross-Entropy Loss.
- **Linear Regression**: Perform linear regression using gradient descent with Mean Squared Error as the loss function.

## Usage

### Creating Arrays

```rust
use nabla_ml::NDArray;
// Create a 1D array
let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
// Create a 2D array
let mat = NDArray::from_matrix(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
]);
```

### Random Arrays

```rust
// Create a 1D array of random numbers between 0 and 1
let rand_arr = NDArray::rand(5);
// Create a 2D array of random integers between 1 and 10
let rand_int_mat = NDArray::randint_2d(1, 10, 3, 3);
```

### Arithmetic Operations

```rust
let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
// Element-wise addition
let sum = arr1.clone() + arr2;
// Scalar multiplication
let scaled = arr1.clone() * 2.0;
```

### Mathematical Functions

```rust
let arr = NDArray::from_vec(vec![1.0, 4.0, 9.0]);
// Calculate square roots
let sqrt_arr = arr.sqrt();
// Calculate exponentials
let exp_arr = arr.exp();
// Calculate hyperbolic tangent
let tanh_arr = arr.tanh();
// Apply ReLU
let relu_arr = arr.relu();
// Apply Leaky ReLU with alpha = 0.01
let leaky_relu_arr = arr.leaky_relu(0.01);
// Apply Sigmoid
let sigmoid_arr = arr.sigmoid();
```

![Leaky ReLU](./docs/leaky_relu.png)
![Sigmoid](./docs/sigmoid.png)
![ReLU](./docs/relu.png)

### Reshaping Arrays

```rust
let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let reshaped = arr.reshape(vec![2, 3]);
```

### Indexing and Slicing

```rust
let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
// Access an element
let first = arr.get(0);
// Slice the array
let slice = arr.slice(1, 4);
```

### Conditional Selection

```rust
let arr = NDArray::from_vec(vec![0.69, 0.94, 0.66, 0.73, 0.83]);
// Filter elements greater than 0.7
let filtered = arr.filter(|&x| x > 0.7);
```

### Array Attributes

```rust
let arr = NDArray::from_matrix(vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
]);
// Get number of dimensions
let ndim = arr.ndim();
// Get shape
let shape = arr.shape();
// Get size
let size = arr.size();
// Get data type
let dtype = arr.dtype();
```

### Axis Manipulation

```rust
let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
// Add a new axis
let row_vector = arr.new_axis(0);
let col_vector = arr.new_axis(1);
```

### File I/O with .nab Format

#### Save and Load a Single Array

```rust
use nabla_ml::{NDArray, save_nab, load_nab};

// Create an NDArray
let array = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

// Save the array to a .nab file
save_nab("data.nab", &array).expect("Failed to save array");

// Load the array from the .nab file
let loaded_array = load_nab("data.nab").expect("Failed to load array");

// Verify the loaded data
assert_eq!(array.data(), loaded_array.data());
assert_eq!(array.shape(), loaded_array.shape());
```

#### Save Multiple Arrays

```rust
use nabla_ml::{NDArray, savez_nab};

fn main() -> std::io::Result<()> {
    let array1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
    let array2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);

    // Save arrays with specified names
    savez_nab("data.nab", vec![("x", &array1), ("y", &array2)])?;

    Ok(())
}
```

#### Load Multiple Arrays

```rust
use nabla_ml::{NDArray, loadz_nab};
use std::collections::HashMap;

fn main() -> std::io::Result<()> {
    // Load multiple arrays
    let arrays: HashMap<String, NDArray> = loadz_nab("data.nab")?;
    
    // Access individual arrays by name
    let x = arrays.get("x").unwrap();
    let y = arrays.get("y").unwrap();

    Ok(())
}
```

### MNIST Dataset Handling

The library provides specialized functionality for handling MNIST-like datasets stored in CSV format.

```rust
use nabla_ml::NDArray;

// Convert MNIST CSV data to .nab format
NDArray::mnist_csv_to_nab(
    "data/mnist_train.csv",          // Source CSV file
    "datasets/mnist_images.nab",     // Output path for images
    "datasets/mnist_labels.nab",     // Output path for labels
    vec![28, 28]                    // Shape of each image
)?;

// Load and split the dataset into training and test sets
let ((train_images, train_labels), (test_images, test_labels)) = 
    NDArray::load_and_split_dataset("datasets/mnist", 80.0)?;

// Access the data
println!("Training samples: {}", train_images.shape()[0]);
println!("Test samples: {}", test_images.shape()[0]);
```

The MNIST dataset handling includes:
- Converting CSV format to compressed .nab files
- Automatic separation of images and labels
- Dataset splitting into training and test sets
- Proper reshaping of image data

## Loss Functions

### Mean Squared Error (MSE)

The MSE measures the average of the squares of the differences between predicted and actual values.

```rust
use nabla_ml::NDArray;

let y_true = NDArray::from_vec(vec![1.0, 0.0, 1.0, 1.0]);
let y_pred = NDArray::from_vec(vec![0.9, 0.2, 0.8, 0.6]);
let mse = NDArray::mean_squared_error(&y_true, &y_pred);
println!("MSE: {}", mse);
```

### Cross-Entropy Loss

Cross-Entropy Loss is commonly used for classification problems, especially with softmax outputs.

```rust
use nabla_ml::NDArray;

let y_true = NDArray::from_matrix(vec![
    vec![1.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
]);
let y_pred = NDArray::from_matrix(vec![
    vec![0.7, 0.2, 0.1],
    vec![0.1, 0.8, 0.1],
    vec![0.05, 0.15, 0.8],
]);
let cross_entropy = NDArray::cross_entropy_loss(&y_true, &y_pred);
println!("Cross-Entropy Loss: {}", cross_entropy);
```

## Linear Regression

Perform linear regression using gradient descent with Mean Squared Error as the loss function.

### Single Feature

```rust
use nabla_ml::NDArray;

// Generate a simple dataset
let X = NDArray::from_matrix((0..100).map(|_| vec![2.0 * rand::random::<f64>()]).collect());
let y = NDArray::from_vec(X.data().iter().map(|&x| 4.0 + 3.0 * x + rand::random::<f64>()).collect());

// Apply linear regression
let (theta, history) = NDArray::linear_regression(&X, &y, 0.01, 2000);

println!("Final Parameters:");
println!("Intercept (theta_0): {:.4}, Weight (theta_1): {:.4}", theta[0], theta[1]);

// Predict on new data
let X_new = NDArray::from_matrix(vec![vec![0.0], vec![2.0]]);
let y_pred: Vec<f64> = X_new.data().iter().map(|&x| theta[0] + theta[1] * x).collect();

println!("Predictions:");
println!("{:?}", y_pred);
```

### Multiple Features

```rust
use nabla_ml::NDArray;

// Generate a simple dataset with two features
let X = NDArray::from_matrix(vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 1.0],
    vec![2.0, 1.0],
    vec![1.0, 2.0],
    vec![2.0, 2.0],
]);
let y = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]); // y = 1 + 1*x1 + 2*x2

// Apply linear regression
let (theta, history) = NDArray::linear_regression(&X, &y, 0.01, 1000);

println!("Final Parameters:");
println!("Intercept (theta_0): {:.4}, Coefficients: {:?}", theta[0], &theta[1..]);

// Predict on new data
let X_new = NDArray::from_matrix(vec![vec![0.0, 0.0], vec![2.0, 2.0]]);
let y_pred: Vec<f64> = X_new.data().chunks(X_new.shape()[1]).map(|x| {
    theta[0] + x.iter().zip(&theta[1..]).map(|(&xi, &ti)| xi * ti).sum::<f64>()
}).collect();

println!("Predictions:");
println!("{:?}", y_pred);
```

![Linear Regression](./docs/regression_plot.png)
![Loss History](./docs/loss_history.png)

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

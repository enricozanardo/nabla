use rand::Rng;
use rand_distr::StandardNormal;
use std::ops::{Add, Sub, Mul};

#[derive(Debug, Clone)]
pub struct NDArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl NDArray {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    #[allow(dead_code)]
    pub fn from_matrix(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = data.get(0).map_or(0, |row| row.len());
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        Self::new(flat_data, vec![rows, cols])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns a reference to the data of the array
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Creates a 2D array (matrix) of random numbers between 0 and 1
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random numbers.
    #[allow(dead_code)]
    pub fn rand_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen()).collect();
        Self::new(data, vec![rows, cols])
    }


    /// Creates a 1D array of random numbers following a normal distribution
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random numbers from a normal distribution.
    #[allow(dead_code)]
    pub fn randn(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random numbers following a normal distribution
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random numbers from a normal distribution.
    #[allow(dead_code)]
    pub fn randn_2d(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.sample(StandardNormal)).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Creates a 1D array of random integers between `low` and `high`
    ///
    /// # Arguments
    ///
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (exclusive).
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random integers.
    #[allow(dead_code)]
    pub fn randint(low: i32, high: i32, size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen_range(low..high) as f64).collect();
        Self::from_vec(data)
    }

    /// Creates a 2D array (matrix) of random integers between `low` and `high`
    ///
    /// # Arguments
    ///
    /// * `low` - The lower bound (inclusive).
    /// * `high` - The upper bound (exclusive).
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with random integers.
    #[allow(dead_code)]
    pub fn randint_2d(low: i32, high: i32, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen_range(low..high) as f64).collect();
        Self::new(data, vec![rows, cols])
    }

    /// Reshapes the array to the specified shape, allowing one dimension to be inferred
    ///
    /// # Arguments
    ///
    /// * `new_shape` - A vector representing the new shape, with at most one dimension as `-1`.
    ///
    /// # Returns
    ///
    /// A new NDArray with the specified shape.
    pub fn reshape(&self, new_shape: Vec<isize>) -> Self {
        let total_elements = self.data.len();
        let mut inferred_index = None;
        let mut known_elements: usize = 1;

        // First pass: calculate product of known dimensions
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == -1 {
                if inferred_index.is_some() {
                    panic!("Only one dimension can be inferred");
                }
                inferred_index = Some(i);
            } else if dim <= 0 {
                panic!("Dimensions must be positive (except -1 for inference)");
            } else {
                known_elements = known_elements.checked_mul(dim as usize)
                    .expect("Shape multiplication overflow");
            }
        }

        // Calculate the inferred dimension
        let mut final_shape: Vec<usize> = Vec::with_capacity(new_shape.len());
        if let Some(idx) = inferred_index {
            // Ensure we can evenly divide the total elements
            assert!(total_elements % known_elements == 0, 
                "Cannot infer dimension that evenly divides total elements");
            
            for (i, &dim) in new_shape.iter().enumerate() {
                if i == idx {
                    final_shape.push(total_elements / known_elements);
                } else {
                    final_shape.push(dim as usize);
                }
            }
        } else {
            final_shape = new_shape.iter()
                .map(|&x| x as usize)
                .collect();
        }

        // Verify total size matches
        let new_total = final_shape.iter().product();
        assert_eq!(total_elements, new_total, 
            "New shape must have the same number of elements as the original array");

        Self::new(self.data.clone(), final_shape)
    }

     /// Returns the maximum value in the array
    ///
    /// # Returns
    ///
    /// The maximum value as an f64.
    #[allow(dead_code)]
    pub fn max(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Returns the index of the maximum value in the array
    ///
    /// # Returns
    ///
    /// The index of the maximum value.
    // #[allow(dead_code)]
    // pub fn argmax(&self) -> usize {
    //     self.data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    // }

    /// Returns the indices of maximum values
    /// For 1D arrays: returns a single index
    /// For 2D arrays: returns indices along the specified axis
    pub fn argmax(&self, axis: Option<usize>) -> Vec<usize> {
        println!("Debug: Array shape: {:?}, ndim: {}, axis: {:?}", 
            self.shape, self.ndim(), axis);
            
        match axis {
            None => {
                println!("Debug: Using global argmax");
                vec![self.data.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()]
            },
            Some(1) => {
                if self.ndim() != 2 {
                    println!("Debug: Expected 2D array for axis=1, got {}D array", self.ndim());
                    panic!("Array must be 2D when using axis=1");
                }
                
                let rows = self.shape[0];
                let cols = self.shape[1];
                println!("Debug: Processing 2D array with shape [{}, {}]", rows, cols);
                
                (0..rows).map(|i| {
                    let start = i * cols;
                    let row_slice = &self.data[start..start + cols];
                    row_slice.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap()
                }).collect()
            },
            Some(ax) => {
                println!("Debug: Unsupported axis: {}", ax);
                panic!("Only axis=1 (row-wise) is supported for argmax")
            }
        }
    }


    /// Returns the minimum value in the array
    ///
    /// # Returns
    ///
    /// The minimum value as an f64.
    #[allow(dead_code)]
    pub fn min(&self) -> f64 {
        *self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Creates an NDArray from a flat vector and a specified shape
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f64 values representing the array's data.
    /// * `shape` - A vector of usize values representing the dimensions of the array.
    ///
    /// # Returns
    ///
    /// A new NDArray instance.
    #[allow(dead_code)]
    pub fn from_vec_reshape(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape dimensions");
        NDArray { data, shape }
    }

    /// Extracts a single sample from a batch of N-dimensional arrays
    ///
    /// # Arguments
    ///
    /// * `sample_index` - The index of the sample to extract
    ///
    /// # Returns
    ///
    /// A new NDArray containing just the specified sample with N-1 dimensions
    #[allow(dead_code)]
    pub fn extract_sample(&self, sample_index: usize) -> Self {
        assert!(self.ndim() >= 2, "Array must have at least 2 dimensions");
        assert!(sample_index < self.shape[0], "Sample index out of bounds");

        let sample_size: usize = self.shape.iter().skip(1).product();
        let start_index = sample_index * sample_size;
        let end_index = start_index + sample_size;
        
        // Create new shape without the first dimension
        let new_shape: Vec<usize> = self.shape.iter().skip(1).cloned().collect();
        
        NDArray::new(
            self.data[start_index..end_index].to_vec(),
            new_shape
        )
    }

    /// Pretty prints an N-dimensional array
    ///
    /// # Arguments
    ///
    /// * `precision` - The number of decimal places to round each value to.
    #[allow(dead_code)]
    pub fn pretty_print(&self, precision: usize) {
        let indent_str = " ".repeat(precision);
        
        let format_value = |x: f64| -> String {
            if x == 0.0 {
                format!("{:.1}", x)
            } else {
                format!("{:.*}", precision, x)
            }
        };
        
        match self.ndim() {
            1 => println!("{}[{}]", indent_str, self.data.iter()
                .map(|&x| format_value(x))
                .collect::<Vec<_>>()
                .join(" ")),
                
            2 => {
                println!("{}[", indent_str);
                for i in 0..self.shape[0] {
                    print!("{}  [", indent_str);
                    for j in 0..self.shape[1] {
                        print!("{}", format_value(self.get_2d(i, j)));
                        if j < self.shape[1] - 1 {
                            print!(" ");
                        }
                    }
                    println!("]");
                }
                println!("{}]", indent_str);
            },
            
            _ => {
                println!("{}[", indent_str);
                for i in 0..self.shape[0] {
                    let slice = self.extract_sample(i);
                    slice.pretty_print(precision + 2);
                }
                println!("{}]", indent_str);
            }
        }
    }


    /// Returns a specific element from the array
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to retrieve.
    ///
    /// # Returns
    ///
    /// The element at the specified index.
    #[allow(dead_code)]
    pub fn get(&self, index: usize) -> f64 {
        self.data[index]
    }

    /// Creates a 1D array with a range of numbers
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the range (inclusive).
    /// * `stop` - The stopping value of the range (exclusive).
    /// * `step` - The step size between each value in the range.
    ///
    /// # Returns
    ///
    /// A 1D NDArray containing the range of numbers.
    #[allow(dead_code)]
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut current = start;
        while current < stop {
            data.push(current);
            current += step;
        }
        Self::from_vec(data)
    }

        /// Creates a 1D array filled with zeros
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with zeros.
    #[allow(dead_code)]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        NDArray {
            data: vec![0.0; total_size],
            shape,
        }
    }


    /// Creates a 2D array (matrix) filled with zeros
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with zeros.
    #[allow(dead_code)]
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![0.0; rows * cols], vec![rows, cols])
    }

    /// Creates a 1D array filled with ones
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with ones.
    #[allow(dead_code)]
    pub fn ones(size: usize) -> Self {
        Self::from_vec(vec![1.0; size])
    }

        /// Creates a 2D array (matrix) filled with ones
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    ///
    /// # Returns
    ///
    /// A 2D NDArray filled with ones.
    #[allow(dead_code)]
    pub fn ones_2d(rows: usize, cols: usize) -> Self {
        Self::new(vec![1.0; rows * cols], vec![rows, cols])
    }

    /// Creates a 1D array with evenly spaced numbers over a specified interval
    ///
    /// # Arguments
    ///
    /// * `start` - The starting value of the interval.
    /// * `end` - The ending value of the interval.
    /// * `num` - The number of evenly spaced samples to generate.
    /// * `precision` - The number of decimal places to round each value to.
    ///
    /// # Returns
    ///
    /// A 1D NDArray containing the evenly spaced numbers.
    #[allow(dead_code)]
    pub fn linspace(start: f64, end: f64, num: usize, precision: usize) -> Self {
        assert!(num > 1, "Number of samples must be greater than 1");
        let step = (end - start) / (num - 1) as f64;
        let mut data = Vec::with_capacity(num);
        let factor = 10f64.powi(precision as i32);
        for i in 0..num {
            let value = start + step * i as f64;
            let rounded_value = (value * factor).round() / factor;
            data.push(rounded_value);
        }
        Self::from_vec(data)
    }

    /// Creates an identity matrix of size `n x n`
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the identity matrix.
    ///
    /// # Returns
    ///
    /// An `n x n` identity matrix as an NDArray.
    #[allow(dead_code)]
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(data, vec![n, n])
    }

    /// Creates a 1D array of random numbers between 0 and 1
    ///
    /// # Arguments
    ///
    /// * `size` - The number of elements in the array.
    ///
    /// # Returns
    ///
    /// A 1D NDArray filled with random numbers.
    #[allow(dead_code)]
    pub fn rand(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
        Self::from_vec(data)
    }


    /// Returns a sub-matrix from a 2D array
    ///
    /// # Arguments
    ///
    /// * `row_start` - The starting row index of the sub-matrix.
    /// * `row_end` - The ending row index of the sub-matrix (exclusive).
    /// * `col_start` - The starting column index of the sub-matrix.
    /// * `col_end` - The ending column index of the sub-matrix (exclusive).
    ///
    /// # Returns
    ///
    /// A new NDArray representing the specified sub-matrix.
    #[allow(dead_code)]
    pub fn sub_matrix(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        assert_eq!(self.ndim(), 2, "sub_matrix is only applicable to 2D arrays");
        let cols = self.shape[1];
        let mut data = Vec::new();
        for row in row_start..row_end {
            for col in col_start..col_end {
                data.push(self.data[row * cols + col]);
            }
        }
        Self::new(data, vec![row_end - row_start, col_end - col_start])
    }

    /// Sets a specific element in the array
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to set.
    /// * `value` - The value to set the element to.
    #[allow(dead_code)]
    pub fn set(&mut self, index: usize, value: f64) {
        self.data[index] = value;
    }

    /// Sets a range of elements in the array to a specific value
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the range.
    /// * `end` - The ending index of the range (exclusive).
    /// * `value` - The value to set the elements to.
    #[allow(dead_code)]
    pub fn set_range(&mut self, start: usize, end: usize, value: f64) {
        for i in start..end {
            self.data[i] = value;
        }
    }

     /// Returns a copy of the array
    ///
    /// # Returns
    ///
    /// A new NDArray that is a copy of the original.
    #[allow(dead_code)]
    pub fn copy(&self) -> Self {
        Self::new(self.data.clone(), self.shape.clone())
    }

    /// Returns a view (slice) of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the view.
    /// * `end` - The ending index of the view (exclusive).
    ///
    /// # Returns
    ///
    /// A slice of f64 values representing the specified view.
    #[allow(dead_code)]
    pub fn view(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start..end]
    }

        /// Returns a mutable view (slice) of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the view.
    /// * `end` - The ending index of the view (exclusive).
    ///
    /// # Returns
    ///
    /// A mutable slice of f64 values representing the specified view.
    #[allow(dead_code)]
    pub fn view_mut(&mut self, start: usize, end: usize) -> &mut [f64] {
        &mut self.data[start..end]
    }


    /// Returns a specific element from a 2D array
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// The element at the specified row and column.
    #[allow(dead_code)]
    pub fn get_2d(&self, row: usize, col: usize) -> f64 {
        assert_eq!(self.ndim(), 2, "get_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col]
    }

    /// Sets a specific element in a 2D array
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    /// * `value` - The value to set the element to.
    #[allow(dead_code)]
    pub fn set_2d(&mut self, row: usize, col: usize, value: f64) {
        assert_eq!(self.ndim(), 2, "set_2d is only applicable to 2D arrays");
        let cols = self.shape[1];
        self.data[row * cols + col] = value;
    }

    /// Adds a new axis to the array at the specified position
    ///
    /// # Arguments
    ///
    /// * `axis` - The position at which to add the new axis.
    ///
    /// # Returns
    ///
    /// A new NDArray with an additional axis.
    #[allow(dead_code)]
    pub fn new_axis(&self, axis: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Self::new(self.data.clone(), new_shape)
    }

    /// Expands the dimensions of the array by adding a new axis at the specified index
    ///
    /// # Arguments
    ///
    /// * `axis` - The index at which to add the new axis.
    ///
    /// # Returns
    ///
    /// A new NDArray with expanded dimensions.
    #[allow(dead_code)]
    pub fn expand_dims(&self, axis: usize) -> Self {
        self.new_axis(axis)
    }

    /// Returns a boolean array indicating whether each element satisfies the condition
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold value to compare each element against.
    ///
    /// # Returns
    ///
    /// A vector of boolean values indicating whether each element is greater than the threshold.
    #[allow(dead_code)]
    pub fn greater_than(&self, threshold: f64) -> Vec<bool> {
        self.data.iter().map(|&x| x > threshold).collect()
    }

    /// Returns a new array containing only the elements that satisfy the condition
    ///
    /// # Arguments
    ///
    /// * `condition` - A closure that takes an f64 and returns a boolean.
    ///
    /// # Returns
    ///
    /// A new NDArray containing only the elements that satisfy the condition.
    #[allow(dead_code)]
    pub fn filter(&self, condition: impl Fn(&f64) -> bool) -> Self {
        let data: Vec<f64> = self.data.iter().cloned().filter(condition).collect();
        Self::from_vec(data)
    }


    /// Returns the data type of the elements in the array
    ///
    /// # Returns
    ///
    /// A string representing the data type of the elements.
    #[allow(dead_code)]
    pub fn dtype(&self) -> &'static str {
        "f64" // Since we're using f64 for all elements
    }

    /// Returns the total number of elements in the array
    ///
    /// # Returns
    ///
    /// The total number of elements in the array.
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the index of the minimum value in the array
    ///
    /// # Returns
    ///
    /// The index of the minimum value.
    #[allow(dead_code)]
    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap()
    }

    /// Returns a slice of the array from start to end (exclusive)
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the slice.
    /// * `end` - The ending index of the slice (exclusive).
    ///
    /// # Returns
    ///
    /// A new NDArray containing the specified slice.
    #[allow(dead_code)]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        println!("Slicing array:");
        println!("  Original shape: {:?}", self.shape);
        println!("  Start: {}, End: {}", start, end);

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        if self.ndim() == 2 {
            let cols = self.shape[1];
            let start_idx = start * cols;
            let end_idx = end * cols;
            println!("  2D array: keeping columns, new shape will be: {:?}", new_shape);
            
            let sliced_data = self.data[start_idx..end_idx].to_vec();
            println!("  Sliced data length: {}", sliced_data.len());
            
            NDArray::new(sliced_data, new_shape)
        } else {
            println!("  1D array: simple slice");
            NDArray::new(self.data[start..end].to_vec(), new_shape)
        }
    }

    /// Converts an NDArray of labels into a one-hot encoded NDArray
    ///
    /// # Arguments
    ///
    /// * `labels` - An NDArray containing numerical labels
    ///
    /// # Returns
    ///
    /// A new NDArray with one-hot encoded labels where each row corresponds to one label
    ///
    /// # Panics
    ///
    /// Panics if the input contains non-integer values
    pub fn one_hot_encode(labels: &NDArray) -> Self {
        // Verify that all values are integers
        for &value in labels.data() {
            // Check if the value is effectively an integer
            if value.fract() != 0.0 {
                panic!("All values must be integers for one-hot encoding");
            }
        }

        // Convert values to integers and find unique classes
        let labels_int: Vec<i32> = labels.data()
            .iter()
            .map(|&x| x as i32)
            .collect();

        // Find min and max to determine the range of classes
        let min_label = labels_int.iter().min().unwrap();
        let max_label = labels_int.iter().max().unwrap();
        let num_classes = (max_label - min_label + 1) as usize;
        
        let mut data = vec![0.0; labels_int.len() * num_classes];
        
        // Shift indices by min_label to handle negative values
        for (i, &label) in labels_int.iter().enumerate() {
            let shifted_label = (label - min_label) as usize;
            data[i * num_classes + shifted_label] = 1.0;
        }
        
        NDArray::new(data, vec![labels_int.len(), num_classes])
    }

    /// Transposes a 2D array (matrix)
    ///
    /// # Returns
    ///
    /// A new NDArray with transposed dimensions.
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "Transpose is only defined for 2D arrays");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        NDArray::new(transposed_data, vec![cols, rows])
    }

    /// Performs matrix multiplication (dot product) between two 2D arrays
    ///
    /// # Arguments
    ///
    /// * `other` - The other NDArray to multiply with.
    ///
    /// # Returns
    ///
    /// A new NDArray resulting from the matrix multiplication.
    pub fn dot(&self, other: &NDArray) -> Self {
        assert_eq!(self.ndim(), 2, "Dot product is only defined for 2D arrays");
        assert_eq!(other.ndim(), 2, "Dot product is only defined for 2D arrays");
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match for dot product");

        let rows = self.shape[0];
        let cols = other.shape[1];
        let mut result_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..self.shape[1] {
                    sum += self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                result_data[i * cols + j] = sum;
            }
        }

        NDArray::new(result_data, vec![rows, cols])
    }

    /// Performs element-wise multiplication between two arrays
    ///
    /// # Arguments
    ///
    /// * `other` - The other NDArray to multiply with.
    ///
    /// # Returns
    ///
    /// A new NDArray resulting from the element-wise multiplication.
    pub fn multiply(&self, other: &NDArray) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise multiplication");

        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Subtracts a scalar from each element in the array
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to subtract.
    ///
    /// # Returns
    ///
    /// A new NDArray with the scalar subtracted from each element.
    pub fn scalar_sub(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x - scalar).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Multiplies each element in the array by a scalar
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply.
    ///
    /// # Returns
    ///
    /// A new NDArray with each element multiplied by the scalar.
    pub fn multiply_scalar(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Clips the values in the array to a specified range
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value to clip to.
    /// * `max` - The maximum value to clip to.
    ///
    /// # Returns
    ///
    /// A new NDArray with values clipped to the specified range.
    pub fn clip(&self, min: f64, max: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.clamp(min, max)).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Performs element-wise division between two arrays
    ///
    /// # Arguments
    ///
    /// * `other` - The other NDArray to divide by.
    ///
    /// # Returns
    ///
    /// A new NDArray resulting from the element-wise division.
    pub fn divide(&self, other: &NDArray) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise division");

        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a / b).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Divides each element in the array by a scalar
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to divide by.
    ///
    /// # Returns
    ///
    /// A new NDArray with each element divided by the scalar.
    pub fn divide_scalar(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x / scalar).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Sums the elements of the array along a specified axis
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to sum the elements.
    ///
    /// # Returns
    ///
    /// A new NDArray with the summed elements along the specified axis.
    pub fn sum_axis(&self, axis: usize) -> Self {
        assert_eq!(self.ndim(), 2, "sum_axis is only defined for 2D arrays");
        let (rows, cols) = (self.shape[0], self.shape[1]);

        match axis {
            0 => {
                // Sum along columns
                let mut result_data = vec![0.0; cols];
                for j in 0..cols {
                    for i in 0..rows {
                        result_data[j] += self.data[i * cols + j];
                    }
                }
                NDArray::new(result_data, vec![1, cols])
            }
            1 => {
                // Sum along rows
                let mut result_data = vec![0.0; rows];
                for i in 0..rows {
                    for j in 0..cols {
                        result_data[i] += self.data[i * cols + j];
                    }
                }
                NDArray::new(result_data, vec![rows, 1])
            }
            _ => panic!("Invalid axis: {}", axis),
        }
    }

    /// Performs element-wise subtraction between two arrays
    ///
    /// # Arguments
    ///
    /// * `other` - The other NDArray to subtract.
    ///
    /// # Returns
    ///
    /// A new NDArray resulting from the element-wise subtraction.
    pub fn subtract(&self, other: &NDArray) -> Self {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise subtraction");

        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Adds a scalar to each element in the array
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to add.
    ///
    /// # Returns
    ///
    /// A new NDArray with the scalar added to each element.
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Calculates the natural logarithm of each element in the array
    ///
    /// # Returns
    ///
    /// A new NDArray with the natural logarithm of each element.
    pub fn log(&self) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x.ln()).collect();
        NDArray::new(data, self.shape.clone())
    }

    /// Sums all elements in the array
    ///
    /// # Returns
    ///
    /// The sum of all elements as an f64.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn pad_to_size(&self, target_size: usize) -> Self {
        if self.shape[0] >= target_size {
            return self.clone();
        }

        let mut new_shape = self.shape.clone();
        new_shape[0] = target_size;
        let total_size: usize = new_shape.iter().product();
        
        // Create new data vector with zeros
        let mut new_data = vec![0.0; total_size];
        
        // Copy existing data
        let row_size = self.shape.iter().skip(1).product::<usize>();
        let existing_data_size = self.shape[0] * row_size;
        new_data[..existing_data_size].copy_from_slice(&self.data);
        
        NDArray::new(new_data, new_shape)
    }
}

impl Add for NDArray {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise addition");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Add<&NDArray> for NDArray {
    type Output = Self;

    fn add(self, other: &NDArray) -> Self::Output {
        println!("Adding arrays:");
        println!("  Left shape: {:?}", self.shape);
        println!("  Right shape: {:?}", other.shape);

        // Handle broadcasting for shapes like [N, M] + [1, M]
        if self.shape.len() == other.shape.len() && 
           other.shape[0] == 1 && 
           self.shape[1] == other.shape[1] {
            
            println!("  Performing broadcasting addition");
            let mut result_data = Vec::with_capacity(self.data.len());
            let cols = other.shape[1];
            
            // Add the broadcasted row to each row of self
            for i in 0..self.shape[0] {
                for j in 0..cols {
                    result_data.push(self.data[i * cols + j] + other.data[j]);
                }
            }
            
            let result = NDArray::new(result_data, self.shape.clone());
            println!("  Result shape: {:?}", result.shape);
            return result;
        }
        
        // Regular element-wise addition for matching shapes
        if self.shape != other.shape {
            panic!("Shapes must match for element-wise addition\n  left: {:?}\n right: {:?}", 
                   self.shape, other.shape);
        }

        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        NDArray::new(data, self.shape.clone())
    }
}

impl Sub for NDArray {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.shape, other.shape, "Shapes must match for element-wise subtraction");
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Mul<f64> for NDArray {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        let data = self.data.iter().map(|a| a * scalar).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Add<f64> for NDArray {
    type Output = Self;

    fn add(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Add<f64> for &NDArray {
    type Output = NDArray;

    fn add(self, scalar: f64) -> Self::Output {
        let data: Vec<f64> = self.data.iter().map(|&x| x + scalar).collect();
        NDArray::new(data, self.shape.clone())
    }
}

impl Mul<&NDArray> for f64 {
    type Output = NDArray;

    fn mul(self, rhs: &NDArray) -> NDArray {
        rhs.multiply_scalar(self)
    }
}

// Add std::fmt::Display implementation for convenient printing
impl std::fmt::Display for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ndarray() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let array = NDArray::new(data.clone(), shape.clone());
        assert_eq!(array.data(), &data);
        assert_eq!(array.shape(), &shape);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let array = NDArray::from_vec(data.clone());
        assert_eq!(array.data(), &data);
        assert_eq!(array.shape(), &[3]);
    }

    #[test]
    fn test_arange() {
        let array = NDArray::arange(0.0, 5.0, 1.0);
        assert_eq!(array.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_element_wise_addition() {
        let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        let sum = arr1.clone() + arr2;
        assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scalar_multiplication() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let scaled = arr.clone() * 2.0;
        assert_eq!(scaled.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_reshape() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = arr.reshape(vec![2, 3]);
        assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let arr1 = NDArray::from_vec(vec![5.0, 7.0, 9.0]);
        let arr2 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let diff = arr1 - arr2;
        assert_eq!(diff.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_scalar_addition() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr + 1.0;
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_combined_operations() {
        let X = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let theta_1 = 2.0;
        let theta_0 = 1.0;
        let predictions = X.clone() * theta_1 + theta_0;
        assert_eq!(predictions.data(), &[3.0, 5.0, 7.0]);
    }


    #[test]
    fn test_one_hot_encode() {
        let labels = NDArray::from_vec(vec![0.0, 1.0, 2.0, 1.0, 0.0]);
        let one_hot = NDArray::one_hot_encode(&labels);
        
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0
        ];
        
        assert_eq!(one_hot.shape(), &[5, 3]);
        assert_eq!(one_hot.data(), &expected);
    }

    #[test]
    fn test_one_hot_encode_negative() {
        let labels = NDArray::from_vec(vec![-1.0, 0.0, 1.0, 0.0, -1.0]);
        let one_hot = NDArray::one_hot_encode(&labels);
        
        let expected = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0
        ];
        
        assert_eq!(one_hot.shape(), &[5, 3]);
        assert_eq!(one_hot.data(), &expected);
    }

    #[test]
    #[should_panic(expected = "All values must be integers")]
    fn test_one_hot_encode_non_integer() {
        let labels = NDArray::from_vec(vec![0.0, 1.5, 2.0]);
        NDArray::one_hot_encode(&labels);
    }

    #[test]
    fn test_transpose() {
        let arr = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let transposed = arr.transpose();
        assert_eq!(transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transposed.shape(), &[3, 2]);
    }

    #[test]
    fn test_dot() {
        let arr1 = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let arr2 = NDArray::from_matrix(vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ]);
        let dot = arr1.dot(&arr2);
        assert_eq!(dot.data(), &[58.0, 64.0, 139.0, 154.0]); // Adjust expected values based on the dot product calculation
    }

    #[test]
    fn test_multiply() {
        let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        let multiply = arr1.multiply(&arr2);
        assert_eq!(multiply.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_scalar_sub() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.scalar_sub(1.0);
        assert_eq!(result.data(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_multiply_scalar() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.multiply_scalar(2.0);
        assert_eq!(result.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_map() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.map(|x| x * 2.0);
        assert_eq!(result.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_clip() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.clip(1.0, 2.0);
        assert_eq!(result.data(), &[1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_divide() {
        let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        let divide = arr1.divide(&arr2);
        assert_eq!(divide.data(), &[0.25, 0.4, 0.5]);
    }

    #[test]
    fn test_divide_scalar() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.divide_scalar(2.0);
        assert_eq!(result.data(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_sum_axis() {
        let arr = NDArray::from_matrix(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let result = arr.sum_axis(0);
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]); // Sum along columns
        assert_eq!(result.shape(), &[1, 3]); // Shape should be [1, 3]

        let result = arr.sum_axis(1);
        assert_eq!(result.data(), &[6.0, 15.0]); // Sum along rows
        assert_eq!(result.shape(), &[2, 1]); // Shape should be [2, 1]
    }

    #[test]
    fn test_subtract() {
        let arr1 = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let arr2 = NDArray::from_vec(vec![4.0, 5.0, 6.0]);
        let subtract = arr1.subtract(&arr2);
        assert_eq!(subtract.data(), &[-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_add_scalar() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.add_scalar(1.0);
        assert_eq!(result.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros() {
        let shape = vec![2, 3];
        let zeros = NDArray::zeros(shape);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.log();
        assert_eq!(result.data(), &[0.0, 0.6931471805599453, 1.0986122886681098]);
    }

    #[test]
    fn test_sum() {
        let arr = NDArray::from_vec(vec![1.0, 2.0, 3.0]);
        let result = arr.sum();
        assert_eq!(result, 6.0);
    }
} 
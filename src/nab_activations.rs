use crate::nab_array::NDArray;

pub trait Activation {
    fn forward(&self, input: &NDArray) -> NDArray;
    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray;
}

pub struct ReLU;
pub struct Softmax;
pub struct Sigmoid;

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Softmax
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }
}

impl Activation for ReLU {
    fn forward(&self, input: &NDArray) -> NDArray {
        let output = input.relu();
        println!("ReLU activation range: [{}, {}]",
            output.data().iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            output.data().iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        output
    }

    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray {
        NDArray::activation_nabla(gradient, output, crate::nabla::ActivationType::ReLU)
    }
}

impl Activation for Softmax {
    fn forward(&self, input: &NDArray) -> NDArray {
        let output = input.softmax();
        println!("Softmax activation range: [{}, {}]",
            output.data().iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            output.data().iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        output
    }
    
    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray {
        NDArray::activation_nabla(gradient, output, crate::nabla::ActivationType::Softmax)
    }
}

impl Activation for Sigmoid {
    fn forward(&self, input: &NDArray) -> NDArray {
        input.sigmoid()
    }
    
    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray {
        NDArray::activation_nabla(gradient, output, crate::nabla::ActivationType::Sigmoid)
    }
} 
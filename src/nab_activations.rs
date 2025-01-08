use crate::nab_array::NDArray;

pub trait Activation {
    fn forward(&self, input: &NDArray) -> NDArray;
    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray;
}

pub struct Sigmoid;
pub struct Softmax;

impl Activation for Sigmoid {
    fn forward(&self, input: &NDArray) -> NDArray {
        input.sigmoid()
    }

    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray {
        NDArray::activation_nabla(gradient, output, crate::nabla::ActivationType::Sigmoid)
    }
}

impl Activation for Softmax {
    fn forward(&self, input: &NDArray) -> NDArray {
        input.softmax()
    }
    
    fn backward(&self, gradient: &NDArray, output: &NDArray) -> NDArray {
        NDArray::activation_nabla(gradient, output, crate::nabla::ActivationType::Softmax)
    }
} 
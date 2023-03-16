use std::{sync::{RwLock, Arc}, rc::Rc, any::{Any, TypeId}, path::Path};

use coaster::{IBackend, SharedTensor, Backend, Cuda, frameworks::{cuda::get_cuda_backend, native::{Cpu, get_native_backend}}, Native};
use juice::{layers::{Convolution, ConvolutionConfig, LinearConfig, SequentialConfig, DropoutConfig, PoolingConfig, PoolingMode}, layer::{Layer, LayerConfig, LayerType}, solver::{Solver, ISolver, SolverConfig}, solvers::Momentum};

type NetData<T> = Arc<RwLock<SharedTensor<T>>>;

pub(crate) fn to_netdata<T: Clone>(t: &[T]) -> NetData<T> {
    let b = get_native_backend();
    let dev = b.device();
    let mut share = SharedTensor::new(&[t.len()]);
    share
        .write_only(dev)
        .unwrap()
        .as_mut_slice::<T>()
        .iter_mut()
        .enumerate()
        .for_each(|(i, p)| *p = t[i].clone());
    Arc::new(RwLock::new(share))
}

pub(crate) fn from_netdata<T: Clone>(t: &NetData<T>) -> Vec<T> {
    let b = get_native_backend();
    let dev = b.device();
    t.read().unwrap().read(dev).unwrap().as_slice::<T>().to_owned()
}

pub(crate) struct NetCrit<B: IBackend> {
    data: Layer<B>,
}

fn net_conf_critic() -> SequentialConfig {
    let mut conf = SequentialConfig::default();
        conf.add_input("image", &[1,3,640,400]);
        conf.add_layer(LayerConfig::new("conv1", ConvolutionConfig{
            num_output: 16,
            filter_shape: vec![5],
            stride: vec![1],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("conv1n", LayerType::ReLU));
        //conf.add_layer(LayerConfig::new("conv1d", DropoutConfig{
        //    probability: 0.2,
        //    seed: 87435987023,
        //}));
        conf.add_layer(LayerConfig::new("conv1p", PoolingConfig{
            mode: PoolingMode::Max,
            filter_shape: vec![2],
            stride: vec![2],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("conv2", ConvolutionConfig{
            num_output: 64,
            filter_shape: vec![5],
            stride: vec![1],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("conv2n", LayerType::ReLU));
        //conf.add_layer(LayerConfig::new("conv2d", DropoutConfig{
        //    probability: 0.2,
        //    seed: 87435985023,
        //}));
        conf.add_layer(LayerConfig::new("conv2p", PoolingConfig{
            mode: PoolingMode::Max,
            filter_shape: vec![2],
            stride: vec![2],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("conv3", ConvolutionConfig{
            num_output: 10,
            filter_shape: vec![5],
            stride: vec![1],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("conv3n", LayerType::ReLU));
        //conf.add_layer(LayerConfig::new("conv3d", DropoutConfig{
        //    probability: 0.2,
        //    seed: 87435023,
        //}));
        conf.add_layer(LayerConfig::new("conv3p", PoolingConfig{
            mode: PoolingMode::Max,
            filter_shape: vec![2],
            stride: vec![2],
            padding: vec![1],
        }));
        conf.add_layer(LayerConfig::new("lin1", LinearConfig{
            output_size: 1000,
        }));
        conf.add_layer(LayerConfig::new("lin1n", LayerType::ReLU));
        //conf.add_layer(LayerConfig::new("lin1d", DropoutConfig{
        //    probability: 0.5,
        //    seed: 8743523,
        //}));
        conf.add_layer(LayerConfig::new("lin2", LinearConfig{
            output_size: 1000,
        }));
        conf.add_layer(LayerConfig::new("lin2n", LayerType::ReLU));
        //conf.add_layer(LayerConfig::new("lin2d", DropoutConfig{
        //    probability: 0.5,
        //    seed: 874323,
        //}));
        conf.add_layer(LayerConfig::new("lin3", LinearConfig{
            output_size: 1,
        }));
        conf.add_layer(LayerConfig::new("lin3n", LayerType::Sigmoid));
        conf
}

impl NetCrit<Backend<Cuda>> {
    pub(crate) fn new() -> Self {
        //let middle = 10*(640-12)*(400-12);
        let back = Rc::new(get_cuda_backend());
        
        Self {
            data: Layer::from_config(back, &LayerConfig::new("net", net_conf_critic())),
        }
    }

    pub(crate) fn train(&mut self, inp: &[NetData<f32>], err: &[NetData<f32>]) {
        //eprintln!("{:?}", inp[0].read().unwrap().desc());
        //eprintln!("{:?}", err[0].read().unwrap().desc());
        self.forward(inp);
        let _ = self.back(err);
        let back = Rc::new(get_cuda_backend());
        let mut solv = Momentum::<Backend<Cuda>>::new(back.clone());
        solv.init(&self.data);
        let solv_c = SolverConfig::default();
        solv.compute_update(&solv_c, &mut self.data, 0);
        self.data.update_weights(back.as_ref());
    }

    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        //let middle = 10*(640-12)*(400-12);
        let back = Rc::new(get_cuda_backend());
        
        if let Ok(l) = Layer::<Backend<Cuda>>::load(back.clone(), p) {
            Self{data: l}
        }
        else {
            Self {
                data: Layer::from_config(back, &LayerConfig::new("net", net_conf_critic())),
            }
        }
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.data.save(p)
    }
}

impl NetCrit<Backend<Native>> {
    pub(crate) fn new() -> Self {
        //let middle = 10*(640-12)*(400-12);
        let back = Rc::new(get_native_backend());
        Self {
            data: Layer::from_config(back, &LayerConfig::new("net", net_conf_critic())),
        }
    }
}

impl<B: IBackend> NetCrit<B> {
    pub(crate) fn forward(&mut self, x: &[NetData<f32>]) -> Vec<NetData<f32>>
    {
        self.data.forward(x)
    }

    fn back(&mut self, error_grad: &[NetData<f32>]) -> Vec<NetData<f32>> {
        self.data.backward(error_grad)
    }
}
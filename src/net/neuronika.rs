use std::{path::Path, fs, borrow::Borrow};

use ndarray::{Dimension, ArrayD, IxDyn, Ix2, Ix4, Data, Dim, Ix1};
use neuronika::{nn::{Conv2d, Linear}, from_ndarray, VarDiff, Replicative, Var, MaxPool, optim::{Optimizer, Adam, ElasticNet, L2}};

type NetData<T> = ArrayD<T>;

pub(crate) fn to_netdata<T: Clone>(t: &[T], d: impl Dimension) -> NetData<T> {
    ndarray::Array::from_shape_vec(d, t.to_vec()).unwrap().into_dyn()
}

pub(crate) fn from_netdata<T: Clone>(t: &NetData<T>) -> Vec<T> {
    t.as_slice().unwrap().to_vec()
}

pub(crate) struct NetCrit {
    layer1: Conv2d<Replicative>,
    layer2: Conv2d<Replicative>,
    layer3: Conv2d<Replicative>,
    layer4: Linear,
    layer5: Linear,
    layer6: Linear,
    opt: Optimizer<Adam<L2>>,
}

impl NetCrit {
    pub(crate) fn new() -> Self {
        
        let opt = Adam::new(0.1, 0.9, 0.999, L2::new(1.0), 1e-8);
        let layer1 = Conv2d::new(3, 16, (5,5), (1,1), Replicative, (1,1), (1,1));
        opt.register(layer1.weight.clone());
        opt.register(layer1.bias.clone());
        let layer2 = Conv2d::new(16, 64, (5,5), (1,1), Replicative, (1,1), (1,1));
        opt.register(layer2.weight.clone());
        opt.register(layer2.bias.clone());
        let layer3 = Conv2d::new(64, 10, (5,5), (1,1), Replicative, (1,1), (1,1));
        opt.register(layer3.weight.clone());
        opt.register(layer3.bias.clone());
        let layer4 = Linear::new(34966, 1000);
        opt.register(layer4.weight.clone());
        opt.register(layer4.bias.clone());
        let layer5 = Linear::new(1000, 1000);
        opt.register(layer5.weight.clone());
        opt.register(layer5.bias.clone());
        let layer6 = Linear::new(1000, 1);
        opt.register(layer6.weight.clone());
        opt.register(layer6.bias.clone());

        Self {
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            opt,
        }
    }

    pub(crate) fn train(&mut self, img: &NetData<f32>, keys: &NetData<f32>, out: &NetData<f32>) {
        let elems = img.dim().size()/768000;
        let inp = from_ndarray(img.to_shape((elems, 3, 640, 400)).unwrap().to_owned());
        let keys = from_ndarray(keys.to_shape((elems, 6)).unwrap().to_owned());
        let out = from_ndarray(out.to_shape((elems, 1)).unwrap().to_owned());
        let l_out = self.forward_raw(inp, keys);
        let diff = l_out.mse(out, neuronika::Reduction::Sum);
        diff.forward();
        diff.backward(-1.0);
        self.opt.step();
    }

    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        
        if let Ok(f) = fs::read(p) {
        if let Ok((l1w, l1b, l2w, l2b, l3w, l3b, l4, l5, l6)) = bincode::deserialize(&f) {
            let mut s = Self::new();
            s.layer1.weight = l1w;
            s.layer1.bias = l1b;
            s.layer2.weight = l2w;
            s.layer2.bias = l2b;
            s.layer3.weight = l3w;
            s.layer3.bias = l3b;
            s.layer4 = l4;
            s.layer5 = l5;
            s.layer6 = l6;
            s
        }
        else {
            Self::new()
        }
    }
    else {
        Self::new()
    }
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        let data = bincode::serialize(&(&self.layer1.weight, &self.layer1.bias,
            &self.layer2.weight, &self.layer2.bias,
            &self.layer3.weight, &self.layer3.bias,
            &self.layer4, &self.layer5, &self.layer6
        )).unwrap();
        fs::write(p, data)?;
        Ok(())
    }
}

fn reshape<S: Dimension>(l: VarDiff<impl Dimension>, shape: S) -> VarDiff<S> {
    let inp_size = l.data().shape().iter().cloned().reduce(|x,y| x*y).unwrap();
    if inp_size != shape.size() {
        eprintln!("Differing sizes: input: {:?}({}), requested:{:?}({})", l.data().shape(), inp_size, shape, shape.size());
    }
    let grad_size = l.grad().shape().iter().cloned().reduce(|x,y| x*y).unwrap();
    if grad_size != shape.size() {
        eprintln!("Differing sizes: grad: {:?}({}), requested:{:?}({})", l.grad().shape(), grad_size, shape, shape.size());
    }
    let w = l.data().to_shape(shape.clone()).unwrap().to_owned();
    let g = l.grad().to_shape(shape.clone()).unwrap().to_owned();
    let r = from_ndarray::<S>(ndarray::Array::from_shape_vec(shape.clone(), vec![0.0; shape.size()]).unwrap()).requires_grad();
    r.data_mut().assign(&w);
    *(r.grad_mut())=g;
    r
}

impl NetCrit {
    pub(crate) fn forward(&mut self, img: &NetData<f32>, keys: &NetData<f32>) -> NetData<f32>
    {
        let elems = img.dim().size()/768000;
        let inp = from_ndarray(img.to_shape((elems, 3, 640, 400)).unwrap().to_owned());
        let keys = from_ndarray(keys.to_shape((elems, 6)).unwrap().to_owned());
        let out = self.forward_raw(inp, keys);
        let d = out.data();
        d.to_shape(IxDyn(&[1])).unwrap().to_owned()
    }

    fn forward_raw(&mut self, img: Var<Ix4>, keys: Var<Ix2>) -> VarDiff<Ix2> {
        let l1 = self.layer1.forward(img).tanh().pool(Ix2(2,2), MaxPool);
        let l2 = self.layer2.forward(l1).tanh().pool(Ix2(2,2), MaxPool);
        let l3 = self.layer3.forward(l2).tanh().pool(Ix2(2,2), MaxPool);
        l3.forward();
        let l3f = reshape(l3, Ix2(1,34960));
        let l3f = l3f.cat(&[keys.requires_grad()], 1);
        //eprintln!("{:?} {:?}", l3f.data().shape(), l3f.grad().shape());
        let l4 = self.layer4.forward(l3f).tanh();
        let l5 = self.layer5.forward(l4).tanh();
        let l6 = self.layer6.forward(l5);
        l6.forward();
        l6
    }
}
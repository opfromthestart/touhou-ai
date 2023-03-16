use std::path::Path;

use dfdx::{prelude::{LoadFromNpz, SaveToNpz, ModuleMut, DeviceBuildExt, Conv2D, Bias2D, BuildOnDevice, Linear, ResetParams, ZeroGrads, mse_loss, MaxPool2D, Tanh, TensorCollection, Module, ModuleVisitor, RecursiveWalker, TensorVisitor, ViewTensorRef, ViewTensorMut}, tensor::{Tensor4D, Tensor2D, TensorFromVec, Cpu, Tensor, Tape, Gradients, Trace}, shapes::{Const, Rank2, Shape}, tensor_ops::{ReshapeTo, TryConcat, PermuteTo, Backward}, optim::{Sgd, SgdConfig, Optimizer}};

type Device = Cpu;

type NetConv = (
    (
    Conv2D<3, 16, 5, 1, 0>,
    Bias2D<16>,
    Tanh,
    MaxPool2D<2,2,0>,
    ),
    (
    Conv2D<16, 64, 5, 1, 0>,
    Bias2D<64>,
    Tanh,
    MaxPool2D<2,2,0>,
    ),
    (
    Conv2D<64, 10, 5, 1, 0>,
    Bias2D<10>,
    Tanh,
    MaxPool2D<2,2,0>,
    ),
);
type NetLinCrit = (
    Linear<34966, 1000>,
    Tanh,
    Linear<1000, 1000>,
    Tanh,
    Linear<1000, 1>,
);

pub(crate) struct NetCrit {
    net: (<NetConv as BuildOnDevice<Device, f32>>::Built, <NetLinCrit as BuildOnDevice<Device, f32>>::Built),
}

impl NetCrit {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = Device::default();

        let mut net = (dev.build_module::<NetConv, f32>(), dev.build_module::<NetLinCrit, f32>());
        let loaded = net.load(p.as_ref()).is_ok();
        if !loaded {
            net.reset_params();
        }
        Self { net }
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        Ok(())
    }

    pub(crate) fn forward(&mut self, img: &[f32], keys: &[f32]) -> Vec<f32> {
        let dev = Device::default();
        let img = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));
        let keys = dev.tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<6>));

        let out = self.forward_raw(img, keys);

        out.as_vec()
    }

    fn forward_raw<T: Tape<f32, Device> + 'static + dfdx::tensor::Merge<T2>, T2: Tape<f32, Device> + 'static>(&mut self, img: Tensor4D<1,3,640,400, T>, keys: Tensor2D<1,6,T2>) -> Tensor2D<1,1, T> {
        let out_conv = self.net.0.forward_mut(img);
        let in_conv : Tensor<(Const::<34960>, Const::<1>), _, _, T> = out_conv.reshape();
        let in_lin = in_conv.concat(keys.reshape::<(Const::<6>, Const::<1>)>()).permute::<Rank2<1, 34966>,_>();
        let out = self.net.1.forward_mut(in_lin);
        out
    }

    pub(crate) fn train(&mut self, img: &[f32], keys: &[f32], out: &[f32]) {
        let dev = Device::default();

        let mut grads = self.net.alloc_grads();

        let mut sgd : Sgd<_,_,Device> = Sgd::new(&self.net, SgdConfig{
            lr: 0.1,
            momentum: Some(dfdx::optim::Momentum::Nesterov(0.9)),
            weight_decay: None,
        });

        let img = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));
        let keys = dev.tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<6>));

        let model_out = self.forward_raw(img.traced(grads), keys);
        let out = dev.tensor_from_vec(out.to_vec(), (Const::<1>, Const::<1>));

        let err = mse_loss(model_out, out);
        grads = err.backward();

        sgd.update(&mut self.net, &grads).unwrap();

        //self.net.zero_grads(&mut grads);
    }
}
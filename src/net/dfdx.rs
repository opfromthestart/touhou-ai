use std::{path::Path, io::Read};

use dfdx::{prelude::{LoadFromNpz, SaveToNpz, ModuleMut, DeviceBuildExt, Conv2D, BuildOnDevice, Linear, ResetParams, ZeroGrads, mse_loss, MaxPool2D, Module, Dropout, Upscale2D, Flatten2D, AvgPool2D, GeneralizedResidual, ConvTrans2D}, tensor::{TensorFromVec, Tensor, Gradients, Trace, Cuda, NoneTape, OwnedTape, AsArray}, shapes::{Const, Rank2, Rank4}, tensor_ops::{ReshapeTo, TryConcat, PermuteTo, Backward, Bilinear}, optim::{Optimizer, Adam, AdamConfig}};
use dfdx::prelude::Bias2D;

use dfdx::prelude::PReLU;
use dfdx::prelude::PReLU1D;

use crate::arena;

type NetDevice = Cuda;

type NetConvCRes = (
    AvgPool2D<4,4,0>,
    (
        Dropout,
        GeneralizedResidual<(
            Conv2D<3, 16, 3, 1, 0>,
            Bias2D<16>,
            PReLU1D<Const<16>>,
            MaxPool2D<2,2,0>,
        ), 
        (Conv2D<3, 16, 1, 3, 0>, Upscale2D<79, 49>)>,
    ),
    (
        Dropout,
        GeneralizedResidual<(
        Conv2D<16, 64, 3, 1, 0>,
        Bias2D<64>,
        PReLU1D<Const<64>>,
        MaxPool2D<2,2,0>,
    ), (Conv2D<16, 64, 1, 3, 0>, Upscale2D<38, 23>)>,
),
    (
        Dropout,
        GeneralizedResidual<(
        Conv2D<64, 10, 3, 1, 0>,
        Bias2D<10>,
        PReLU1D<Const<10>>,
        MaxPool2D<2,2,0>,
    ), (Conv2D<64, 10, 1, 3, 0>, Upscale2D<18, 10>)>,),
    Dropout,
);
type NetConvCN = (
    AvgPool2D<4,4,0>,
    (
        Dropout,
        (
            Conv2D<3, 64, 3, 1, 0>,
            Bias2D<64>,
            PReLU1D<Const<64>>,
            MaxPool2D<2,2,0>,
        ), 
    ),
    (
        Dropout,
        (
        Conv2D<64, 128, 3, 1, 0>,
        Bias2D<128>,
        PReLU1D<Const<128>>,
        MaxPool2D<2,2,0>,
    ),
),
    (
        Dropout,
        (
        Conv2D<128, 20, 3, 1, 0>,
        Bias2D<20>,
        PReLU1D<Const<20>>,
        MaxPool2D<2,2,0>,
        )
    ),
    Dropout,
);
type NetConvLin = (
    Flatten2D,
    Linear<3600, 128>,
    PReLU,
);
type NetConv = (
    NetConvCN,
    NetConvLin,
);
type NetConvTransLin = (Linear<128, 128>, PReLU, Linear<128, 3600>, PReLU);
type NetConvTransCRes = (
    (
        Dropout,
        GeneralizedResidual<(
            Upscale2D<40,25>,
        Conv2D<20, 128, 3, 2, 0>,
        Bias2D<128>,
        PReLU1D<Const<128>>,),
        (Upscale2D<38, 23>, Conv2D<20, 128, 1, 1, 0>)>,
    ),
    (
        Dropout,
        GeneralizedResidual<(
            Upscale2D<81,51>,
        Conv2D<64, 16, 3, 2, 0>,
        Bias2D<16>,
        PReLU1D<Const<16>>,),
        (Upscale2D<79, 49>, Conv2D<64, 16, 1, 1, 0>)>,
    ),
    (
        Dropout,
        GeneralizedResidual<(
            Upscale2D<162,102>,
        Conv2D<16, 3, 3, 2, 0>,
        Bias2D<3>,
        PReLU1D<Const<3>>,),
        (Upscale2D<160, 100>, Conv2D<16, 3, 1, 1, 0>)>,
    ),
    Upscale2D<640, 400, Bilinear>,
);
type NetConvTransCN = (
    (
        Dropout,
        (
        ConvTrans2D<20, 128, 3, 2, 0>,
        Bias2D<128>,
        PReLU1D<Const<128>>,),
    ),
    (
        Dropout,
        (
        ConvTrans2D<128, 64, 3, 2, 0>,
        Bias2D<64>,
        PReLU1D<Const<64>>,),
    ),
    (
        Dropout,
        (
        ConvTrans2D<64, 3, 3, 2, 0>,
        Bias2D<3>,
        PReLU1D<Const<3>>,),
    ),
    ConvTrans2D<3, 3, 4, 4, 0>,
    Upscale2D<640, 400, Bilinear>,
);
type NetConvTransC = NetConvTransCN;
type NetLinCrit = (
    (
    Linear<134, 128>,
    Dropout,
    PReLU,
    ),
    (
    Linear<128, 128>,
    Dropout,
    PReLU,
    ),
    Linear<128, 1>,
);

pub(crate) struct NetAutoEncode {
    net: <(NetConv, NetConvTransLin, NetConvTransC) as BuildOnDevice<NetDevice, f32>>::Built,
    optim: Adam<<(NetConv, NetConvTransLin, NetConvTransC) as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
}

pub(crate) struct NetCrit {
    net: <(NetConv, NetLinCrit) as BuildOnDevice<NetDevice, f32>>::Built,
    optim: Adam<<NetLinCrit as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
}

impl NetCrit {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<(NetConv, NetLinCrit), f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            net.reset_params();
        }
        net.0.0.1.0.p = 0.2;
        net.0.0.2.0.p = 0.2;
        net.0.0.3.0.p = 0.2;
        net.0.0.4.p = 0.2;
        net.1.0.1.p = 0.2;
        net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        Self {
        optim: Adam::new(&net.1, AdamConfig{
            lr: 0.001,
            betas: [0.9, 0.999],
            ..Default::default()
        }), net
        }
    }

    pub(crate) fn load_from_encode<P: AsRef<Path>>(p: P, enc_p: P) -> Self {
        let dev = NetDevice::default();

        let mut net_enc = dev.build_module::<(NetConv, NetConvTransLin, NetConvTransC), f32>();
        let res_enc = net_enc.load(enc_p.as_ref());
        let mut net = dev.build_module::<(NetConv, NetLinCrit), _>();
        let res_lin = net.load(p.as_ref());
        net.0 = net_enc.0;
        let loaded = res_enc.is_ok() && res_lin.is_ok();
        if !loaded {
            eprintln!("Made new, {res_enc:?} {res_lin:?}");
            net.reset_params();
        }
        net.0.0.1.0.p = 0.2;
        net.0.0.2.0.p = 0.2;
        net.0.0.3.0.p = 0.2;
        net.0.0.4.p = 0.2;
        net.1.0.1.p = 0.2;
        net.1.1.1.p = 0.2;
        Self {
        optim: Adam::new(&net.1, AdamConfig{
            lr: 0.001,
            betas: [0.9, 0.999],
            ..Default::default()
        }), net
        }
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        Ok(())
    }

    pub(crate) fn forward(&self, img: &[f32], keys: &[f32]) -> Vec<f32> {
        let dev = NetDevice::default();
        let img = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));
        let keys = dev.tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<6>));

        let out = self.forward_raw(img, keys);

        out.as_vec()
    }

    fn forward_raw<const BATCH: usize>(&self, img: Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, NoneTape>, keys: Tensor<Rank2<BATCH,6>, f32, NetDevice, NoneTape>) -> Tensor<Rank2<BATCH,1>, f32, NetDevice, NoneTape> {
        let out_conv = self.net.0.forward(img);
        let in_conv : Tensor<(Const::<128>, Const::<BATCH>), _, _, _> = out_conv.reshape();
        let in_lin = in_conv.concat(keys.reshape::<(Const::<6>, Const::<BATCH>)>()).permute::<Rank2<BATCH, 134>,_>();
        let out = self.net.1.forward(in_lin);
        out
    }

    fn forward_raw_mut<const BATCH: usize>(&mut self, img: Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, OwnedTape<f32, NetDevice>>, keys: Tensor<Rank2<BATCH,6>, f32, NetDevice, NoneTape>) -> Tensor<Rank2<BATCH,1>, f32, NetDevice, OwnedTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward_mut(img);
        let in_conv : Tensor<(Const::<BATCH>,Const::<128>), _, _, _> = out_conv.reshape();
        let in_conv : Tensor<(Const::<128>,Const::<BATCH>), _, _, _> = in_conv.permute().reshape();
        let keys : Tensor<(Const::<6>, Const::<BATCH>), _, _, _> = keys.permute().reshape();
        let in_lin = in_conv.concat(keys).permute::<Rank2<BATCH, 134>,_>();
        let out = self.net.1.forward_mut(in_lin);
        out
    }

    pub(crate) fn do_conv(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let dev = NetDevice::default();
        let img: Tensor<Rank4<1,3,640,400>, f32, NetDevice, NoneTape> = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));
        let out = self.net.0.0.forward(img);
        let arr = out.array();
        arr[0].map(|x| x.iter().cloned().flatten().collect()).to_vec()
    }

    pub(crate) fn train(&mut self, img: &[f32], keys: &[f32], out: &[f32]) -> f32 {

        let grads = self.net.alloc_grads();

        let (r, _) = self.train_grad::<1>(&[img], &[keys], &[out], grads);

        //self.net.zero_grads(&mut grads);

        r
    }

    pub(crate) fn train_batch<const BATCH: usize>(&mut self, img: &[&[f32]], keys: &[&[f32]], out: &[&[f32]]) -> f32 {

        let grads = self.net.alloc_grads();

        let (r, _) = self.train_grad::<BATCH>(img, keys, out, grads);

        //self.net.zero_grads(&mut grads);

        r
    }

    pub(crate) fn train_grad<const SS: usize>(&mut self, img: &[&[f32]], keys: &[&[f32]], out: &[&[f32]], mut grad: Gradients<f32, NetDevice>) -> (f32, Gradients<f32, NetDevice>) {

        let dev = NetDevice::default();

        let img = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>,Const::<3>,Const::<640>,Const::<400>));
        let keys = dev.tensor_from_vec(keys.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>, Const::<6>));

        let model_out = self.forward_raw_mut(img.traced(grad), keys);
        let out = dev.tensor_from_vec(out.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>, Const::<1>));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);
        
        grad = err.backward();

        //dbg!(&grads);

        self.optim.update(&mut self.net.1, &grad).unwrap();

        self.net.zero_grads(&mut grad);

        (r, grad)
    }

    pub(crate) fn acc_grad<const SS: usize>(&mut self, img: &[&[f32]], keys: &[&[f32]], out: &[&[f32]], mut grad: Gradients<f32, NetDevice>) -> (f32, Gradients<f32, NetDevice>) {

        let dev = NetDevice::default();

        let img = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>,Const::<3>,Const::<640>,Const::<400>));
        let keys = dev.tensor_from_vec(keys.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>, Const::<6>));

        let model_out = self.forward_raw_mut(img.traced(grad), keys);
        let out = dev.tensor_from_vec(out.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>, Const::<1>));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);
        
        grad = err.backward();

        (r, grad)
    }

    pub(crate) fn backward(&mut self, mut grad: Gradients<f32, NetDevice>) -> Gradients<f32, NetDevice>{
        self.optim.update(&mut self.net.1, &grad).unwrap();

        self.net.zero_grads(&mut grad);

        grad
    }

    pub(crate) fn grad(&self) -> Gradients<f32, NetDevice> {
        self.net.alloc_grads()
    }
}

impl NetAutoEncode {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<(NetConv, NetConvTransLin, NetConvTransC), f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            net.reset_params();
        }
        net.0.0.1.0.p = 0.1;
        net.0.0.2.0.p = 0.1;
        net.0.0.3.0.p = 0.1;
        net.0.0.4.p = 0.1;
        net.2.0.0.p = 0.1;
        net.2.1.0.p = 0.1;
        net.2.2.0.p = 0.1;
        // net.1.3.p = 0.2;
        Self {optim: Adam::new(&net, AdamConfig{
            lr: 0.0001,
            betas: [0.995, 0.99995],
            ..Default::default()
        }), net}
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        Ok(())
    }

    pub(crate) fn get_latent(&self, img: &[f32]) -> Vec<f32> {
        let dev = NetDevice::default();
        let img = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));

        let out = self.net.0.forward(img);

        out.as_vec()
    }

    pub(crate) fn forward(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let dev = NetDevice::default();
        let img = dev.tensor_from_vec(img.to_vec(), (Const::<1>,Const::<3>,Const::<640>,Const::<400>));

        let out = self.forward_raw::<1>(img).array();

        out[0].iter().map(|x| x.iter().cloned().flatten().collect()).collect()
    }

    fn forward_raw<const BATCH: usize>(&self, img: Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, NoneTape>) -> Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, NoneTape> {
        let out_conv = self.net.0.forward(img);
        let out_lin = self.net.1.forward(out_conv).reshape::<Rank4<BATCH, 20, 18, 10>>();
        let out = self.net.2.forward(out_lin);
        out
    }

    fn forward_raw_mut<const BATCH: usize>(&mut self, img: Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, OwnedTape<f32, NetDevice>>) -> Tensor<Rank4<BATCH,3,640,400>, f32, NetDevice, OwnedTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward_mut(img);
        let out_lin = self.net.1.forward_mut(out_conv).reshape::<Rank4<BATCH, 20, 18, 10>>();
        let out = self.net.2.forward_mut(out_lin);
        out
    }

    pub(crate) fn train(&mut self, img: &[f32]) -> f32 {

        let grads = self.net.alloc_grads();

        let (r, _) = self.train_grad::<1>(&[img], grads);

        //self.net.zero_grads(&mut grads);

        r
    }

    pub(crate) fn train_batch<const BATCH: usize>(&mut self, img: &[&[f32]]) -> f32 {

        let grads = self.net.alloc_grads();

        let (r, _) = self.train_grad::<BATCH>(img, grads);

        //self.net.zero_grads(&mut grads);

        r
    }

    pub(crate) fn train_grad<const SS: usize>(&mut self, img: &[&[f32]], mut grad: Gradients<f32, NetDevice>) -> (f32, Gradients<f32, NetDevice>) {

        let dev = NetDevice::default();

        let out = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>, Const::<3>, Const::<640>, Const::<400>));
        let img = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>,Const::<3>,Const::<640>,Const::<400>));

        let model_out = self.forward_raw_mut(img.traced(grad));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);
        
        grad = err.backward();

        //dbg!(&grads);

        self.optim.update(&mut self.net, &grad).unwrap();

        self.net.zero_grads(&mut grad);

        (r, grad)
    }

    pub(crate) fn acc_grad<const SS: usize>(&mut self, img: &[&[f32]], mut grad: Gradients<f32, NetDevice>) -> (f32, Gradients<f32, NetDevice>) {

        let dev = NetDevice::default();

        let out = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>,Const::<3>,Const::<640>,Const::<400>));
        let img = dev.tensor_from_vec(img.iter().map(|&x| x).flatten().cloned().collect(), (Const::<SS>,Const::<3>,Const::<640>,Const::<400>));

        let model_out = self.forward_raw_mut(img.traced(grad));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);
        
        grad = err.backward();

        (r, grad)
    }

    pub(crate) fn backward(&mut self, mut grad: Gradients<f32, NetDevice>) -> Gradients<f32, NetDevice>{
        self.optim.update(&mut self.net, &grad).unwrap();

        self.net.zero_grads(&mut grad);

        grad
    }

    pub(crate) fn grad(&self) -> Gradients<f32, NetDevice> {
        self.net.alloc_grads()
    }
}

pub(crate) fn to_pixels(img: &arena::Image) -> Vec<f32> {
    let frames = arena::split_channel(&img);
    frames.into_iter().map(|x| x.bytes().map(|x| (x.unwrap() as f32)/128. - 1.0).collect::<Vec<_>>()).flatten().collect()
}

pub(crate) fn from_pixels(img: &[f32], shape: (u32, u32)) -> arena::GrayImage {
    let img_data: Vec<_> = img.iter().map(|x| ((x+1.0)*128.) as u8).collect();
    arena::GrayImage::from_vec(shape.0, shape.1, img_data).unwrap()
}

pub(crate) fn to_rb(img: &[f32], shape: (u32, u32), norm: bool) -> arena::Image {
    let max = img.iter().fold(0.0f32, |m, v| {
        if v.abs() > m.abs() {
            *v
        }
        else {
            m
        }
    });
    let max = if max == 0.0 || !norm {
        1.0
    } else {
        max
    };
    let img_data: Vec<_> = img.iter().cloned().flat_map(|x| {
        let x = x/max;
        if x<0. {
            [(-x*255.) as u8, 0, 0]
        }
        else if x>0. {
            [0,0,(x*255.) as u8]
        }
        else {
            [0,0,0]
        }
    }).collect();
    arena::Image::from_vec(shape.0, shape.1, img_data).unwrap()
}
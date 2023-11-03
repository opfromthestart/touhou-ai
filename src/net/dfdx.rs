use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::{io::Read, path::Path};

use dfdx::prelude::{
    Bias2D, LeakyReLU, NonMutableModule, RecursiveWalker, Softmax, TensorCollection, TensorVisitor,
    ViewTensorRef, ZeroSizedModule,
};
use dfdx::shapes::{Axes, Axes3, Axis, Dtype, Rank0, ReduceShape, Shape};
use dfdx::tensor::{PutTape, Storage, Tape};
use dfdx::tensor_ops::{
    huber_error, mul, BroadcastTo, Device, MaxTo, MeanTo, NearestNeighbor, WeightDecay,
};
use dfdx::tensor_ops::{SumTo, TryConcatAlong};
use dfdx::{
    optim::{Adam, AdamConfig, Optimizer},
    prelude::{
        mse_loss, AvgPool2D, BuildOnDevice, Conv2D, DeviceBuildExt, Flatten2D, Linear, LoadFromNpz,
        MaxPool2D, Module, ModuleMut, SaveToNpz, Upscale2D, Upscale2DBy, ZeroGrads,
    },
    shapes::{Const, Rank2, Rank4},
    tensor::{AsArray, Cuda, Gradients, NoneTape, OwnedTape, Tensor, TensorFromVec},
    tensor_ops::{Backward, Bilinear},
};

use dfdx::prelude::PReLU;
use dfdx::prelude::PReLU1D;
use rand::{thread_rng, Rng};

use crate::arena;

const LATENT_SIZE: usize = 2400;
const INPUT_SIZE: usize = 32;
const CRITIC_SIZE: usize = LATENT_SIZE + INPUT_SIZE;
pub(crate) const LR: f64 = 0.000001;
const MIDSIZE: usize = CRITIC_SIZE * 4;

pub(crate) type NetDevice = Cuda;
type Layer = LeakyReLU<f32>;
type NormLayer = Softmax;
pub(crate) type NetTape<E, D> = Arc<Mutex<OwnedTape<E, D>>>;

pub(crate) struct Normalize<A: Axes> {
    eps: f64,
    a: PhantomData<A>,
}

impl<S: Shape + ReduceShape<A>, E: Dtype, D: Device<E>, T: Tape<E, D>, A: Axes>
    Module<Tensor<S, E, D, T>> for Normalize<A>
{
    type Output = Tensor<S, E, D, T>;

    type Error = D::Err;

    fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, Self::Error> {
        Ok(input.normalize::<A>(self.eps))
    }
}
impl<A: Axes> NonMutableModule for Normalize<A> {}
impl<A: Axes> ZeroSizedModule for Normalize<A> {}
impl<A: Axes> Default for Normalize<A> {
    fn default() -> Self {
        Self {
            eps: 0.00001,
            a: PhantomData,
        }
    }
}

type NetConvCN = (
    AvgPool2D<2, 2, 0>,
    (
        (
            Conv2D<3, 20, 5, 1, 2>,
            Bias2D<20>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<20>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        (
            Conv2D<20, 20, 1, 1, 0>,
            Bias2D<20>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<20>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        MaxPool2D<2, 2, 0>,
    ),
    (
        (
            //     Dropout,
            Conv2D<20, 60, 3, 1, 1>,
            Bias2D<60>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        (
            //     Dropout,
            Conv2D<60, 60, 1, 1, 0>,
            Bias2D<60>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        MaxPool2D<2, 2, 0>,
    ),
    (
        (
            //     Dropout,
            Conv2D<60, 60, 3, 1, 1>,
            Bias2D<60>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        (
            //     Dropout,
            Conv2D<60, 60, 1, 1, 0>,
            Bias2D<60>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        MaxPool2D<2, 2, 0>,
    ),
    (
        (
            //     Dropout,
            Conv2D<60, 60, 3, 1, 1>,
            Bias2D<60>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        (
            //     Dropout,
            Conv2D<60, 10, 1, 1, 0>,
            Bias2D<10>,
            // BatchNorm2D<10>,
            // PReLU1D<Const<60>>,
            // LeakyReLU<f32>,
            Layer,
        ),
        MaxPool2D<2, 2, 0>,
    ),
    // (
    // //     Dropout,
    //     Conv2D<32, 32, 3, 1, 1>,
    //     Bias2D<32>,
    //     PReLU1D<Const<32>>,
    //     MaxPool2D<2, 2, 0>,
    // ),
    // (
    // //     // Dropout,
    //     Conv2D<32, 32, 3, 1, 1>,
    //     Bias2D<32>,
    //     PReLU1D<Const<32>>,
    //     MaxPool2D<2, 2, 1>,
    // ),
    // (
    // //     // Dropout,
    //     Conv2D<32, 32, 3, 1, 1>,
    //     Bias2D<32>,
    //     PReLU1D<Const<32>>,
    //     MaxPool2D<2, 2, 1>,
    // ),
    // Dropout,
    Normalize<Axes3<1, 2, 3>>,
);
type NetConv = NetConvCN;

type NetConvTransCUp = (
    // (
    //     (
    // //         // Dropout,
    //         Upscale2DBy<2, 2, NearestNeighbor>,
    //         Conv2D<128, 128, 3, 1, 2>,
    //         Bias2D<128>,
    //         PReLU1D<Const<128>>,
    //     ),
    //     (
    // //         // Dropout,
    //         Upscale2DBy<2, 2, NearestNeighbor>,
    //         Conv2D<128, 128, 3, 1, 2>,
    //         Bias2D<128>,
    //         PReLU1D<Const<128>>,
    //     ),
    // ),
    // (
    // //     Dropout,
    //     Upscale2DBy<2, 2, NearestNeighbor>,
    //     Conv2D<32, 32, 3, 1, 1>,
    //     Bias2D<32>,
    //     PReLU1D<Const<32>>,
    //     // (Conv2D<32, 32, 3, 1, 1>, Bias2D<32>, PReLU1D<Const<32>>),
    // ),
    (
        //     Dropout,
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<10, 60, 3, 1, 1>,
        Bias2D<60>,
        PReLU1D<Const<60>>,
        // (Conv2D<60, 60, 3, 1, 1>, Bias2D<60>, PReLU1D<Const<60>>),
    ),
    (
        //     Dropout,
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<60, 60, 3, 1, 1>,
        Bias2D<60>,
        PReLU1D<Const<60>>,
        // (Conv2D<60, 60, 3, 1, 1>, Bias2D<60>, PReLU1D<Const<60>>),
    ),
    (
        //     Dropout,
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<60, 20, 3, 1, 1>,
        Bias2D<20>,
        PReLU1D<Const<20>>,
        // (Conv2D<20, 20, 3, 1, 1>, Bias2D<20>, PReLU1D<Const<20>>),
    ),
    (
        //     Dropout,
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<20, 3, 5, 1, 2>,
        Bias2D<3>,
        PReLU1D<Const<3>>,
        //     Dropout,
    ),
    // Upscale2D<400, 640, Bilinear>,
    (
        Conv2D<3, 32, 5, 1, 2>,
        Bias2D<32>,
        PReLU1D<Const<32>>,
        Conv2D<32, 32, 5, 1, 2>,
        Bias2D<32>,
        PReLU1D<Const<32>>,
    ),
    (
        Upscale2D<400, 640, Bilinear>,
        // Upscale2DBy<4, 4, NearestNeighbor>,
        Conv2D<32, 3, 5, 1, 2>,
        Bias2D<3>,
        PReLU,
    ),
    // Upscale2DBy<4, 4, Bilinear>,
);
type NetConvTransC = NetConvTransCUp;
type NetConvTrans = NetConvTransC;
type NetLinCrit = (
    (Linear<CRITIC_SIZE, MIDSIZE>, Layer),
    // (Linear<MIDSIZE, MIDSIZE>, Layer),
    Linear<MIDSIZE, 1>,
    Layer,
);
type NetLinActor = (
    (Linear<LATENT_SIZE, MIDSIZE>, Layer),
    // (Linear<MIDSIZE, MIDSIZE>, Layer),
    (Linear<MIDSIZE, INPUT_SIZE>, NormLayer),
);
type NetLinQ = (
    (Linear<LATENT_SIZE, MIDSIZE>, Layer),
    // (Linear<MIDSIZE, MIDSIZE>, Layer),
    Linear<MIDSIZE, INPUT_SIZE>,
);

type NetAutoModel = (NetConv, NetConvTrans);
type NetCritModel = (NetConv, NetLinCrit);
type NetActorModel = (NetConv, NetLinActor);
type NetQModel = (NetConv, NetLinQ);
#[allow(dead_code)]
type NetCritLearn = NetCritModel;
type NetActorLearn = NetActorModel;

pub(crate) struct NetAutoEncode {
    dev: NetDevice,
    pub(crate) net: <NetAutoModel as BuildOnDevice<NetDevice, f32>>::Built,
    pub(crate) optim: Adam<<NetAutoModel as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
    grad: NetTape<f32, NetDevice>,
}

pub(crate) struct NetCrit {
    pub(crate) dev: NetDevice,
    pub(crate) net: <NetCritModel as BuildOnDevice<NetDevice, f32>>::Built,
    optim: Adam<<NetCritLearn as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
    grad: NetTape<f32, NetDevice>,
}

pub(crate) struct NetActor {
    pub(crate) dev: NetDevice,
    pub(crate) net: <NetActorModel as BuildOnDevice<NetDevice, f32>>::Built,
    optim: Adam<<NetActorLearn as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
    grad: NetTape<f32, NetDevice>,
}

fn make_config(lr: f64) -> AdamConfig {
    AdamConfig {
        lr,
        weight_decay: Some(WeightDecay::Decoupled(1.0)),
        ..Default::default()
    }
}

#[allow(dead_code)]
impl NetCrit {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetCritModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            // net.reset_params();
        }
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        }
    }

    pub(crate) fn load<P: AsRef<Path> + Display>(p: P) -> Option<Self> {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetCritModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            return None;
        }
        let lr: f64 = std::fs::read_to_string(format!("{p}.lr"))
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?
            .trim()
            .parse()
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Some(Self {
            optim: Adam::new(&net, make_config(lr)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn load_from_encode<P: AsRef<Path>>(p: P, enc_p: P) -> Result<Self, String> {
        let dev = NetDevice::default();

        let mut net_enc = dev.build_module::<NetAutoModel, f32>();
        net_enc
            .load(enc_p.as_ref())
            .map_err(|x| format!("Encoder: {}", x))?;
        let mut net = dev.build_module::<NetCritModel, _>();
        let _ = net.load(p.as_ref()); // ignore since it could be empty.
        net.0 = net_enc.0;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Ok(Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn save<P: AsRef<Path> + Display>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        std::fs::write(format!("{p}.lr"), format!("{}", self.optim.cfg.lr))?;
        Ok(())
    }

    pub(crate) fn forward(&self, img: &[f32], keys: &[f32]) -> Vec<f32> {
        let img = self.dev.tensor_from_vec(
            img.to_vec(),
            (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
        );
        let keys = self
            .dev
            .tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<INPUT_SIZE>));

        let out = self.forward_raw(
            img.put_tape(self.grad.clone()),
            keys.put_tape(self.grad.clone()),
        );

        out.as_vec()
    }

    fn forward_raw<const BATCH: usize>(
        &self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>>,
        keys: Tensor<Rank2<BATCH, 32>, f32, NetDevice, NetTape<f32, NetDevice>>,
    ) -> Tensor<Rank2<BATCH, 1>, f32, NetDevice, NetTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward(img);
        let in_conv: Tensor<(Const<BATCH>, Const<2400>), _, _, _> = Flatten2D.forward(out_conv);
        // let in_conv: Tensor<(_, Const<2400>), _, _, _> = in_conv;
        // let (a, t) = in_conv.split_tape();
        // let conv_v = a.as_vec();
        // let key_v = keys.as_vec();
        // let lin_v = conv_v
        //     .chunks_exact(LATENT_SIZE)
        //     .zip(key_v.chunks_exact(6))
        //     .map(|(a, b)| {
        //         let mut a = a.to_vec();
        //         a.extend_from_slice(b);
        //         a
        //     })
        //     .flatten()
        //     .collect::<Vec<f32>>();
        // let in_lin = dev
        //     .tensor_from_vec(lin_v, (Const::<BATCH>, Const::<10032>))
        //     .put_tape(t);
        let in_lin: Tensor<(Const<BATCH>, Const<CRITIC_SIZE>), _, _, _> =
            (in_conv, keys).concat_along(Axis::<1>);
        let out = self.net.1.forward(in_lin);
        out
    }

    fn forward_raw_mut<const BATCH: usize>(
        &mut self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>>,
        keys: Tensor<Rank2<BATCH, 32>, f32, NetDevice, NetTape<f32, NetDevice>>,
    ) -> Tensor<Rank2<BATCH, 1>, f32, NetDevice, NetTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward_mut(img);
        let in_conv: Tensor<(Const<BATCH>, Const<2400>), _, _, _> = Flatten2D.forward(out_conv);
        let in_lin = (in_conv, keys).concat_along(Axis::<1>);
        let out = self.net.1.forward_mut(in_lin);
        out
    }

    pub(crate) fn do_conv(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let img: Tensor<Rank4<1, 3, 400, 640>, f32, NetDevice, NoneTape> =
            self.dev.tensor_from_vec(
                img.to_vec(),
                (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
            );
        let out = self.net.0.forward(img);
        let arr = out.array();
        arr[0]
            .map(|x| x.iter().cloned().flatten().collect::<Vec<_>>())
            .to_vec()
    }

    pub(crate) fn from_conv(&self, conv: &[f32], keys: &[f32]) -> Vec<f32> {
        let mut iv = conv.to_vec();
        iv.extend_from_slice(keys);
        let conv: Tensor<Rank2<1, CRITIC_SIZE>, _, _> = self
            .dev
            .tensor_from_vec(iv, (Const::<1>, Const::<CRITIC_SIZE>));
        let out = self.net.1.forward(conv);
        out.array()[0].to_vec()
    }

    pub(crate) fn train(&mut self, img: &[f32], keys: &[f32], out: &[f32]) -> f32 {
        self.train_grad::<1>(&[img], &[keys], &[out])

        //self.net.zero_grads(&mut grads);
    }

    pub(crate) fn train_batch<const BATCH: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
    ) -> f32 {
        self.train_grad::<BATCH>(img, keys, out)

        //self.net.zero_grads(&mut grads);
    }

    pub(crate) fn train_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
    ) -> f32 {
        let (r, g) = self.acc_grad::<SS>(img, keys, out);
        // let s = g.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();
        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);
        self.backward(g);
        r
    }

    pub(crate) fn acc_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
    ) -> (f32, Gradients<f32, NetDevice>) {
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        let keys = self.dev.tensor_from_vec(
            keys.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<32>),
        );

        let model_out = self.forward_raw_mut(
            img.put_tape(self.grad.clone()),
            keys.put_tape(self.grad.clone()),
        );
        let out = self.dev.tensor_from_vec(
            out.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<1>),
        );
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);

        (r, err.backward())
    }

    pub(crate) fn backward(&mut self, grad: Gradients<f32, NetDevice>) {
        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();
        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);
        self.optim.update(&mut self.net, &grad).unwrap();
        self.clear_grad();
    }

    pub(crate) fn decay_lr(&mut self, factor: f64) {
        self.optim.cfg.lr *= factor;
    }

    pub(crate) fn get_lr(&self) -> f64 {
        self.optim.cfg.lr
    }

    pub(crate) fn clear_grad(&self) {
        let t = self
            .dev
            .tensor_from_vec(vec![0.], ())
            .put_tape(self.grad.clone());
        let mut grad = t.backward();
        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();
    }
}
#[allow(dead_code)]

impl NetAutoEncode {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetAutoModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            // net.reset_params();
        }
        // net.0 .0 .0.p = 0.4;
        // net.0 .1 .0.p = 0.4;
        // net.0 .2 .0.p = 0.4;
        // net.0 .3 .0.p = 0.4;
        // net.0 .4.p = 0.04;
        // // net.0 .5.p = 0.1;
        // net.1 .0 .0.p = 0.0;
        // net.1 .1 .0.p = 0.0;
        // net.1 .2 .0.p = 0.0;
        // net.1 .3 .5.p = 0.0;
        Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        }
    }

    pub(crate) fn save<P: AsRef<Path>>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        Ok(())
    }

    pub(crate) fn get_latent(&self, img: &[f32]) -> Vec<f32> {
        let img = self.dev.tensor_from_vec(
            img.to_vec(),
            (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
        );

        let out = self.net.0.forward(img);

        out.as_vec()
    }

    pub(crate) fn forward(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let img = self.dev.tensor_from_vec(
            img.to_vec(),
            (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
        );

        let out = self
            .forward_raw::<1>(img)
            .as_vec()
            .chunks(640 * 400)
            .map(|x| x.to_vec())
            .collect();

        out
    }

    pub(crate) fn forward_raw<const BATCH: usize>(
        &self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NoneTape>,
    ) -> Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NoneTape> {
        // let out_conv = self.net.0.forward(img);
        // let out_lin = self
        //     .net
        //     .1
        //      .0
        //     .forward(out_conv)
        //     .reshape::<Rank4<BATCH, 128, 7, 11>>();
        // let out = self.net.1 .1.forward(out_lin);

        // out
        self.net.forward(img)
    }

    fn forward_raw_mut<const BATCH: usize>(
        &mut self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>>,
    ) -> Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>> {
        // let out_conv = self.net.0 .0.forward_mut(img);
        // let out_conv = self.net.0 .1.forward_mut(out_conv);
        // let out_lin = self
        //     .net
        //     .1
        //      .0
        //     .forward_mut(out_conv)
        //     .reshape::<Rank4<BATCH, 128, 7, 11>>();
        // let out = self.net.1 .1.forward_mut(out_lin);
        // out
        self.net.forward_mut(img)
    }

    pub(crate) fn train(&mut self, img: &[f32]) -> f32 {
        self.train_grad::<1>(&[img])

        //self.net.zero_grads(&mut grads);
    }

    pub(crate) fn train_batch<const BATCH: usize>(&mut self, img: &[&[f32]]) -> f32 {
        self.train_grad::<BATCH>(img)

        //self.net.zero_grads(&mut grads);
    }

    pub(crate) fn train_grad<const SS: usize>(&mut self, img: &[&[f32]]) -> f32 {
        let out = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );

        let model_out = self.forward_raw_mut(img.put_tape(self.grad.clone()));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);

        let mut grad = err.backward();

        // println!("{:?}", grad.get(&self.net.0.0.1.0.weight).as_vec().into_iter().reduce(|x,y| x.abs() + y.abs()));
        // println!("{:?}", grad.get(&self.net.1.1.3.3.a).as_vec().into_iter().reduce(|x,y| x.abs() + y.abs()));

        //dbg!(&grads);

        self.optim.update(&mut self.net, &grad).unwrap();

        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();

        r
    }

    pub(crate) fn acc_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
    ) -> (f32, Gradients<f32, NetDevice>) {
        let out = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );

        let model_out = self.forward_raw_mut(img.put_tape(self.grad.clone()));
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let err = mse_loss(model_out, out);
        let r = err.as_vec()[0];
        //dbg!(mo, ou, r);

        let grad = err.backward();

        (r, grad)
    }

    pub(crate) fn backward(
        &mut self,
        mut grad: Gradients<f32, NetDevice>,
    ) -> Gradients<f32, NetDevice> {
        // println!("{:?}", grad.get(&self.net.0.0.1.0.weight).as_vec().into_iter().reduce(|x,y| x.abs() + y.abs()));
        // println!("{:?}", grad.get(&self.net.1.1.4.3.a).as_vec().into_iter().reduce(|x,y| x.abs() + y.abs()));
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
    frames
        .into_iter()
        .map(|x| {
            x.bytes()
                .map(|x| (x.unwrap() as f32) / 255.)
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

pub(crate) fn from_pixels(img: &[f32], shape: (u32, u32)) -> arena::GrayImage {
    if img.len() as u32 != shape.0 * shape.1 {
        panic!(
            "Does not fit, data is {} and shape is {:?}",
            img.len(),
            shape
        );
    }
    let img_data: Vec<_> = img.iter().map(|x| ((x) * 255.) as u8).collect();
    arena::GrayImage::from_vec(shape.1, shape.0, img_data).unwrap()
}

#[allow(dead_code)]
pub(crate) fn to_rb(img: &[f32], shape: (u32, u32), norm: bool) -> arena::Image {
    let max = img
        .iter()
        .fold(0.0f32, |m, v| if v.abs() > m.abs() { *v } else { m });
    let max = if max == 0.0 || !norm { 1.0 } else { max };
    let img_data: Vec<_> = img
        .iter()
        .cloned()
        .flat_map(|x| {
            let x = x / max;
            if x < 0. {
                [(-x * 255.) as u8, 0, 0]
            } else if x > 0. {
                [0, 0, (x * 255.) as u8]
            } else {
                [0, 0, 0]
            }
        })
        .collect();
    arena::Image::from_vec(shape.1, shape.0, img_data).unwrap()
}

// struct DummyTC<'a> {
//     actor: &'a mut NetActorModel,
//     crit: &'a mut NetCritModel,
// }

// impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for DummyTC {
//     type To<E2: dfdx::shapes::Dtype, D2: dfdx::tensor_ops::Device<E2>>;

//     fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
//         visitor: &mut V,
//     ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
//         visitor.visit_fields(
//             (
//                 Self::module("actor", |d| &d.actor, |d| &mut d.actor),
//                 Self::module("crit", |d| &d.crit, |d| &mut d.crit),
//             ),
//             |(actor, crit)| DummyTC { actor, crit },
//         )
//     }
// }

#[allow(dead_code)]
impl NetActor {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetActorModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            // net.reset_params();
        }
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        }
    }

    pub(crate) fn load<P: AsRef<Path> + Display>(p: P) -> Option<Self> {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetActorModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            return None;
        }
        let lr: f64 = std::fs::read_to_string(format!("{p}.lr"))
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?
            .trim()
            .parse()
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Some(Self {
            optim: Adam::new(&net, make_config(lr)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn load_from_encode<P: AsRef<Path>>(p: P, enc_p: P) -> Result<Self, String> {
        let dev = NetDevice::default();

        let mut net_enc = dev.build_module::<NetAutoModel, f32>();
        net_enc
            .load(enc_p.as_ref())
            .map_err(|x| format!("Encoder: {}", x))?;
        let mut net = dev.build_module::<NetActorModel, _>();
        let _ = net.load(p.as_ref()); // ignore since it could be empty.
        net.0 = net_enc.0;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Ok(Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }
    pub(crate) fn load_from_critic<P: AsRef<Path>>(p: P, crit_p: P) -> Result<Self, String> {
        let dev = NetDevice::default();

        let mut net_crit = dev.build_module::<NetCritModel, f32>();
        net_crit
            .load(crit_p.as_ref())
            .map_err(|x| format!("Encoder: {}", x))?;
        let mut net = dev.build_module::<NetActorModel, _>();
        let _ = net.load(p.as_ref()); // ignore since it could be empty.
        net.0 = net_crit.0;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Ok(Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn save<P: AsRef<Path> + Display>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        std::fs::write(format!("{p}.lr"), format!("{}", self.optim.cfg.lr))?;
        Ok(())
    }

    pub(crate) fn forward(&self, img: &[f32]) -> Vec<f32> {
        let img = self.dev.tensor_from_vec(
            img.to_vec(),
            (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
        );
        // let keys = dev.tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<6>));

        let out = self.forward_raw(img);

        out.as_vec()
    }

    fn forward_raw<const BATCH: usize>(
        &self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NoneTape>,
    ) -> Tensor<Rank2<BATCH, INPUT_SIZE>, f32, NetDevice, NoneTape> {
        let out_conv = self.net.0.forward(img);
        let in_conv: Tensor<(Const<BATCH>, Const<LATENT_SIZE>), _, _, _> =
            Flatten2D.forward(out_conv);
        let out = self.net.1.forward(in_conv);
        out
    }

    fn forward_raw_mut<const BATCH: usize>(
        &mut self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>>,
    ) -> Tensor<Rank2<BATCH, 32>, f32, NetDevice, NetTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward_mut(img);
        let in_conv: Tensor<(Const<BATCH>, Const<LATENT_SIZE>), _, _, _> =
            Flatten2D.forward(out_conv);
        // let in_lin = (in_conv, keys).concat_along(Axis::<1>);
        let out = self.net.1.forward_mut(in_conv);
        out
    }

    pub(crate) fn do_conv(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let img: Tensor<Rank4<1, 3, 400, 640>, f32, NetDevice, NoneTape> =
            self.dev.tensor_from_vec(
                img.to_vec(),
                (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
            );
        let out = self.net.0.forward(img);
        let arr = out.array();
        arr[0]
            .map(|x| x.iter().cloned().flatten().collect())
            .to_vec()
    }

    pub(crate) fn acc_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        critic: &NetCrit,
    ) -> (f32, Gradients<f32, NetDevice>) {
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        // grad.try_alloc_for(&critic.net);
        let enc = self.net.0.forward(img);
        let enc_flat: Tensor<(Const<SS>, Const<2400>), _, _, _> = Flatten2D.forward(enc);
        let model_out = self
            .net
            .1
            .forward_mut(enc_flat.clone().put_tape(self.grad.clone()));
        // let mc = m.clone();
        // let mcc = mc.clone();
        // let expand = (-mc.put_tape(g).clamp(0.0, 1.0) + mcc).abs();
        // let expand = expand.put_tape(g);
        // let (expand, g) = expand.split_tape();
        let crit_enc = (enc_flat.put_tape(self.grad.clone()), model_out).concat_along(Axis::<1>);
        let out = critic.net.1.forward(crit_enc);
        // let (crit_out, g) = crit_out.split_tape();
        // let out = dev.tensor_from_vec(
        // crit_out.iter().map(|&x| x).flatten().cloned().collect(),
        // (Const::<SS>, Const::<1>),
        // );
        //let mo = model_out.as_vec()[0];
        //let ou = out.as_vec()[0];

        let avg_score = out.mean();
        let r = avg_score.array();
        let err: Tensor<Rank0, f32, NetDevice, _> = -avg_score; // - crit_out.sum()

        // let (ent, tape) = err.split_tape();
        // let e2 = ent.clone();
        // let err = ent.put_tape(tape);
        // let r = -err.array();
        //dbg!(mo, ou, r);

        let grad = err.backward();

        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();

        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);

        // println!("{:?}", &self.net.0 .1 .0.weight.as_vec()[0..15]);

        (r, grad)
    }
    pub(crate) fn acc_grad2<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
    ) -> (f32, Gradients<f32, NetDevice>) {
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        let keys = self.dev.tensor_from_vec(
            keys.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<INPUT_SIZE>),
        );
        let out = self.dev.tensor_from_vec(
            out.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>,),
        );
        let model_out = self.forward_raw_mut(img.put_tape(self.grad.clone()));
        let key_dist = (model_out - keys).powi(2).sum::<(Const<SS>,), _>();
        let scale_score = dfdx::tensor_ops::mul(key_dist, out);
        let err = scale_score.mean();
        let r = err.array();
        let grad = err.backward();
        (r, grad)
    }
    pub(crate) fn backward(&mut self, mut grad: Gradients<f32, NetDevice>) {
        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();

        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);

        // TODO this is important
        // clamp_grad(&mut grad, &self.net);

        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();

        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);

        self.optim.update(&mut self.net, &grad).unwrap();

        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();
    }

    pub(crate) fn train_grad<const SS: usize>(&mut self, img: &[&[f32]], critic: &NetCrit) -> f32 {
        let (r, _) = self.acc_grad::<SS>(img, critic);
        r
    }
    pub(crate) fn grad(&self) -> Gradients<f32, NetDevice> {
        self.net.alloc_grads()
    }
    pub(crate) fn pair_grad(
        mut self,
        mut critic: NetCrit,
    ) -> (Self, NetCrit, Gradients<f32, NetDevice>) {
        let g = (self.net, critic.net);
        let grad = g.alloc_grads();
        self.net = g.0;
        critic.net = g.1;
        (self, critic, grad)
    }
    pub(crate) fn decay_lr(&mut self, factor: f64) {
        self.optim.cfg.lr *= factor;
    }

    pub(crate) fn get_lr(&self) -> f64 {
        self.optim.cfg.lr
    }
    pub(crate) fn clear_grad(&self) {
        let t = self
            .dev
            .tensor_from_vec(vec![0.], ())
            .put_tape(self.grad.clone());
        let mut grad = t.backward();
        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();
    }
}

struct ClampVisitor<'a, D: Storage<f32>> {
    grad: &'a mut Gradients<f32, D>,
    clampr: Range<f32>,
}

impl<'a> TensorVisitor<f32, Cuda> for ClampVisitor<'a, Cuda> {
    type Viewer = ViewTensorRef;

    type Err = String;

    type E2 = f32;

    type D2 = Cuda;

    fn visit<S: dfdx::shapes::Shape>(
        &mut self,
        _opts: dfdx::prelude::TensorOptions<S, f32, Cuda>,
        t: &Tensor<S, f32, Cuda>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err> {
        let cl = self.grad.get_or_alloc_mut(t).unwrap();
        let clamped = cl
            .device()
            .dtoh_sync_copy(cl)
            .unwrap()
            .into_iter()
            .map(|x| x.clamp(self.clampr.start, self.clampr.end))
            .collect::<Vec<_>>();
        cl.device().htod_copy_into(clamped, cl).unwrap();
        // *t = clamped;
        Ok(None)
    }
}
#[allow(dead_code)]
fn clamp_grad<T: TensorCollection<f32, Cuda>>(grad: &mut Gradients<f32, Cuda>, t: &T) {
    T::iter_tensors(&mut RecursiveWalker {
        m: t,
        f: &mut ClampVisitor {
            grad,
            clampr: -10. ..10.,
        },
    })
    .unwrap();
}

pub(crate) fn to_flat(keys: &[f32]) -> Vec<f32> {
    let mut ind = 0;
    if keys[0] == 1.0 {
        ind += 16;
    }
    if keys[2] == 1.0 {
        ind += 8;
    }
    if keys[3] == 1.0 {
        ind += 4;
    }
    if keys[4] == 1.0 {
        ind += 2;
    }
    if keys[5] == 1.0 {
        ind += 1;
    }
    let mut out = vec![0.0; 32];
    out[ind] = 1.0;
    out
}
pub(crate) fn from_flat(flat: &[f32]) -> Vec<f32> {
    let chosen = flat.iter().position(|x| *x == 1.0).unwrap();
    let mut keys = vec![0.0; 6];
    if chosen & 16 == 0 {
        keys[0] = 1.0
    }
    if chosen & 8 == 0 {
        keys[2] = 1.0
    }
    if chosen & 4 == 0 {
        keys[3] = 1.0
    }
    if chosen & 2 == 0 {
        keys[4] = 1.0
    }
    if chosen & 1 == 0 {
        keys[5] = 1.0
    }
    keys
}
pub(crate) fn collapse(flat: &[f32], explore: f32) -> Vec<f32> {
    let mut r = thread_rng();
    let mut rv = vec![0.0; 32];
    if r.gen::<f32>() < explore {
        rv[r.gen_range(0..32)] = 1.0;
        return rv;
    }
    let sum = flat.iter().sum::<f32>();
    let rn = r.gen_range(0f32..sum);
    let mut acc = 0.0;
    for (i, v) in flat.iter().enumerate() {
        acc += v;
        if acc >= rn {
            rv[i] = 1.0;
            return rv;
        }
    }
    panic!("Something failed to add up");
}

pub(crate) struct NetQ {
    net: <NetQModel as BuildOnDevice<NetDevice, f32>>::Built,
    optim: Adam<<NetQModel as BuildOnDevice<NetDevice, f32>>::Built, f32, NetDevice>,
    grad: Arc<Mutex<OwnedTape<f32, NetDevice>>>,
    dev: NetDevice,
}
impl NetQ {
    pub(crate) fn load_or_new<P: AsRef<Path>>(p: P) -> Self {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetQModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            eprintln!("Made new, {res:?}");
            // net.reset_params();
        }
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        }
    }

    pub(crate) fn load<P: AsRef<Path> + Display>(p: P) -> Option<Self> {
        let dev = NetDevice::default();

        let mut net = dev.build_module::<NetQModel, f32>();
        let res = net.load(p.as_ref());
        let loaded = res.is_ok();
        if !loaded {
            return None;
        }
        let lr: f64 = std::fs::read_to_string(format!("{p}.lr"))
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?
            .trim()
            .parse()
            .map_err(|x| eprintln!("{x:?}"))
            .ok()?;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1.0.1.p = 0.2;
        // net.1.1.1.p = 0.2;
        // net.1.3.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Some(Self {
            optim: Adam::new(&net, make_config(lr)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn load_from_encode<P: AsRef<Path>>(p: P, enc_p: P) -> Result<Self, String> {
        let dev = NetDevice::default();

        let mut net_enc = dev.build_module::<NetAutoModel, f32>();
        net_enc
            .load(enc_p.as_ref())
            .map_err(|x| format!("Encoder: {}", x))?;
        let mut net = dev.build_module::<NetQModel, _>();
        let _ = net.load(p.as_ref()); // ignore since it could be empty.
        net.0 = net_enc.0;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Ok(Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }
    pub(crate) fn load_from_critic<P: AsRef<Path>>(p: P, crit_p: P) -> Result<Self, String> {
        let dev = NetDevice::default();

        let mut net_crit = dev.build_module::<NetCritModel, f32>();
        net_crit
            .load(crit_p.as_ref())
            .map_err(|x| format!("Encoder: {}", x))?;
        let mut net = dev.build_module::<NetQModel, _>();
        let _ = net.load(p.as_ref()); // ignore since it could be empty.
        net.0 = net_crit.0;
        // net.0.0.1.0.p = 0.2;
        // net.0.0.2.0.p = 0.2;
        // net.0.0.3.0.p = 0.2;
        // net.0.0.4.p = 0.2;
        // net.1 .0 .1.p = 0.2;
        // net.1 .1 .1.p = 0.2;
        Ok(Self {
            optim: Adam::new(&net, make_config(LR)),
            grad: Arc::new(Mutex::new(net.alloc_grads().into())),
            net,
            dev,
        })
    }

    pub(crate) fn save<P: AsRef<Path> + Display>(&mut self, p: P) -> Result<(), std::io::Error> {
        self.net.save(p.as_ref())?;
        std::fs::write(format!("{p}.lr"), format!("{}", self.optim.cfg.lr))?;
        Ok(())
    }

    pub(crate) fn forward(&self, img: &[f32]) -> Vec<f32> {
        let img = self.dev.tensor_from_vec(
            img.to_vec(),
            (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
        );
        // let keys = dev.tensor_from_vec(keys.to_vec(), (Const::<1>, Const::<6>));

        let out = self.forward_raw(img);

        out.as_vec()
    }

    pub(crate) fn get_move(&self, img: &[f32]) -> usize {
        let p = self.forward(img);
        let m = p.iter().fold(f32::NEG_INFINITY, |x, y| x.max(*y));
        p.iter().position(|x| *x == m).expect("Was in the array")
    }

    fn forward_raw<const BATCH: usize>(
        &self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NoneTape>,
    ) -> Tensor<Rank2<BATCH, INPUT_SIZE>, f32, NetDevice, NoneTape> {
        let out_conv = self.net.0.forward(img);
        let in_conv: Tensor<(Const<BATCH>, Const<LATENT_SIZE>), _, _, _> =
            Flatten2D.forward(out_conv);
        let out = self.net.1.forward(in_conv);
        out
    }

    fn forward_raw_mut<const BATCH: usize>(
        &mut self,
        img: Tensor<Rank4<BATCH, 3, 400, 640>, f32, NetDevice, NetTape<f32, NetDevice>>,
    ) -> Tensor<Rank2<BATCH, 32>, f32, NetDevice, NetTape<f32, NetDevice>> {
        let out_conv = self.net.0.forward_mut(img);
        let in_conv: Tensor<(Const<BATCH>, Const<LATENT_SIZE>), _, _, _> =
            Flatten2D.forward(out_conv);
        // let in_lin = (in_conv, keys).concat_along(Axis::<1>);
        let out = self.net.1.forward_mut(in_conv);
        out
    }

    pub(crate) fn do_conv(&self, img: &[f32]) -> Vec<Vec<f32>> {
        let img: Tensor<Rank4<1, 3, 400, 640>, f32, NetDevice, NoneTape> =
            self.dev.tensor_from_vec(
                img.to_vec(),
                (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
            );
        let out = self.net.0.forward(img);
        let arr = out.array();
        arr[0]
            .map(|x| x.iter().cloned().flatten().collect())
            .to_vec()
    }

    pub(crate) fn acc_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
        imgnext: &[Option<&[f32]>],
        gamma: f32,
    ) -> (f32, Gradients<f32, NetDevice>) {
        let img = self.dev.tensor_from_vec(
            img.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>, Const::<3>, Const::<400>, Const::<640>),
        );
        let out = self.dev.tensor_from_vec(
            out.iter().map(|&x| x).flatten().cloned().collect(),
            (Const::<SS>,),
        );
        let keys: Tensor<Rank2<SS, INPUT_SIZE>, _, _, _> = self.dev.tensor_from_vec(
            keys.iter().map(|&x| x).flatten().cloned().collect(),
            Rank2::default(),
        );
        let model_out = mul(self.forward_raw_mut(img.put_tape(self.grad.clone())), keys).sum();
        let next_score = {
            let mask = imgnext
                .iter()
                .map(|x| if x.is_some() { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let mask: Tensor<Rank2<SS, INPUT_SIZE>, _, _, _> =
                self.dev.tensor_from_vec(mask, (Const::<SS>,)).broadcast();
            let blank = vec![0.0; 3 * 640 * 400];
            let imgnext: Vec<_> = imgnext
                .iter()
                .map(|&x| x.unwrap_or(&blank))
                .flatten()
                .cloned()
                .collect();
            // println!("{} {}", imgnext.len(), SS * 3 * 400 * 640);
            let imgnext: Tensor<Rank4<SS, 3, 400, 640>, _, _, _> =
                self.dev.tensor_from_vec(imgnext, Rank4::default());
            let pred = mul(
                self.forward_raw_mut(imgnext.put_tape(self.grad.clone())),
                mask,
            );
            pred.max() * gamma + out
        };
        let err = next_score.huber_error(model_out, 1.0).sum();
        let r = err.array();
        let grad = err.backward();
        (r, grad)
    }
    pub(crate) fn backward(&mut self, mut grad: Gradients<f32, NetDevice>) {
        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();

        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);

        // TODO this is important
        // clamp_grad(&mut grad, &self.net);

        // let s = grad.get_ref_checked(&self.net.0 .1 .0.weight).unwrap();

        // let v = &s.device().dtoh_sync_copy(s.deref()).unwrap();
        // dbg!(&v[0..10]);

        self.optim.update(&mut self.net, &grad).unwrap();

        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();
    }

    pub(crate) fn train_grad<const SS: usize>(
        &mut self,
        img: &[&[f32]],
        keys: &[&[f32]],
        out: &[&[f32]],
        imgnext: &[Option<&[f32]>],
        gamma: f32,
    ) -> f32 {
        let (r, _) = self.acc_grad::<SS>(img, keys, out, imgnext, gamma);
        r
    }
    pub(crate) fn grad(&self) -> Gradients<f32, NetDevice> {
        self.net.alloc_grads()
    }
    pub(crate) fn pair_grad(
        mut self,
        mut critic: NetCrit,
    ) -> (Self, NetCrit, Gradients<f32, NetDevice>) {
        let g = (self.net, critic.net);
        let grad = g.alloc_grads();
        self.net = g.0;
        critic.net = g.1;
        (self, critic, grad)
    }
    pub(crate) fn decay_lr(&mut self, factor: f64) {
        self.optim.cfg.lr *= factor;
    }

    pub(crate) fn get_lr(&self) -> f64 {
        self.optim.cfg.lr
    }
    pub(crate) fn clear_grad(&self) {
        let t = self
            .dev
            .tensor_from_vec(vec![0.], ())
            .put_tape(self.grad.clone());
        let mut grad = t.backward();
        self.net.zero_grads(&mut grad);
        grad.drop_non_leafs();
        *self.grad.lock().unwrap() = grad.into();
    }
}

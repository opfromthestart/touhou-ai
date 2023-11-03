#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(array_chunks)]
#![feature(iter_array_chunks)]

#[allow(unused_imports)]
use std::time::SystemTime;
use std::{
    collections::BTreeMap,
    ops::{Div, Mul},
    thread,
    time::Duration,
};

use arena::{analyze_crit, get_q_iter, join_channel, process_crit, GrayImage};
use clap::Parser;
use dfdx::{
    shapes::Const,
    tensor::{AsArray, Gradients, Tensor, TensorFromVec, ZerosTensor},
};
use net::{from_pixels, NetActor};
use once_cell::sync::Lazy;
use rand::{thread_rng, Rng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    arena::{
        do_dialog, get_actor_iter, get_crit_iter, get_dia, get_gameover, get_life, get_pixel_range,
        has_lost, num_lives, reset_from_gameover, wants_exit,
    },
    net::{from_flat, to_flat, NetCrit, NetDevice, NetQ},
};

//use coaster::{Backend, Cuda, frameworks::cuda::get_cuda_backend};
//use ndarray::Dim;

mod arena;
mod net;

#[derive(Parser, Debug)]
enum Mode {
    EvalCritic,
    // Eval real time
    EvalGame,
    // Train critic on active data
    TrainCritic,
    // Train on existing autoencoder data
    TrainEncode,
    // Test autoencoder
    EvalEncode,
    // Test outoencoder created in python
    EvalPython,
    // Gather images for the autoencoder
    GatherEncode,
    // Gather images and score for the critic
    GatherCritic,
    // Train actor from critic
    TrainActor { use_crit: String },
    ProcessCritic,
    GatherPlay,
    Analyze,
    AnalyzeActor { bins: String },
    ResetLr,
    DoLoop,
    TestQ,
}

fn main() {
    let m = Mode::parse();

    match m {
        Mode::EvalCritic => test_img(None),
        Mode::TrainCritic => train(None),
        Mode::TrainEncode => train_encode(),
        Mode::EvalEncode => test_encode(None),
        Mode::GatherEncode => gather_encode(),
        Mode::EvalPython => test_python(),
        Mode::EvalGame => todo!(),
        Mode::GatherCritic => gather_critic(),
        Mode::TrainActor { use_crit } => train_actor(
            if use_crit == "crit" {
                true
            } else if use_crit == "nocrit" {
                false
            } else {
                panic!("Must use either \"crit\" or \"nocrit\".")
            },
            None,
        ),
        Mode::ProcessCritic => process_crit(0.97),
        Mode::GatherPlay => gather_live_actor(10000, None, None),
        Mode::Analyze => analyze_crit(),
        Mode::AnalyzeActor { bins } => analyze_actor(bins),
        Mode::ResetLr => reset_lr(),
        Mode::DoLoop => do_loop(),
        Mode::TestQ => test_q(),
    }
    //test_img();
}
const SINGLE: bool = false;

const BATCH: usize = 10;
const ENC_BATCH: usize = 2;
const DECAY_RATE: f64 = 1.0;
static DATASIZE: Lazy<Option<usize>> = Lazy::new(|| {
    let x = std::fs::read_to_string("ai-file/batch")
        .unwrap()
        .trim()
        .parse()
        .ok();
    match x {
        Some(v) => println!("Using batch size {v}"),
        None => println!("Using all data"),
    }
    x
});

fn train(limit: Option<usize>) {
    eprintln!("making net");
    let limit = limit.unwrap_or(usize::MAX);

    let mut net = match NetCrit::load("ai-file/th2_critic.net") {
        Some(n) => n,
        None => {
            println!("Loading from encoder");
            match NetCrit::load_from_encode("ai-file/th2_critic.net", "ai-file/th2_encode.net") {
                Ok(n) => n,
                Err(e) => {
                    println!("Could not use encoder: {e:?}");
                    NetCrit::load_or_new("ai-file/th2_critic.net")
                }
            }
        }
    }; //::<Backend<Cuda>>

    // let gamma = 0.97;
    let mut acc_err = f32::MAX;
    let mut epoch = 0;
    let target_err = 0.01f32.powi(2) * 0.0001;
    let mut prev_err = f32::MAX;

    let mut grad: Option<Gradients<_, _>>;

    let (mut outp, mut test_outp) = get_crit_iter().split_test(16000);
    let datasize = DATASIZE.unwrap_or(outp.len());

    let mut r = thread_rng();

    test_img(Some(&net));

    println!("Training");
    while acc_err > target_err && epoch < limit {
        // println!("Getting images and data");

        outp.reset();
        outp.shuffle(&mut r);
        test_outp.reset();
        // permute(&mut outp, &perm);

        let epoch_start = std::time::SystemTime::now();

        acc_err = 0.0;

        let samples = datasize;

        // let img = img.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
        // let keys = keys.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
        for (i, b) in (&mut outp)
            .array_chunks::<BATCH>()
            .take(datasize)
            .enumerate()
        {
            let img = b.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
            let keys = b.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
            let acc_own = b.iter().map(|x| [x.2]).collect::<Vec<_>>();
            let acc = acc_own.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
            let (derr, grad2) = net.acc_grad::<BATCH>(&img, &keys, &acc);
            grad = Some(grad2);
            acc_err += derr;

            net.backward(grad.unwrap());
            // grad = None;
            let est_remain = epoch_start
                .elapsed()
                .unwrap()
                .div((i + 1) as u32)
                .mul((samples / BATCH - i - 1) as u32);
            eprint!(
                "{:8}/{samples:8} Err: {} Remaining: {:?}    \r",
                (i + 1) * BATCH,
                acc_err / ((i + 1) * BATCH) as f32,
                est_remain
            );
        }
        // if i % 400 == 0 {
        //     if !acc_err.is_nan() && acc_err.is_finite() {
        //         net.save("ai-file/th2_critic.net").unwrap();
        //         test_img()
        //     } else {
        //         eprintln!("Get [NaN|inf]ed lol.");
        //         break 'ep_loop;
        //     }
        // }

        //grad = net.backward(grad);
        acc_err /= datasize as f32;
        epoch += 1;
        if prev_err < acc_err {
            net.decay_lr(DECAY_RATE);
        }
        prev_err = acc_err;
        let mut test_err = 0.0;
        for b in (&mut test_outp).array_chunks::<BATCH>() {
            let img = b.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
            let keys = b.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
            let acc_own = b.iter().map(|x| [x.2]).collect::<Vec<_>>();
            let acc = acc_own.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
            let (derr, _) = net.acc_grad::<BATCH>(&img, &keys, &acc);
            test_err += derr;
        }
        test_err /= test_outp.len() as f32;
        eprintln!(
            "Epoch:{epoch:4} Error:      {acc_err} Test err: {test_err} New LR: {}                           ",
            net.get_lr()
        );

        if !acc_err.is_nan() && acc_err.is_finite() {
            net.save("ai-file/th2_critic.net").unwrap();
            test_img(Some(&net));
        } else {
            eprintln!("Get [NaN|inf]ed lol.");
            break;
        }
    }

    if !acc_err.is_nan() && acc_err.is_finite() {
        net.save("ai-file/th2_critic.net").unwrap();
    }

    //eprintln!("{:?}", start.elapsed());

    //eprintln!("{:?}", arena::get_score(images.last().unwrap(), &num_list[..]));
    //fs::write("win_ss.bmp", images.last().unwrap().buffer()).unwrap();
}

// Does a weird but stable permutation
#[allow(dead_code)]
fn permute<T>(v: &mut [T], p: &[usize]) {
    for i in p {
        v.swap(0, *i);
    }
}

fn test_img(net: Option<&NetCrit>) {
    let to_run = image::open("dbg_images/test img4.png").unwrap().to_rgb8();
    let to_run_pix = net::to_pixels(&to_run);
    let to_run4 = image::open("dbg_images/test img3.png").unwrap().to_rgb8();
    let to_run4_pix = net::to_pixels(&to_run4);
    //eprintln!("{:?}", to_run_pix)eprintln!("making net");
    let net = net.unwrap_or_else(|| {
        let net = match NetCrit::load("ai-file/th2_critic.net") {
            Some(n) => n,
            None => {
                println!("Could not load critic in test");
                NetCrit::load_or_new("ai-file/th2_critic.net")
            }
        };
        let net = Box::new(net);
        Box::<NetCrit>::leak(net)
    });
    //::<Backend<Cuda>>
    // println!("{}", to_run_pix.len());
    // let conv = net.do_conv(&to_run_pix);
    // println!("{}", conv.len());
    // let s = conv[0].len() as f32;
    // let t = conv
    //     .iter()
    //     .map(|x| x.iter().map(|x| x.abs()).sum::<f32>() / s)
    //     .collect::<Vec<_>>();
    // eprintln!("weight: {:?}", t);
    let keys = to_flat(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let zkeys = to_flat(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // conv.into_iter()
    //     .enumerate()
    //     .map(|(i, p)| (to_rb(&p, (8, 13), true), i))
    //     .for_each(|(x, i)| x.save(format!("dbg_images/post_conv{i}.png")).unwrap());
    let score = net.forward(&to_run_pix, &keys);
    eprint!("\t\t\t\t\t\t\tScore nothing: {score:?}");
    let score = net.forward(&to_run4_pix, &keys);
    eprint!("Score stuff: {score:?}");
    let score = net.forward(&to_run4_pix, &zkeys);
    eprintln!("Score stuff no keys: {score:?}");

    // let to_run = image::open("dbg_images/test img2.png").unwrap().to_rgb8();
    // let to_run_pix = net::to_pixels(&to_run);
    //eprintln!("{:?}", to_run_pix)eprintln!("making net");
    // let net = net::NetCrit::load_or_new("ai-file/th2_critic.net"); //::<Backend<Cuda>>
    // let conv = net.do_conv(&to_run_pix);
    // let t = conv
    //     .iter()
    //     .map(|x| x.iter().map(|x| x.abs()).sum::<f32>() / s)
    //     .collect::<Vec<_>>();
    // eprintln!("weight: {:?}", t);
    // conv.into_iter()
    //     .enumerate()
    //     .map(|(i, p)| (to_rb(&p, (33, 53), true), i))
    //     .for_each(|(x, i)| x.save(format!("dbg_images/post_conv{i}_2.png")).unwrap());

    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     // println!("{:?}", net.forward(&to_run_pix, &keys));
    //     // let start = Instant::now();
    //     net.forward(&to_run_pix, &keys);
    //     // println!("{:?}", start.elapsed());
    // }
    // println!("critic took {:?} for 60", now.elapsed().unwrap());

    // drop(net);
    // let act = NetActor::load_or_new("ai-file/th2_actor.net");
    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     // println!("{:?}", net.forward(&to_run_pix, &keys));
    //     // let start = Instant::now();
    //     act.forward(&to_run_pix);
    //     // println!("{:?}", start.elapsed());
    // }
    // println!("actor took {:?} for 60", now.elapsed().unwrap());

    // let latent = net
    //     .do_conv(&to_run_pix)
    //     .into_iter()
    //     .flatten()
    //     .collect::<Vec<_>>();
    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     let lat = net.do_conv(&to_run_pix);
    //     let out = net.from_conv(&lat.into_iter().flatten().collect::<Vec<_>>(), &keys);
    //     // println!("{out:?}");
    // }
    // println!("Took {:?} for 60 decomp", now.elapsed().unwrap());

    // let score = net.forward(&to_run_pix, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    // eprintln!("Score: {score:?}");

    // let bytes = std::fs::read("ai-file/1.0.bias").unwrap()[74..].to_vec();
    // let l = bytes.len()/4;
    // let floats : Vec<f32> = unsafe {
    //     std::mem::transmute(bytes)
    // };
    // eprintln!("{:?}", &floats[..l]);
    net.clear_grad();
}

#[test]
fn does_permute() {
    let perm = vec![0, 2, 1, 4, 3];
    let mut v1 = vec![12, 23, 34, 45, 56];
    let mut v2 = v1.clone();
    permute(&mut v1, &perm);
    permute(&mut v2, &perm);
    assert_eq!(&v1, &v2);
}

fn train_encode() {
    let datasize = ENC_BATCH * 5000;

    eprintln!("making net");
    let mut net = net::NetAutoEncode::load_or_new("ai-file/th2_encode.net"); //::<Backend<Cuda>>

    let mut acc_err = 1.0;
    let mut epoch = 0;
    let target_err = 0.03f32.powi(2);

    let mut outp = get_actor_iter();

    let mut r = thread_rng();

    if SINGLE {
        outp.files = vec!["dbg_images/test img.png".into()];
    }

    while acc_err > target_err {
        eprint!("Loading images\r");
        if !SINGLE {
            outp.reset();
            outp.shuffle(&mut r);
        }

        let epoch_start = std::time::SystemTime::now();

        acc_err = 0.0;

        // let samples = outp.len() / ENC_BACK_BATCH;

        for (i, imgb) in (outp)
            .take(datasize)
            .array_chunks::<ENC_BATCH>()
            .enumerate()
        {
            let imgb = imgb.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
            let derr = net.train_grad::<ENC_BATCH>(&imgb);
            acc_err += derr;
            // grad = net.backward(grad);
            // acc_err += net.train_batch::<ENC_MEM_BATCH>(imgb);
            let est_remain = epoch_start
                .elapsed()
                .unwrap()
                .div((i + 1) as u32)
                .mul((datasize / ENC_BATCH - i - 1) as u32);
            eprint!(
                "{:8}/{:8} Err: {} Remaining: {:?}    \r",
                i + 1,
                datasize / ENC_BATCH,
                acc_err / ((i + 1) * ENC_BATCH) as f32,
                est_remain
            );
        }

        //grad = net.backward(grad);
        acc_err /= datasize as f32;
        epoch += 1;
        eprintln!("Epoch:{epoch:4}      Error: {acc_err}                            ");

        if !acc_err.is_nan() && acc_err.is_finite() && epoch % 1 == 0 {
            net.save("ai-file/th2_encode.net").unwrap();
            // println!("Saved");
            test_encode((&mut outp).next());
        } else if acc_err.is_nan() || acc_err.is_infinite() {
            eprintln!("Get [NaN|inf]ed lol.");
            break;
        }
    }

    if !acc_err.is_nan() && acc_err.is_finite() {
        net.save("ai-file/th2_encode.net").unwrap();
    }

    //eprintln!("{:?}", start.elapsed());

    //eprintln!("{:?}", arena::get_score(images.last().unwrap(), &num_list[..]));
    //fs::write("win_ss.bmp", images.last().unwrap().buffer()).unwrap();
}

fn test_encode(pixels: Option<Vec<f32>>) {
    // println!("in encoe");
    let to_run_pix = match pixels {
        Some(p) => p,
        None => {
            let Ok(to_run) = image::open("dbg_images/test img.png") else {
                println!("debug images not found, make an image dbg_images/test img.png");
                return;
            };
            let to_run = to_run.into_rgb8();
            net::to_pixels(&to_run)
        }
    };
    // println!("Made pixls");
    let c = 640 * 400;
    let mut ins = [
        GrayImage::default(),
        GrayImage::default(),
        GrayImage::default(),
    ];
    ins[0] = from_pixels(&&to_run_pix[0..c], (400, 640));
    ins[1] = from_pixels(&&to_run_pix[c..2 * c], (400, 640));
    ins[2] = from_pixels(&&to_run_pix[2 * c..], (400, 640));
    ins[0].save("dbg_images/as_input_r.png").unwrap();
    ins[1].save("dbg_images/as_input_g.png").unwrap();
    ins[2].save("dbg_images/as_input_b.png").unwrap();
    join_channel(&ins).save("dbg_images/as_input.png").unwrap();
    //eprintln!("{:?}", to_run_pix);
    // eprintln!("making net");
    let net = net::NetAutoEncode::load_or_new("ai-file/th2_encode.net"); //::<Backend<Cuda>>

    // eprintln!("Loaded");
    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     net.get_latent(&to_run_pix);
    // }
    // println!("Took {:?} for 60", now.elapsed().unwrap());
    // let hl = net.get_latent(&to_run_pix);
    // println!("{:?}", &hl.to_vec()[0..50]);
    let conv = net.forward(&to_run_pix);
    // eprintln!("Ran network");
    let mut grays: [GrayImage; 3] = [
        GrayImage::default(),
        GrayImage::default(),
        GrayImage::default(),
    ];
    grays[0] = from_pixels(&conv[0], (400, 640));
    grays[1] = from_pixels(&conv[1], (400, 640));
    grays[2] = from_pixels(&conv[2], (400, 640));

    join_channel(&grays).save("dbg_images/encoded.png").unwrap();
    // println!(
    //     "{:?} {:?}",
    //     to_run_pix.iter().fold(f32::INFINITY, |a, b| a.min(*b)),
    //     to_run_pix
    //         .as_slice()
    //         .iter()
    //         .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
    // );
    // println!(
    //     "{:?} {:?}",
    //     conv[0].iter().fold(f32::INFINITY, |a, b| a.min(*b)),
    //     conv[0]
    //         .as_slice()
    //         .iter()
    //         .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
    // );
}

trait IsClose {
    fn is_close(&self, other: &Self) -> bool;
}

impl IsClose for f32 {
    fn is_close(&self, other: &Self) -> bool {
        (self - other).abs() < 1e-5
    }
}

impl<T: IsClose> IsClose for [T] {
    fn is_close(&self, other: &Self) -> bool {
        // st_break(&format!("{}, {}", self.len(), other.len()));
        if self.len() != other.len() {
            false
        } else {
            self.iter().zip(other.iter()).all(|(i, j)| i.is_close(j))
        }
    }
}
impl<T: IsClose> IsClose for Vec<T> {
    fn is_close(&self, other: &Self) -> bool {
        // st_break(&format!("{}, {}", self.len(), other.len()));
        if self.len() != other.len() {
            false
        } else {
            self.iter().zip(other.iter()).all(|(i, j)| i.is_close(j))
        }
    }
}
impl<T: IsClose, const N: usize> IsClose for [T; N] {
    fn is_close(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            false
        } else {
            // println!("{}, {}", self.len(), other.len());
            self.iter().zip(other.iter()).all(|(i, j)| i.is_close(j))
        }
    }
}

#[allow(dead_code)]
fn st_break(s: &str) {
    println!("{}", s);
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).unwrap();
}

fn assert_close<S: IsClose>(t1: &S, t2: &S) {
    // st_break("in asert");

    assert!(t1.is_close(t2), "Do not match");
}

fn test_python() {
    let to_run_pix = {
        let Ok(to_run) = image::open("dbg_images/test img.png") else {
            println!("debug images not found, make an image dbg_images/test img.png");
            return;
        };
        let to_run = to_run.into_rgb8();
        net::to_pixels(&to_run)
    };
    println!("tes1");
    let c = 640 * 400;
    let mut ins = [
        GrayImage::default(),
        GrayImage::default(),
        GrayImage::default(),
    ];
    println!("tes2");
    ins[0] = from_pixels(&&to_run_pix[0..c], (400, 640));
    ins[1] = from_pixels(&&to_run_pix[c..2 * c], (400, 640));
    ins[2] = from_pixels(&&to_run_pix[2 * c..], (400, 640));
    ins[0].save("dbg_images/as_input_r.png").unwrap();
    ins[1].save("dbg_images/as_input_g.png").unwrap();
    ins[2].save("dbg_images/as_input_b.png").unwrap();
    println!("tes3");
    join_channel(&ins).save("dbg_images/as_input.png").unwrap();
    println!("tes4");
    //eprintln!("{:?}", to_run_pix);
    //eprintln!("making net");
    let net = net::NetAutoEncode::load_or_new("ai-file/pymodel.npz"); //::<Backend<Cuda>>
                                                                      //eprintln!("Loaded");
    let dev = NetDevice::default();
    println!("tes5");
    // let cpudev = Cpu::default();
    let t = dev.tensor_from_vec(
        to_run_pix.clone(),
        (Const::<1>, Const::<3>, Const::<400>, Const::<640>),
    );
    println!("tes6");
    let mut npin: Tensor<_, f32, _> = dev.zeros_like(&t);
    npin.load_from_npy("dbg_images/npin.npy").unwrap();

    println!("tes7");
    assert_close(&t.as_vec(), &npin.as_vec());

    println!("tes8");
    // let conv = net.forward(&to_run_pix);
    let conv_raw = net.forward_raw(t);
    println!("tes9");
    let mut npout: Tensor<_, f32, _> = dev.zeros_like(&conv_raw);
    npout.load_from_npy("dbg_images/outp.npy").unwrap();
    println!("tes1");
    assert_close(&conv_raw.as_vec(), &npout.as_vec());
    println!("tes2");
    let conv: Vec<Vec<_>> = conv_raw.array()[0]
        .iter()
        .map(|x| x.iter().cloned().flatten().collect())
        .collect();
    println!("tes3");

    //eprintln!("Ran network");
    let mut grays: [GrayImage; 3] = [
        GrayImage::default(),
        GrayImage::default(),
        GrayImage::default(),
    ];
    grays[0] = from_pixels(&conv[0], (400, 640));
    grays[1] = from_pixels(&conv[1], (400, 640));
    grays[2] = from_pixels(&conv[2], (400, 640));

    println!("tes4");
    join_channel(&grays)
        .save("dbg_images/encoded_rs.png")
        .unwrap();
    println!(
        "{:?} {:?}",
        to_run_pix.iter().fold(f32::INFINITY, |a, b| a.min(*b)),
        to_run_pix
            .as_slice()
            .iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
    );
    println!(
        "{:?} {:?}",
        conv[0].iter().fold(f32::INFINITY, |a, b| a.min(*b)),
        conv[0]
            .as_slice()
            .iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(*b))
    );
}

fn gather_encode() {
    eprintln!("Open the game within 2 seconds of starting this.");
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");

    let num_list = arena::get_nums();
    let gameover = get_gameover();
    let mut r = thread_rng();

    let ss = arena::screenshot(pos);
    ss.save("dbg_images/test_ss.bmp").unwrap();

    arena::start(pos, &num_list);

    eprintln!("Start playing now");

    let start = std::time::SystemTime::now();

    let frame_time = 1.0 / 60.0;
    let mut frame_el;
    let mut frames = 0;
    loop {
        frame_el = std::time::SystemTime::now();
        let img = arena::screenshot(pos);
        img.save(format!(
            "images/encoder_data/{}_{}_{}.png",
            r.next_u32(),
            r.next_u32(),
            r.next_u32()
        ))
        .unwrap();
        frames += 1;

        // Will run out of memory for me if larger (i have 32 GB)
        if wants_exit() || has_lost(&img, &gameover) {
            break;
        }

        while frame_el.elapsed().unwrap().as_secs_f32() < frame_time {
            thread::sleep(
                Duration::from_secs_f32(frame_time).saturating_sub(frame_el.elapsed().unwrap()),
            )
        }
    }
    dbg!(start.elapsed().unwrap());
    println!(
        "{}",
        (frames as f32) / start.elapsed().unwrap().as_secs_f32()
    );
}

fn gather_critic() {
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");

    let num_list = arena::get_nums();
    let gameover = get_gameover();
    let mut r = thread_rng();

    let ss = arena::screenshot(pos);
    ss.save("dbg_images/test_ss.bmp").unwrap();

    eprintln!("Initializing");

    arena::start(pos, &num_list);

    eprintln!("Start playing now");

    let start = std::time::SystemTime::now();

    let mut frames = 0;
    let mut score = 0;

    let frame_time = 1.0 / 60.0;
    let mut frame_el;
    let rn = format!("{}{}", r.next_u64(), r.next_u64());
    loop {
        frame_el = std::time::SystemTime::now();
        let img = arena::screenshot(pos);
        // let img_pix = net::to_pixels(&img);

        //dbg!(&data.as_slice().unwrap()[0..15]);
        let keys = arena::get_keys()
            .map(|x| if x { 1.0 } else { 0.0 })
            .to_vec();
        //let c = get_cuda_backend();
        //dbg!(&res);
        //res.forward();
        let new_score = arena::get_score(&img, &num_list).unwrap_or(score);

        //eprintln!("{:?}", new_score);
        let score_d = new_score - score;
        score = new_score;

        frames += 1;

        let name = format!("{}_{}", rn, frames);
        img.save(format!("images/critic_data/{}.png", name))
            .unwrap();
        let data = serde_json::to_string(&(keys, score_d)).unwrap();
        std::fs::write(format!("images/critic_data/{}.json", name), data).unwrap();

        // TEST
        if wants_exit() || has_lost(&img, &gameover) {
            break;
        }

        //let keys = res.iter().map(|x| *x>0.5).collect::<Vec<_>>();

        //arena::do_out(&keys);
        //outp.push(res);

        //eprint!("{:?}         \\r", arena::get_score(&img, &num_list[..]));
        while frame_el.elapsed().unwrap().as_secs_f32() < frame_time {
            thread::sleep(
                Duration::from_secs_f32(frame_time).saturating_sub(frame_el.elapsed().unwrap()),
            )
        }
    }
    dbg!(start.elapsed().unwrap());
    println!(
        "{}",
        (frames as f32) / start.elapsed().unwrap().as_secs_f32()
    );
    // let mut r = thread_rng();
    std::fs::write(
        format!("images/critic_data/{}.seq", rn),
        format!("{}", frames),
    )
    .unwrap();
}

fn train_actor(use_critic: bool, limit: Option<usize>) {
    let critic = if use_critic {
        println!("Using critic");
        Some(NetCrit::load("ai-file/th2_critic.net").expect("No critic found"))
    } else {
        println!("Not using critic");
        None
    };
    let mut actor = match NetActor::load("ai-file/th2_actor.net") {
        Some(n) => n,
        None => {
            println!("Loading from critic");
            match NetActor::load_from_critic("ai-file/th2_actor.net", "ai-file/th2_critic.net") {
                Ok(n) => n,
                Err(_) => {
                    println!("Could not use encoder");
                    NetActor::load_or_new("ai-file/th2_actor.net")
                }
            }
        }
    }; //::<Backend<Cuda>>
    if let Some(c) = critic.as_ref() {
        actor.dev = c.dev.clone();
    }
    let limit = limit.unwrap_or(2000);

    let mut r = thread_rng();
    let (mut outpc, mut testc) = get_crit_iter().split_test(16000);
    let (mut outpa, mut testa) = get_actor_iter().split_test(16000);
    let datasize = DATASIZE.unwrap_or(outpc.len());
    // let img = get_enc_batch(1, &mut r);
    // let mut outp: Vec<Vec<f32>> = vec![net::to_pixels(&img[0])];
    // let datasize = outp.len();
    let mut acc_err;
    let mut prev_err = f32::MIN;
    let mut test_err;

    // let (mut actor, critic, mut grad) = actor.pair_grad(critic);
    let mut grad;

    for epoch in 0..limit {
        // eprint!("Loading images\r");
        // if !SINGLE || epoch == 0 {
        //     outp = get_enc_iter();
        // }

        let epoch_start = std::time::SystemTime::now();

        acc_err = 0.0;

        // let samples = outpc.len() / BACK_BATCH;

        // if true {
        //     for (i, img) in outp
        //         .array_chunks::<BACK_BATCH>()
        //         .enumerate()
        //         .take(datasize / BACK_BATCH)
        //     {
        //         for imgb in img.array_chunks::<MEM_BATCH>() {
        //             let imgb = imgb.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
        //             (derr, grad) = actor.acc_grad::<MEM_BATCH>(&imgb, &critic, grad);
        //             acc_err += derr;
        //             // grad = net.backward(grad);
        //             // acc_err += net.train_batch::<MEM_BATCH>(imgb);
        //         }
        //         grad = actor.backward(grad);

        //         let est_remain = epoch_start
        //             .elapsed()
        //             .unwrap()
        //             .div((i + 1) as u32)
        //             .mul((datasize / BACK_BATCH - i - 1) as u32);

        //         // let ao = actor.forward(img[0]);
        //         // let co = critic.forward(img[0], &ao);
        //         eprint!(
        //             "{}/{datasize} Err: {} Remaining: {:?}  \r", // Result {ao:?} Score {co:?}
        //             (i + 1) * BACK_BATCH,
        //             acc_err / ((i + 1) * BACK_BATCH) as f32,
        //             est_remain,
        //         );
        //     }
        // }
        test_err = 0.0;
        if use_critic {
            let cr = critic.as_ref().unwrap();
            for (i, img) in outpa
                .array_chunks::<BATCH>()
                .enumerate()
                .take(datasize / BATCH)
            {
                let img = img.iter().map(|x| &x as &[f32]).collect::<Vec<_>>();
                let (derr, grad2) = actor.acc_grad::<BATCH>(&img, cr);
                grad = Some(grad2);
                acc_err += derr;
                // grad = net.backward(grad);
                // acc_err += net.train_batch::<MEM_BATCH>(imgb);
                actor.backward(grad.unwrap());
                // grad = None;

                let est_remain = epoch_start
                    .elapsed()
                    .unwrap()
                    .div((i + 1) as u32)
                    .mul((datasize / BATCH - i - 1) as u32);

                // let ao = actor.forward(img[0]);
                // let co = critic.forward(img[0], &ao);
                eprint!(
                    "{:8}/{datasize:8} Err: {} Remaining: {:?}  \r", // Result {ao:?} Score {co:?}
                    (i + 1) * BATCH,
                    acc_err / ((i + 1) * BATCH) as f32,
                    est_remain,
                );
            }
            for b in (&mut testa).array_chunks::<BATCH>() {
                let img = b.iter().map(|x| &x as &[f32]).collect::<Vec<_>>();
                let (derr, _) = actor.acc_grad::<BATCH>(&img, cr);
                test_err += derr;
            }
            test_err /= testa.len() as f32;
        } else {
            for (i, img) in outpc
                .array_chunks::<BATCH>()
                .enumerate()
                .take(datasize / BATCH)
            {
                let imgb = img.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
                let keys = img.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
                let scores_ = img.iter().map(|x| vec![x.2]).collect::<Vec<_>>();
                let scores = scores_.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
                let (derr, grad2) = actor.acc_grad2::<BATCH>(&imgb, &keys, &scores);
                grad = Some(grad2);
                acc_err += derr;
                // grad = net.backward(grad);
                // acc_err += net.train_batch::<MEM_BATCH>(imgb);

                actor.backward(grad.unwrap());
                // grad = None;

                let est_remain = epoch_start
                    .elapsed()
                    .unwrap()
                    .div((i + 1) as u32)
                    .mul((datasize / BATCH - i - 1) as u32);

                // let ao = actor.forward(img[0]);
                // let co = critic.forward(img[0], &ao);
                eprint!(
                    "{:8}/{datasize:8} Err: {} Remaining: {:?}  \r", // Result {ao:?} Score {co:?}
                    (i + 1) * BATCH,
                    acc_err / ((i + 1) * BATCH) as f32,
                    est_remain,
                );
            }
            for b in (&mut testc).array_chunks::<BATCH>() {
                let img = b.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
                let keys = b.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
                let acc_own = b.iter().map(|x| [x.2]).collect::<Vec<_>>();
                let acc = acc_own.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
                let (derr, _) = actor.acc_grad2::<BATCH>(&img, &keys, &acc);
                test_err += derr;
            }
            test_err /= testc.len() as f32;
        }
        // println!("{:?}", actor.net.1 .0 .0.bias);
        //grad = actor.backward(grad);
        acc_err /= datasize as f32;
        if prev_err < acc_err {
            actor.decay_lr(DECAY_RATE);
        }
        prev_err = acc_err;
        // epoch += 1;
        actor.clear_grad();
        eprintln!(
            "Epoch:{epoch:4} Error:      {acc_err} TestErr: {test_err} LR: {}                           ",
            actor.get_lr()
        );

        if !acc_err.is_nan() && acc_err.is_finite() && epoch % 1 == 0 {
            actor.save("ai-file/th2_actor.net").unwrap();
            // test_encode(outp.last().cloned());
        } else if acc_err.is_nan() || acc_err.is_infinite() {
            eprintln!("Get [NaN|inf]ed lol.");
            break;
        }
        if !SINGLE {
            outpa.reset();
            outpa.shuffle(&mut r);
            outpc.reset();
            outpc.shuffle(&mut r);
            testa.reset();
            testc.reset();
        }
    }
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
struct PlayOptions {
    control_shoot: bool,
    allow_bomb: bool,
    explore: f64,
}

impl Default for PlayOptions {
    fn default() -> Self {
        Self {
            control_shoot: true,
            allow_bomb: false,
            explore: 0.01,
        }
    }
}
fn gather_live_actor(games: usize, override_explore: Option<f64>, explore_step: Option<f64>) {
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");

    let num_list = arena::get_nums();
    let gameover = get_gameover();
    let life = get_life();
    let dia = get_dia();
    // let mut r = thread_rng();

    let ss = arena::screenshot(pos);
    ss.save("dbg_images/test_ss.bmp").unwrap();
    // let actor =
    // NetActor::load("ai-file/th2_actor.net").unwrap_or_else(|| NetActor::load_or_new(""));
    let q = NetQ::load_or_new("ai-file/th2_q.net");
    // let explore = std::fs::read_to_string("ai-file/explore")
    //     .expect("Could not find ai-file/explore")
    //     .trim()
    //     .parse()
    //     .expect("explore was not an f32");
    let mut settings: PlayOptions = serde_json::from_str(
        std::fs::read_to_string("ai-file/config")
            .expect("Config file not found")
            .trim(),
    )
    .unwrap_or_else(|e| {
        eprintln!("{e:?}");
        Default::default()
    });
    if let Some(f) = override_explore {
        settings.explore = f;
    }
    // println!("{settings:?}");
    // std::fs::write("ai-file/config", serde_json::to_string(&settings).unwrap()).unwrap();
    let mut r = thread_rng();

    for _ in 0..games {
        eprintln!("Initializing");

        let mut settings = settings.clone();
        // println!("{settings:?}");

        arena::start(pos, &num_list);
        let rn = format!("{}{}", r.next_u64(), r.next_u64());

        // eprintln!("Start playing now");

        let start = std::time::SystemTime::now();

        let mut score: isize = 0;
        // let mut lives = 0;

        let frame_time = 1.0 / 60.0;
        let mut frame_el;
        let mut frames = 0;
        let mut stats = vec![0.0; 8];
        loop {
            if wants_exit() {
                return;
            }
            // let mut s = String::new();
            frame_el = std::time::SystemTime::now();
            let img = arena::screenshot(pos);
            // s = s + &format!("ss {:?}\n", frame_el.elapsed().unwrap());
            stats[0] += frame_el.elapsed().unwrap().as_secs_f64();
            let pix = net::to_pixels(&img);
            // s = s + &format!("pix {:?}\n", frame_el.elapsed().unwrap());
            stats[1] += frame_el.elapsed().unwrap().as_secs_f64();
            // let img_pix = net::to_pixels(&img);

            //dbg!(&data.as_slice().unwrap()[0..15]);
            // let keys = actor.forward(&pix);
            let action = if r.gen::<f64>() < settings.explore {
                r.gen_range(0..32)
            } else {
                q.get_move(&pix)
            };
            // s = s + &format!("apply net {:?}\n", frame_el.elapsed().unwrap());
            stats[2] += frame_el.elapsed().unwrap().as_secs_f64();

            // let true_keys = from_flat(&collapse(&keys, settings.explore as f32));
            let true_keys = from_flat(&{
                let mut v = vec![0.0; 32];
                v[action] = 1.0;
                v
            });
            let mut key_b = true_keys.iter().map(|x| x == &1.0).collect::<Vec<_>>();
            if !settings.control_shoot {
                key_b[0] = true;
            }
            if !settings.allow_bomb {
                key_b[1] = false;
            }
            // s = s + &format!("conv keys {:?}\n", frame_el.elapsed().unwrap());
            do_dialog(&img, &dia);
            stats[3] += frame_el.elapsed().unwrap().as_secs_f64();

            let key_act: Vec<_> = key_b.iter().map(|x| if *x { 1.0 } else { 0.0 }).collect();
            let lost = has_lost(&img, &gameover);
            if !lost {
                thread::spawn(move || arena::do_keys(&key_b));
            } else {
                arena::do_keys(&[false; 6]);
            }
            // arena::do_keys(&key_b);
            // s = s + &format!("do keys {:?}\n", frame_el.elapsed().unwrap());
            stats[4] += frame_el.elapsed().unwrap().as_secs_f64();
            //let c = get_cuda_backend();
            //dbg!(&res);
            //res.forward();
            let lives = num_lives(&img, &life);
            print!("  {lives}  \r");
            let new_score = arena::get_score(&img, &num_list).unwrap_or(score) as isize
                + (lives - if lost { 0 } else { 1 }) * 50000;
            // s = s + &format!("get score {:?}\n", frame_el.elapsed().unwrap());
            stats[5] += frame_el.elapsed().unwrap().as_secs_f64();

            //eformat!("{:?}\n", new_score);
            let score_d = new_score - score;
            score = new_score;

            let name = format!("{}_{}", rn, frames);
            img.save(format!("images/actor_critic_data/{}.png", name))
                .unwrap();
            let data = serde_json::to_string(&(key_act, score_d as f32 / 100000.)).unwrap();
            std::fs::write(format!("images/actor_critic_data/{}.json", name), data).unwrap();
            stats[6] += frame_el.elapsed().unwrap().as_secs_f64();
            frames += 1;

            if lost {
                let gameover_img = get_pixel_range(&img, (147..271, 193..209));
                gameover_img.save("dbg_images/gameover box.png").unwrap();
                break;
            }
            // s = s + &format!("scan loss {:?}\n", frame_el.elapsed().unwrap());
            stats[7] += frame_el.elapsed().unwrap().as_secs_f64();
            if let Some(es) = explore_step {
                settings.explore += es;
                // print!("\r{}", settings.explore);
            }
            // println!("{s}");

            //let keys = res.iter().map(|x| *x>0.5).collect::<Vec<_>>();

            //arena::do_out(&keys);
            //outp.push(res);

            //eprint!("{:?}         \\r", arena::get_score(&img, &num_list[..]));
            while frame_el.elapsed().unwrap().as_secs_f32() < frame_time {
                thread::sleep(
                    Duration::from_secs_f32(frame_time).saturating_sub(frame_el.elapsed().unwrap()),
                )
            }
        }
        dbg!(start.elapsed().unwrap());
        std::fs::write(
            format!("images/actor_critic_data/{}.seq", rn),
            format!("{}", frames),
        )
        .unwrap();
        // for (_, time) in stats.iter().enumerate() {
        // println!("{}", time / (frames as f64));
        // }
        println!(
            "{} {}|",
            (frames as f32) / start.elapsed().unwrap().as_secs_f32(),
            score
        );
        // println!("Reset");
        reset_from_gameover();
    }
}
fn analyze_actor(bins: String) {
    let mut use_bins = bins == "true";
    let actor = NetActor::load("ai-file/th2_actor.net").expect("No actor found");
    let mut data = get_actor_iter();
    data.shuffle(&mut thread_rng());
    let mut bt_full: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    let mut bt_bin = BTreeMap::new();
    let bin_num = if let Ok(n) = bins.parse::<f32>() {
        use_bins = true;
        n
    } else {
        20.
    };
    let l = data.len();
    for (i, img) in (&mut data).enumerate() {
        let out = actor.forward(&img);
        if use_bins {
            let out_disc = out
                .into_iter()
                .map(|x| (x * bin_num) as u8)
                .collect::<Vec<_>>();
            match bt_bin.get_mut(&out_disc) {
                Some(v) => *v += 1,
                None => {
                    bt_bin.insert(out_disc, 1);
                }
            }
        } else {
            let out_disc = out
                .into_iter()
                .map(|x| unsafe { std::mem::transmute::<_, u32>(x) })
                .collect::<Vec<_>>();
            match bt_full.get_mut(&out_disc) {
                Some(v) => *v += 1,
                None => {
                    bt_full.insert(out_disc, 1);
                }
            }
        }
        print!("{i}/{l}\r");
        if i % 2000 == 0 {
            // println!("{bt:?}");
            if !use_bins {
                for (k, v) in bt_full.iter() {
                    print!(
                        "{:?}:{v},",
                        k.iter()
                            .map(|k| unsafe { std::mem::transmute::<u32, f32>(*k) })
                            .collect::<Vec<_>>()
                    );
                }
                println!();
            } else {
                println!("{bt_bin:?}");
            }
            println!();
        }
    }
    // println!("{bt:?}");
    for (k, v) in bt_full.iter() {
        print!(
            "{:?}:{v},",
            k.iter()
                .map(|k| unsafe { std::mem::transmute::<u32, f32>(*k) })
                .collect::<Vec<_>>()
        );
    }
    println!("{bt_bin:?}");
}

fn reset_lr() {
    // End critic at: 0.00000002
    std::fs::write("ai-file/th2_critic.net.lr", 0.00001.to_string()).unwrap();
    // End actor at:
    std::fs::write("ai-file/th2_actor.net.lr", 0.00001.to_string()).unwrap();
}

fn reset_crit() {
    let fd = std::fs::read_dir("images/actor_critic_data").unwrap();
    let num = std::fs::read_dir("images/actor_critic_data")
        .unwrap()
        .count();
    if num > 500000 {
        let mut sort = fd.collect::<Vec<_>>();
        sort.sort_unstable_by_key(|f| f.as_ref().unwrap().metadata().unwrap().modified().unwrap());
        let old = &sort[0..num - 500000];
        for f in old {
            let f = f.as_ref().unwrap();
            std::fs::remove_file(f.path()).unwrap();
        }
    }
}

fn do_loop() {
    const EPOCHS: usize = 1;
    const GAMES: usize = 1;
    // let mut step = 0.0001;
    loop {
        reset_crit();
        // todo!("Make partial reset to keep 500,000 images");
        let mut p = std::process::Command::new("wine")
            .arg("/home/opfromthestart/rust/game/touhou-ai/games/np2fmgen/np21nt.exe")
            .spawn()
            .unwrap();
        gather_live_actor(GAMES, None, None);
        // gather_live_actor(GAMES, Some(1.0), None);
        p.kill().unwrap();
        // process_crit(0.997);
        // step = 0.8 / ((get_q_iter().len() / GAMES) as f64);
        // train(Some(EPOCHS));
        // train_actor(true, Some(EPOCHS));
        train_q(Some(EPOCHS));
    }
}
fn train_q(limit: Option<usize>) {
    let mut q = NetQ::load_or_new("ai-file/th2_q.net"); //::<Backend<Cuda>>
    let limit = limit.unwrap_or(2000);

    let mut r = thread_rng();
    let (mut outp, mut test) = get_q_iter().split_test(16000);
    // let (mut outpa, mut testa) = get_actor_iter().split_test(16000);
    let datasize = DATASIZE.unwrap_or(outp.len());
    // let img = get_enc_batch(1, &mut r);
    // let mut outp: Vec<Vec<f32>> = vec![net::to_pixels(&img[0])];
    // let datasize = outp.len();
    let mut acc_err;
    let mut prev_err = f32::MIN;
    let mut test_err;

    // let (mut actor, critic, mut grad) = actor.pair_grad(critic);
    let mut grad;

    for epoch in 0..limit {
        // eprint!("Loading images\r");
        // if !SINGLE || epoch == 0 {
        //     outp = get_enc_iter();
        // }

        let epoch_start = std::time::SystemTime::now();

        acc_err = 0.0;

        // let samples = outpc.len() / BACK_BATCH;

        // if true {
        //     for (i, img) in outp
        //         .array_chunks::<BACK_BATCH>()
        //         .enumerate()
        //         .take(datasize / BACK_BATCH)
        //     {
        //         for imgb in img.array_chunks::<MEM_BATCH>() {
        //             let imgb = imgb.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
        //             (derr, grad) = actor.acc_grad::<MEM_BATCH>(&imgb, &critic, grad);
        //             acc_err += derr;
        //             // grad = net.backward(grad);
        //             // acc_err += net.train_batch::<MEM_BATCH>(imgb);
        //         }
        //         grad = actor.backward(grad);

        //         let est_remain = epoch_start
        //             .elapsed()
        //             .unwrap()
        //             .div((i + 1) as u32)
        //             .mul((datasize / BACK_BATCH - i - 1) as u32);

        //         // let ao = actor.forward(img[0]);
        //         // let co = critic.forward(img[0], &ao);
        //         eprint!(
        //             "{}/{datasize} Err: {} Remaining: {:?}  \r", // Result {ao:?} Score {co:?}
        //             (i + 1) * BACK_BATCH,
        //             acc_err / ((i + 1) * BACK_BATCH) as f32,
        //             est_remain,
        //         );
        //     }
        // }
        test_err = 0.0;
        for (i, img) in outp
            .array_chunks::<BATCH>()
            .enumerate()
            .take(datasize / BATCH)
        {
            let imgb = img.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
            let keys = img.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
            let scores_ = img.iter().map(|x| vec![x.2]).collect::<Vec<_>>();
            let scores = scores_.iter().map(|x| x as &[f32]).collect::<Vec<_>>();

            let imgnb = img
                .iter()
                .map(|x| x.3.as_ref().map(|a| a as &[f32]))
                .collect::<Vec<_>>();
            let (derr, grad2) = q.acc_grad::<BATCH>(&imgb, &keys, &scores, &imgnb, 0.995);
            grad = Some(grad2);
            acc_err += derr;
            // grad = net.backward(grad);
            // acc_err += net.train_batch::<MEM_BATCH>(imgb);

            q.backward(grad.unwrap());
            // grad = None;

            let est_remain = epoch_start
                .elapsed()
                .unwrap()
                .div((i + 1) as u32)
                .mul((datasize / BATCH - i - 1) as u32);

            // let ao = actor.forward(img[0]);
            // let co = critic.forward(img[0], &ao);
            eprint!(
                "{:8}/{datasize:8} Err: {} Remaining: {:?}  \r", // Result {ao:?} Score {co:?}
                (i + 1) * BATCH,
                acc_err / ((i + 1) * BATCH) as f32,
                est_remain,
            );
        }
        for b in (&mut test).array_chunks::<BATCH>() {
            let imgb = b.iter().map(|x| &x.0 as &[f32]).collect::<Vec<_>>();
            let keys = b.iter().map(|x| &x.1 as &[f32]).collect::<Vec<_>>();
            let scores_ = b.iter().map(|x| vec![x.2]).collect::<Vec<_>>();
            let scores = scores_.iter().map(|x| x as &[f32]).collect::<Vec<_>>();

            let imgnb = b
                .iter()
                .map(|x| x.3.as_ref().map(|a| a as &[f32]))
                .collect::<Vec<_>>();
            let (derr, _) = q.acc_grad::<BATCH>(&imgb, &keys, &scores, &imgnb, 0.995);
            test_err += derr;
        }
        test_err /= test.len() as f32;
        // println!("{:?}", actor.net.1 .0 .0.bias);
        //grad = actor.backward(grad);
        acc_err /= datasize as f32;
        if prev_err < acc_err {
            q.decay_lr(DECAY_RATE);
        }
        prev_err = acc_err;
        // epoch += 1;
        q.clear_grad();
        eprintln!(
            "Epoch:{epoch:4} Error:      {acc_err} TestErr: {test_err} LR: {}                           ",
            q.get_lr()
        );

        if !acc_err.is_nan() && acc_err.is_finite() && epoch % 1 == 0 {
            q.save("ai-file/th2_q.net").unwrap();
            // test_encode(outp.last().cloned());
        } else if acc_err.is_nan() || acc_err.is_infinite() {
            eprintln!("Get [NaN|inf]ed lol.");
            break;
        }
        if !SINGLE {
            outp.reset();
            outp.shuffle(&mut r);
            test.reset();
        }
    }
}
fn test_q() {
    let to_run = image::open("dbg_images/test img4.png").unwrap().to_rgb8();
    let to_run_pix = net::to_pixels(&to_run);
    let to_run4 = image::open("dbg_images/test img3.png").unwrap().to_rgb8();
    let to_run4_pix = net::to_pixels(&to_run4);
    let net = NetQ::load_or_new("ai-file/th2_q.net");
    let best_bove = net.get_move(&to_run_pix);
    eprint!("\t\t\t\t\t\t\tbest move nothing: {best_bove:?}");
    let best_bove = net.forward(&to_run4_pix);
    eprint!("best move stuff: {best_bove:?}");

    // let to_run = image::open("dbg_images/test img2.png").unwrap().to_rgb8();
    // let to_run_pix = net::to_pixels(&to_run);
    //eprintln!("{:?}", to_run_pix)eprintln!("making net");
    // let net = net::NetCrit::load_or_new("ai-file/th2_critic.net"); //::<Backend<Cuda>>
    // let conv = net.do_conv(&to_run_pix);
    // let t = conv
    //     .iter()
    //     .map(|x| x.iter().map(|x| x.abs()).sum::<f32>() / s)
    //     .collect::<Vec<_>>();
    // eprintln!("weight: {:?}", t);
    // conv.into_iter()
    //     .enumerate()
    //     .map(|(i, p)| (to_rb(&p, (33, 53), true), i))
    //     .for_each(|(x, i)| x.save(format!("dbg_images/post_conv{i}_2.png")).unwrap());

    let now = SystemTime::now();
    for _ in 0..1000 {
        // println!("{:?}", net.forward(&to_run_pix, &keys));
        // let start = Instant::now();
        net.get_move(&to_run_pix);
        // println!("{:?}", start.elapsed());
    }
    println!("net took {:?} for 1000", now.elapsed().unwrap());

    // drop(net);
    // let act = NetActor::load_or_new("ai-file/th2_actor.net");
    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     // println!("{:?}", net.forward(&to_run_pix, &keys));
    //     // let start = Instant::now();
    //     act.forward(&to_run_pix);
    //     // println!("{:?}", start.elapsed());
    // }
    // println!("actor took {:?} for 60", now.elapsed().unwrap());

    // let latent = net
    //     .do_conv(&to_run_pix)
    //     .into_iter()
    //     .flatten()
    //     .collect::<Vec<_>>();
    // let now = SystemTime::now();
    // for _ in 0..60 {
    //     let lat = net.do_conv(&to_run_pix);
    //     let out = net.from_conv(&lat.into_iter().flatten().collect::<Vec<_>>(), &keys);
    //     // println!("{out:?}");
    // }
    // println!("Took {:?} for 60 decomp", now.elapsed().unwrap());

    // let score = net.forward(&to_run_pix, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    // eprintln!("Score: {score:?}");

    // let bytes = std::fs::read("ai-file/1.0.bias").unwrap()[74..].to_vec();
    // let l = bytes.len()/4;
    // let floats : Vec<f32> = unsafe {
    //     std::mem::transmute(bytes)
    // };
    // eprintln!("{:?}", &floats[..l]);
    net.clear_grad();
}

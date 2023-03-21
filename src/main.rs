#![feature(generic_const_exprs)]

use std::{thread, time::Duration, ops::{Div, Mul}};

use clap::Parser;
use net::to_rb;
use rand::{seq::SliceRandom, thread_rng};

//use coaster::{Backend, Cuda, frameworks::cuda::get_cuda_backend};
//use ndarray::Dim;

mod arena;
mod net;

#[derive(Parser, Debug)]
enum Mode {
    Eval,
    Train,
}

fn main() {
    let m = Mode::parse();

    match m {
        Mode::Eval => test_img(),
        Mode::Train => train(),
    }
    //test_img();
}

fn train() {

    eprintln!("Open the game within 2 seconds of starting this.");
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");
    
    let num_list = arena::get_nums();

    let ss = arena::screenshot(pos);
    ss.save("images/test_ss.bmp").unwrap();

    eprintln!("making net");
    let mut net = net::NetCrit::load_or_new("ai-file/th2_critic.net"); //::<Backend<Cuda>>
    eprintln!("Initializing");

    arena::start(pos, &num_list);

    eprintln!("Start playing now");

    let start = std::time::SystemTime::now();

    let mut outp = vec![];
    let mut score = 0;
    loop {
        let img = arena::screenshot(pos);
        let img_pix = net::to_pixels(&img);

        //dbg!(&data.as_slice().unwrap()[0..15]);
        let keys = arena::get_keys().map(|x| if x {
            1.0
        }
        else {
            0.0
        }).to_vec();
        //let c = get_cuda_backend();
        //dbg!(&res);
        //res.forward();
        let new_score = arena::get_score(&img, &num_list).unwrap_or(score);
        
        //eprintln!("{:?}", new_score);
        let score_d = new_score - score;
        score = new_score;

        outp.push((img_pix, keys, (score_d as f32)/1_000.));

        // TEST
        if outp.len() > 4000 {
            break;
        }

            //let keys = res.iter().map(|x| *x>0.5).collect::<Vec<_>>();

        //arena::do_out(&keys);
        //outp.push(res);
        
        //eprint!("{:?}         \\r", arena::get_score(&img, &num_list[..]));
    }
    dbg!(start.elapsed().unwrap());

    let gamma = 0.98;
    let mut acc_err = 1.0;
    let mut epoch = 0;
    let target_err = 0.03f32.powi(2);

    let mut grad = net.grad();
    let mut derr;

    let mut img_vec = vec![];
    let mut key_vec = vec![];
    let mut out_vec = vec![];
    {
        let mut acc = 0.0;
        for (img,key,sc) in outp.into_iter().rev() {
            out_vec.push(acc);
            img_vec.push(img);
            key_vec.push(key.into_iter().map(|x| x as f32).collect::<Vec<_>>());
            acc += sc;
            acc *= gamma;
        }
    }

    const BATCH: usize = 4;

    let mut r = thread_rng();
    let mut perm = (0..out_vec.len()).collect::<Vec<_>>();

    while acc_err > target_err {
        perm.shuffle(&mut r);

        permute(&mut img_vec, &perm);
        permute(&mut key_vec, &perm);
        permute(&mut out_vec, &perm);

        let epoch_start = std::time::SystemTime::now();

        acc_err = 0.0;

        let samples = out_vec.len()/BATCH;

        for (i, ((img, keys), acc)) in img_vec.chunks_exact(BATCH)
        .zip(key_vec.chunks_exact(BATCH))
        .zip(out_vec.chunks_exact(BATCH)).enumerate() {
            let img = img.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
            let keys = keys.iter().map(|x| x as &[f32]).collect::<Vec<_>>();
            (derr, grad) = net.train_grad::<BATCH>(&img, &keys, &[acc], grad);
            acc_err += derr;
            let est_remain = epoch_start.elapsed().unwrap().div((i+1) as u32).mul((samples - i-1) as u32);
            eprint!("{}/{samples} Err: {} Remaining: {:?}    \r",i+1, acc_err/((i+1)*BATCH) as f32, est_remain);
        }
        //grad = net.backward(grad);
        acc_err /= out_vec.len() as f32;
        epoch += 1;
        eprintln!("Epoch:{epoch} Error: {acc_err}                            ");

        if !acc_err.is_nan() && acc_err.is_finite() {
            net.save("ai-file/th2_critic.net").unwrap();
        }
        else {
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
fn permute<T>(v: &mut [T], p: &[usize]) {
    for i in p {
        v.swap(0, *i);
    }
}

fn test_img() {

    let to_run = image::open("images/test img.png").unwrap().to_rgb8();
    let to_run_pix = net::to_pixels(&to_run);
    //eprintln!("{:?}", to_run_pix)eprintln!("making net");
    let net = net::NetCrit::load_or_new("ai-file/th2_critic.net"); //::<Backend<Cuda>>
    let conv = net.do_conv(&to_run_pix);
    let s = conv[0].len() as f32;
    let t = conv.iter().map(|x| x.iter().map(|x| x.abs()).sum::<f32>()/s).collect::<Vec<_>>();
    eprintln!("weight: {:?}", t);
    conv.into_iter().enumerate()
    .map(|(i,p)| (to_rb(&p, (76, 46), true), i))
    .for_each(|(x, i)| x.save(format!("images/post_conv{i}.png")).unwrap());
    let score = net.forward(&to_run_pix, &[1.0,0.0,0.0,1.0,0.0,0.0]);
    eprintln!("Score: {score:?}");

    let to_run = image::open("images/test img2.png").unwrap().to_rgb8();
    let to_run_pix = net::to_pixels(&to_run);
    //eprintln!("{:?}", to_run_pix)eprintln!("making net");
    let net = net::NetCrit::load_or_new("ai-file/th2_critic.net"); //::<Backend<Cuda>>
    let conv = net.do_conv(&to_run_pix);
    let t = conv.iter().map(|x| x.iter().map(|x| x.abs()).sum::<f32>()/s).collect::<Vec<_>>();
    eprintln!("weight: {:?}", t);
    conv.into_iter().enumerate()
    .map(|(i,p)| (to_rb(&p, (76, 46), true), i))
    .for_each(|(x, i)| x.save(format!("images/post_conv{i}_2.png")).unwrap());
    let score = net.forward(&to_run_pix, &[1.0,0.0,0.0,1.0,0.0,0.0]);
    eprintln!("Score: {score:?}");

    // let bytes = std::fs::read("ai-file/1.0.bias").unwrap()[74..].to_vec();
    // let l = bytes.len()/4;
    // let floats : Vec<f32> = unsafe {
    //     std::mem::transmute(bytes)
    // };
    // eprintln!("{:?}", &floats[..l]);
}

#[test]
fn does_permute() {
    let perm = vec![0,2,1,4,3];
    let mut v1 = vec![12, 23, 34, 45, 56];
    let mut v2 = v1.clone();
    permute(&mut v1, &perm);
    permute(&mut v2, &perm);
    assert_eq!(&v1, &v2);
}
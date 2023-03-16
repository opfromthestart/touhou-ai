#![feature(generic_const_exprs)]

use std::{thread, time::Duration};

//use coaster::{Backend, Cuda, frameworks::cuda::get_cuda_backend};
//use ndarray::Dim;

mod arena;
mod net;

fn main() {

    eprintln!("Open the game within 2 seconds of starting this.");
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");
    
    let num_list = arena::get_nums();

    let ss = arena::screenshot(pos);
    ss.save("test_ss.bmp").unwrap();

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
        let img_pix = arena::to_pixels(&img);

        //dbg!(&data.as_slice().unwrap()[0..15]);
        let keys = arena::get_keys().map(|x| if x {
            1.0
        }
        else {
            0.0
        }).to_vec();
        let res = &net.forward(&img_pix, &keys)[0];
        //let c = get_cuda_backend();
        //dbg!(&res);
        //res.forward();
        let new_score = arena::get_score(&img, &num_list);
        
        if let Some(new_score) = new_score {
            //eprintln!("{:?}", new_score);
            let score_d = new_score - score;
            score = new_score;

            outp.push((img_pix, keys, res.clone(), (score_d as f32)/100_000.));

            // TEST
            if outp.len() > 20 {
                break;
            }

            //let keys = res.iter().map(|x| *x>0.5).collect::<Vec<_>>();

            //arena::do_out(&keys);
        }
        else {
            break;
        }
        //outp.push(res);
        
        //eprint!("{:?}         \\r", arena::get_score(&img, &num_list[..]));
    }
    dbg!(start.elapsed().unwrap());

    let gamma = 0.98;
    let mut acc;
    let mut acc_err = 1.0;
    let mut epoch = 0;
    let target_err = 0.001 / outp.len() as f32;

    while acc_err > target_err {
        acc_err = 0.0;
        acc = 0.0;
        for (data, keys, pred, sc) in outp.iter().rev() {
            eprintln!("{acc} {pred}");
            acc_err += (acc-pred).powi(2);
            net.train(data, keys, &[acc]);
            acc += sc;
            acc *= gamma;
        }
        acc_err /= outp.len() as f32;
        epoch += 1;
        eprintln!("Error: {acc_err}");

        if epoch%5==0 {
            net.save("ai-file/th2_critic.net").unwrap();
        }
    }

    net.save("ai-file/th2_critic.net").unwrap();
    
    //eprintln!("{:?}", start.elapsed());

    //eprintln!("{:?}", arena::get_score(images.last().unwrap(), &num_list[..]));
    //fs::write("win_ss.bmp", images.last().unwrap().buffer()).unwrap();
}

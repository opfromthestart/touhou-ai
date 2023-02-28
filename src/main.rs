use std::{thread, time::Duration};

use coaster::{Backend, Cuda};
use image::EncodableLayout;
use screenshots::Screen;

use crate::{arena::split_channel, net::{to_netdata, from_netdata}};

mod arena;
mod net;

fn main() {

    eprintln!("Open the game within 2 seconds of starting this.");
    thread::sleep(Duration::from_millis(2000));

    let pos = arena::get_active_window_pos();

    eprintln!("{pos:?}");
    
    let num_list = arena::get_nums();

    dbg!("making net");
    let mut net = net::Net::<Backend<Cuda>>::load_or_new("ai-file/th2_critic.net");
    dbg!("MAde net");

    let start = std::time::SystemTime::now();
    let mut outp = vec![];
    let mut score = 0;
    loop {
        let img = arena::screenshot(pos);
        let data = to_netdata(& img.as_bytes().iter().map(|x| *x as f32/256.).collect::<Vec<_>>());

        //dbg!(&data.as_slice().unwrap()[0..15]);
        let res = from_netdata(&net.forward(&[data.clone()])[0]);
        //dbg!(&res);
        //res.forward();
        let new_score = arena::get_score(&img, &num_list);
        if let Some(new_score) = new_score {
            let score_d = new_score - score;
            score = new_score;

            outp.push((data, res.clone(), score_d));

            //let keys = res.iter().map(|x| *x>0.5).collect::<Vec<_>>();

            //arena::do_out(&keys);
        }
        else {
            break;
        }
        //outp.push(res);
        
        //eprint!("{:?}         \r", arena::get_score(&img, &num_list[..]));
    }
    dbg!(start.elapsed().unwrap());

    let gamma = 0.98;
    let mut acc = 0.0;
    for (data, pred, sc) in outp.iter().rev() {
        acc += *sc as f32;
        net.train(&[data.to_owned()], &[to_netdata(&[acc-pred[0]])]);
        acc *= gamma;
    }

    net.save("ai-file/th2_critic.net").unwrap();
    
    //eprintln!("{:?}", start.elapsed());

    //eprintln!("{:?}", arena::get_score(images.last().unwrap(), &num_list[..]));
    //fs::write("win_ss.bmp", images.last().unwrap().buffer()).unwrap();
}

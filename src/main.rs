use std::{thread, time::Duration, fs, ops::Range};

use image::{ImageBuffer, Pixel, Luma, buffer::ConvertBuffer};
use screenshots::Screen;
use show_image::{ImageView, ImageInfo, PixelFormat, create_window};

type Image = ImageBuffer<Luma<u8>, Vec<u8>>; 

fn get_pixel_range(img: &Image, area: (Range<u32>, Range<u32>)) -> Image {
    let mut ret = image::ImageBuffer::new(area.0.len() as u32, area.1.len() as u32);
    for x in area.0.clone() {
        for y in area.1.clone() {
            ret.put_pixel(x-area.0.start, y-area.1.start, *(img.get_pixel(x, y)));
        }
    }
    ret
}

fn image_to_u8(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> Image {
    let mut ret_img : Image = image::ImageBuffer::new(img.width(), img.height());
    for x in 0..img.width() {
        for y in 0..img.height() {
            ret_img.put_pixel(x, y, Luma::from([(img.get_pixel(x, y).0[0]*256.0) as u8]));
        }
    }
    ret_img
}

fn get_score(img: &Image, nums: &[Image]) -> [u8;8] {
    let mut score = [0;8];

    let score_img = get_pixel_range(img, (449..577, 97..113));
    score_img.save("score check/score.png").unwrap();
    for (i,n) in nums.iter().enumerate() {
        let matching = imageproc::template_matching::match_template(&score_img, n, imageproc::template_matching::MatchTemplateMethod::SumOfSquaredErrorsNormalized);
        //let ext = imageproc::template_matching::find_extremes(&matching);
        
        image_to_u8(&matching).save(format!("score check/{i}.png")).unwrap();
        //eprintln!("{ext:?}");
        for s in 0..8 {
            if matching.get_pixel((s*16) as u32, 0).0[0] < 0.0024 { // For some reason 6 doesnt match pixel perfect, this is zero for all others.
                score[s] = i as u8;
            }
        }
    }

    score
}

fn main() {
    let (c, screen) = xcb::Connection::connect(None).unwrap();

    thread::sleep(Duration::from_millis(1200));
    let active_cookie = c.send_request(&xcb::x::GetInputFocus {});
    let active = c.wait_for_reply(active_cookie).unwrap();

    //let name = get_window_name(d, active);

    let mut window = active.focus();
    let mut pos = (1, 20);

    loop {
        let info_cookie = c.send_request(&xcb::x::GetGeometry {
            drawable: xcb::x::Drawable::Window(window),
        });
        let info = c.wait_for_reply(info_cookie).unwrap();

        pos.0 += info.x() as i32;
        pos.1 += info.y() as i32;

        let tree_cookie = c.send_request(&xcb::x::QueryTree { window });
        let tree = c.wait_for_reply(tree_cookie).unwrap();
        if tree.root() == window {
            break;
        }
        window = tree.parent();
    }
    eprintln!("{pos:?}");
    
    let start = std::time::SystemTime::now();
    let screens = Screen::all().unwrap();
    eprintln!("Screens: {}", screens.len());
    let mut images = vec![];
    for i in 0..1 {
        let ss = screens[0].capture_area(pos.0, pos.1, 641, 401).unwrap();
        let mut img = image::load_from_memory(ss.buffer()).unwrap().to_luma8();
        images.push(img);
    }
    eprintln!("{:?}", start.elapsed());

    images.last().unwrap().save_with_format("other_ss.bmp", image::ImageFormat::Bmp).unwrap();

    let font = image::open("th2 font.png").unwrap().to_luma8();
    let nums = get_pixel_range(&font, (0..160, 32..48));
    let num_list = (0..10).into_iter().map(|x| {
        get_pixel_range(&nums, ((16*x)..(16*x+16), 0..16))
    }).collect::<Vec<_>>();

    eprintln!("{:?}", get_score(images.last().unwrap(), &num_list[..]));
    //fs::write("win_ss.bmp", images.last().unwrap().buffer()).unwrap();
}

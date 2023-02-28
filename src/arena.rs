// Everything to setup the training, eg the arena.

use std::ops::Range;

use enigo::{KeyboardControllable, Key};
use image::{ImageBuffer, Rgb, Luma};
use imageproc::template_matching::{match_template, MatchTemplateMethod};

type Image = ImageBuffer<Rgb<u8>, Vec<u8>>; 

pub(crate) fn get_active_window_pos() -> (i32, i32) {
    let (c, _screen) = xcb::Connection::connect(None).unwrap();

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

        let name = c.wait_for_reply(c.send_request(&xcb::x::GetProperty{
            delete: false,
            window,
            property: xcb::x::ATOM_WM_NAME,
            r#type: xcb::x::ATOM_STRING,
            long_offset: 0,
            long_length: 0,
        })).unwrap();
        eprint!("{} / ", String::from_utf8(name.value::<u8>().to_vec()).unwrap());
        // TODO find why this doesnt work

        let tree_cookie = c.send_request(&xcb::x::QueryTree { window });
        let tree = c.wait_for_reply(tree_cookie).unwrap();
        if tree.root() == window {
            break;
        }
        window = tree.parent();
    }

    pos
}

pub(crate) fn screenshot(pos: (i32, i32)) -> Image {
    let ss = screenshots::Screen::from_point(pos.0, pos.1).unwrap().capture_area(pos.0, pos.1, 641, 401).unwrap();
    image::load_from_memory(ss.buffer()).unwrap().to_rgb8()
}

pub(crate) fn get_nums() -> Vec<[ImageBuffer<Luma<u8>, Vec<u8>>; 3]> {
    let font = image::open("th2 font.png").unwrap().to_rgb8();
    let nums = get_pixel_range(&font, (0..160, 32..48));
    (0..10).into_iter().map(|x| {
        get_pixel_range(&nums, ((16*x)..(16*x+16), 0..16))
    }).map(|x| split_channel(&x)).collect::<Vec<_>>()
}

pub(crate) fn get_pixel_range(img: &Image, area: (Range<u32>, Range<u32>)) -> Image {
    let mut ret = image::ImageBuffer::new(area.0.len() as u32, area.1.len() as u32);
    for x in area.0.clone() {
        for y in area.1.clone() {
            ret.put_pixel(x-area.0.start, y-area.1.start, *(img.get_pixel(x, y)));
        }
    }
    ret
}

pub(crate) fn image_to_u8(img: &ImageBuffer<Rgb<f32>, Vec<f32>>) -> Image {
    let mut ret_img : Image = image::ImageBuffer::new(img.width(), img.height());
    for x in 0..img.width() {
        for y in 0..img.height() {
            ret_img.put_pixel(x, y, Rgb::from(img.get_pixel(x, y).0.map(|x| (x*256.0) as u8)));
        }
    }
    ret_img
}

pub(crate) fn split_channel(img: &Image) -> [ImageBuffer<Luma<u8>, Vec<u8>>; 3] {
    let mut ret = [ImageBuffer::new(img.width(), img.height()), ImageBuffer::new(img.width(), img.height()), ImageBuffer::new(img.width(), img.height()), ];
    let [r,g,b] = &mut ret;
    for (((p, r), g), b) in img.pixels().zip(r.iter_mut()).zip(g.iter_mut()).zip(b.iter_mut()) {
        [*r, *g, *b] = p.0;
    }
    ret
}

pub(crate) fn get_score(img: &Image, nums: &[[ImageBuffer<Luma<u8>, Vec<u8>>;3]]) -> Option<u32> {
    let mut score = [20;8];

    let score_img = get_pixel_range(img, (449..577, 97..113));
    let [r,g,b] = split_channel(&score_img);
    //score_img.save("score check/score.png").unwrap();
    for (i,n) in nums.iter().enumerate() {
        let matching_r = match_template(&r, &n[0], MatchTemplateMethod::SumOfSquaredErrorsNormalized);
        let matching_g = match_template(&g, &n[1], MatchTemplateMethod::SumOfSquaredErrorsNormalized);
        let matching_b = match_template(&b, &n[2], MatchTemplateMethod::SumOfSquaredErrorsNormalized);
        let matching = matching_r.pixels().zip(matching_g.pixels()).zip(matching_b.pixels()).map(|((r,g),b)| r.0[0]+b.0[0]+g.0[0]).collect::<Vec<f32>>();
        //let ext = matching.iter().copied().reduce(|x,y| x.min(y)).unwrap();
        
        //image_to_u8(&matching).save(format!("score check/{i}.png")).unwrap();
        //eprintln!("{ext:?}");
        for s in 0..8 {
            if matching[s*16] < 0.024 { // For some reason 6 doesnt match pixel perfect, this is zero for all others.
                score[s] = i as u8;
            }
        }
    }

    if score.iter().find(|x| **x==20).is_some() {
        None
    }
    else {
        Some(score.iter().rev().enumerate().rev().map(|(i,v)| 10u32.pow(i as u32)* *v as u32).sum())
    }
}

pub(crate) fn do_out(buttons: &[bool]) {
    let mut e = enigo::Enigo::new();
    if buttons[0] {
        e.key_click(Key::Layout('z'));
    }
    if buttons[1] {
        e.key_click(Key::Layout('x'));
    }
    if buttons[2] {
        e.key_click(Key::UpArrow);
    }
    if buttons[3] {
        e.key_click(Key::DownArrow);
    }
    if buttons[4] {
        e.key_click(Key::LeftArrow);
    }
    if buttons[5] {
        e.key_click(Key::RightArrow);
    }
}
// Everything to setup the training, eg the arena.

use std::{ops::Range, fs};

use device_query::Keycode;
use enigo::{KeyboardControllable, Key};
use image::{ImageBuffer, Rgb, Luma, GenericImageView};
use rand::{thread_rng, seq::SliceRandom, Rng};

pub(crate) type Image = ImageBuffer<Rgb<u8>, Vec<u8>>; 
pub(crate) type GrayImage = ImageBuffer<Luma<u8>, Vec<u8>>;

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

pub(crate) fn get_nums() -> Vec<Image> {
    let font = image::open("images/th2 font.png").unwrap().to_rgb8();
    let nums = get_pixel_range(&font, (0..160, 32..48));
    (0..10).into_iter().map(|x| {
        get_pixel_range(&nums, ((16*x)..(16*x+16), 0..16))
    }).collect::<Vec<_>>()
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

pub(crate) fn split_channel(img: &Image) -> [GrayImage; 3] {
    let mut ret = [ImageBuffer::new(img.width(), img.height()), ImageBuffer::new(img.width(), img.height()), ImageBuffer::new(img.width(), img.height()), ];
    let [r,g,b] = &mut ret;
    for (((p, r), g), b) in img.pixels().zip(r.iter_mut()).zip(g.iter_mut()).zip(b.iter_mut()) {
        [*r, *g, *b] = p.0;
    }
    ret
}

pub(crate) fn join_channel(img: &[GrayImage; 3]) -> Image {
    let mut ret: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(img[0].width(), img[0].height());
    let [r,g,b] = &img;
    for (((p, r), g), b) in ret.pixels_mut().zip(r.iter()).zip(g.iter()).zip(b.iter()) {
        p.0 = [*r, *g, *b];
    }
    ret
}

pub(crate) fn match_template(
    image: &Image,
    template: &Image,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(
        image_width >= template_width,
        "image width must be greater than or equal to template width"
    );
    assert!(
        image_height >= template_height,
        "image height must be greater than or equal to template height"
    );
    
    let mut result = ImageBuffer::new(
        image_width - template_width + 1,
        image_height - template_height + 1,
    );

    for y in 0..result.height() {
        for x in 0..result.width() {
            let mut score = 0f32;

            for dy in 0..template_height {
                for dx in 0..template_width {
                    let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy).0};
                    let template_value = unsafe { template.unsafe_get_pixel(dx, dy).0};

                    score += image_value.iter().zip(template_value.iter()).map(|(i,t)| (*i as f32 - *t as f32).powf(2.0)).sum::<f32>();
                }
            }

            result.put_pixel(x, y, Luma([score]));
        }
    }

    result
}

pub(crate) fn img_diff(
    image1: &Image,
    image2: &Image,
) -> Image {
    let (width1, height1) = image1.dimensions();
    let (width2, height2) = image2.dimensions();

    assert!(
        width1 == width2,
        "image widths must be equal"
    );
    assert!(
        height1 == height2,
        "image heights must be equal"
    );
    
    let mut result = ImageBuffer::new(
        width1,
        height1,
    );

    for y in 0..result.height() {
        for x in 0..result.width() {

            let image_value = unsafe { image1.unsafe_get_pixel(x, y ).0};
            let template_value = unsafe { image2.unsafe_get_pixel(x, y).0};

            let score = image_value.iter().zip(template_value.iter()).map(|(i,t)| i-t).collect::<Vec<_>>();
            let score = [score[0],score[1],score[2]];

            result.put_pixel(x, y, Rgb(score));
        }
    }

    result
}

//pub(crate) fn to_pixels(img: &Image) -> Vec<f32> {
//    img.as_bytes().iter().map(|x| *x as f32/256.).collect::<Vec<_>>()
//}

pub(crate) fn get_score(img: &Image, nums: &[Image]) -> Option<u32> {
    let mut score = [20;8];

    let score_img = get_pixel_range(img, (449..577, 97..113));
    //score_img.save("score check/score.png").unwrap();
    for (i,n) in nums.iter().enumerate() {
        let matching = match_template(&score_img, n).pixels().map(|p| p.0[0]).collect::<Vec<_>>();

        //let ext = matching.iter().copied().reduce(|x,y| x.min(y)).unwrap();
        
        //image_to_u8(&matching).save(format!("score check/{i}.png")).unwrap();
        //eprintln!("{i}: {ext:?}");

        //if i == 6 {
        //    get_pixel_range(&score_img, (64..80, 0..16)).save("6 ss.png").unwrap();
        //    n.save("6 temp.png").unwrap();
        //    img_diff(n, &get_pixel_range(&score_img, (64..80, 0..16))).save("6 comp.png").unwrap();
        //}
        //eprintln!("{}", matching.len());
        for s in 0..8 {
            if matching[s*16] == 0.0 { // For some reason 6 doesnt match pixel perfect, this is zero for all others.
                score[s] = i as u8;
            }
        }
    }

    //eprintln!("{score:?}");

    if score.iter().find(|x| **x==20).is_some() {
        None
    }
    else {
        Some(score.iter().rev().enumerate().rev().map(|(i,v)| 10u32.pow(i as u32)* *v as u32).sum())
    }
}

pub(crate) fn do_keys(buttons: &[bool]) {
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

pub(crate) fn get_keys() -> [bool;6] {
    let d = device_query::DeviceState::new();
    let keys = d.query_keymap();
    [Keycode::Z, Keycode::X, Keycode::Up, Keycode::Down, Keycode::Left, Keycode::Right]
    .map(|v| keys.contains(&v))
}

pub(crate) fn wants_exit() -> bool {
    let d = device_query::DeviceState::new();
    let keys = d.query_keymap();
    keys.contains(&Keycode::Dot) || keys.contains(&Keycode::Escape)
}

#[derive(PartialEq, Eq)]
enum PauseState {
    Return,
    Quit,
    Kidding,
    Really,
}

#[derive(PartialEq, Eq)]
enum Mode {
    Start,
    Demo,
    Select,
    Game,
    Paused(PauseState),
    GameOver(bool),
    Continue(bool),
    Between,
}

fn get_mode(pos: (i32, i32), nums: &[Image]) -> Mode {
    let ss = screenshot(pos);

    if ss.get_pixel(420, 69).0 == [119, 119, 153] { // Checks a background pixel for blueish color
        Mode::Select
    }
    else if ss.get_pixel(420, 69).0 == [255, 255, 0] { // Checks a text pixel for yellow color
        Mode::Start
    }
    else if (244..=254).all(|y| ss.get_pixel(202, y).0 == [255,255,255]) { // Checks the vertical line of the R in Return
        Mode::Paused(PauseState::Return)
    }
    else if (260..=269).all(|y| ss.get_pixel(236, y).0 == [255,255,255]) { // Checks the vertical line of the I in I was just kidding. Sorry.
        Mode::Paused(PauseState::Quit)
    }
    else if (244..=254).all(|y| ss.get_pixel(141, y).0 == [255,255,255]) { // Checks the vertical line of the I in Yes, I'll quit.
        Mode::Paused(PauseState::Kidding)
    }
    else if (260..=270).all(|y| ss.get_pixel(213, y).0 == [255,255,255]) { // Checks the vertial line of the t in Quit
        Mode::Paused(PauseState::Really)
    }
    else if get_score(&ss, nums).is_some() {
        Mode::Game
    }
    else {
        Mode::Between
    }
}

pub(crate) fn start(pos: (i32, i32), nums: &[Image]) {
    let mut e = enigo::Enigo::new();
    while get_mode(pos, nums) == Mode::Game {
        e.key_click(Key::Layout('z'));
    }
    while get_mode(pos, nums) != Mode::Game {
        e.key_click(Key::Layout('z'));
    }
}

pub(crate) fn reset(pos: (i32, i32), nums: &[Image]) {
    let mut e = enigo::Enigo::new();
    //e.key_click(Key::Escape)

    start(pos, nums);
}

pub(crate) fn get_enc_batch<R: Rng>(n: usize, r: &mut R) -> Vec<Image> {
    let files = fs::read_dir("images/encoder_data").unwrap();
    let to_ret : Vec<_> = files.filter_map(|x| {
        let y = x.unwrap();
        if y.file_type().unwrap().is_file() {
            Some(y.path())
        }
        else {
            None
        }
    }).collect();
    to_ret.choose_multiple(r, n).into_iter().map(|f| image::open(f).unwrap().into_rgb8()).collect()
}
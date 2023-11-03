// Everything to setup the training, eg the arena.

use std::{fs, ops::Range, sync::Mutex, thread, time::Duration};

use device_query::Keycode;
use enigo::{Key, KeyboardControllable};
use image::{GenericImageView, ImageBuffer, Luma, Pixel, Rgb, Rgba};
use once_cell::sync::Lazy;
use rand::{seq::SliceRandom, Rng};

use crate::net::{to_flat, to_pixels};

pub(crate) type Image = ImageBuffer<Rgb<u8>, Vec<u8>>;
pub(crate) type ImageA = ImageBuffer<Rgba<u8>, Vec<u8>>;
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

        let name = c
            .wait_for_reply(c.send_request(&xcb::x::GetProperty {
                delete: false,
                window,
                property: xcb::x::ATOM_WM_NAME,
                r#type: xcb::x::ATOM_STRING,
                long_offset: 0,
                long_length: 0,
            }))
            .unwrap();
        eprint!(
            "{} / ",
            String::from_utf8(name.value::<u8>().to_vec()).unwrap()
        );
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
    let ss = screenshots::Screen::from_point(pos.0, pos.1)
        .unwrap()
        .capture_area(pos.0, pos.1, 641, 401)
        .unwrap();
    image::load_from_memory(ss.buffer()).unwrap().to_rgb8()
}

pub(crate) fn get_nums() -> Vec<ImageA> {
    let font = image::open("images/th2 font.png").unwrap().to_rgba8();
    let nums = get_pixel_range(&font, (0..160, 32..48));
    (0..10)
        .into_iter()
        .map(|x| get_pixel_range(&nums, ((16 * x)..(16 * x + 16), 0..16)))
        .collect::<Vec<_>>()
}

pub(crate) fn get_gameover() -> ImageA {
    image::open("images/gameover.png").unwrap().to_rgba8()
}
pub(crate) fn get_life() -> ImageA {
    image::open("images/life.png").unwrap().to_rgba8()
}
pub(crate) fn get_dia() -> ImageA {
    image::open("images/text_box.png").unwrap().to_rgba8()
}

pub(crate) fn get_pixel_range<P: Pixel<Subpixel = u8>>(
    img: &ImageBuffer<P, Vec<u8>>,
    area: (Range<u32>, Range<u32>),
) -> ImageBuffer<P, Vec<u8>> {
    let mut ret = image::ImageBuffer::new(area.0.len() as u32, area.1.len() as u32);
    for x in area.0.clone() {
        for y in area.1.clone() {
            ret.put_pixel(x - area.0.start, y - area.1.start, *(img.get_pixel(x, y)));
        }
    }
    ret
}

#[allow(dead_code)]
pub(crate) fn image_to_u8(img: &ImageBuffer<Rgb<f32>, Vec<f32>>) -> Image {
    let mut ret_img: Image = image::ImageBuffer::new(img.width(), img.height());
    for x in 0..img.width() {
        for y in 0..img.height() {
            ret_img.put_pixel(
                x,
                y,
                Rgb::from(img.get_pixel(x, y).0.map(|x| (x * 256.0) as u8)),
            );
        }
    }
    ret_img
}

pub(crate) fn split_channel(img: &Image) -> [GrayImage; 3] {
    let mut ret = [
        ImageBuffer::new(img.width(), img.height()),
        ImageBuffer::new(img.width(), img.height()),
        ImageBuffer::new(img.width(), img.height()),
    ];
    let [r, g, b] = &mut ret;
    for (((p, r), g), b) in img
        .pixels()
        .zip(r.iter_mut())
        .zip(g.iter_mut())
        .zip(b.iter_mut())
    {
        [*r, *g, *b] = p.0;
    }
    ret
}

pub(crate) fn join_channel(img: &[GrayImage; 3]) -> Image {
    let mut ret: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(img[0].width(), img[0].height());
    let [r, g, b] = &img;
    for (((p, r), g), b) in ret.pixels_mut().zip(r.iter()).zip(g.iter()).zip(b.iter()) {
        p.0 = [*r, *g, *b];
    }
    ret
}

pub(crate) fn match_template(image: &Image, template: &ImageA) -> ImageBuffer<Luma<f32>, Vec<f32>> {
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
                    let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy).0 };
                    let template_value = unsafe { template.unsafe_get_pixel(dx, dy).0 };

                    score += image_value
                        .iter()
                        .zip(template_value.iter())
                        .map(|(i, t)| (*i as f32 - *t as f32).powf(2.0))
                        .sum::<f32>()
                        * (template_value[3] as f32);
                }
            }

            result.put_pixel(x, y, Luma([score]));
        }
    }

    result
}

#[allow(dead_code)]
pub(crate) fn img_diff(image1: &Image, image2: &Image) -> Image {
    let (width1, height1) = image1.dimensions();
    let (width2, height2) = image2.dimensions();

    assert!(width1 == width2, "image widths must be equal");
    assert!(height1 == height2, "image heights must be equal");

    let mut result = ImageBuffer::new(width1, height1);

    for y in 0..result.height() {
        for x in 0..result.width() {
            let image_value = unsafe { image1.unsafe_get_pixel(x, y).0 };
            let template_value = unsafe { image2.unsafe_get_pixel(x, y).0 };

            let score = image_value
                .iter()
                .zip(template_value.iter())
                .map(|(i, t)| i - t)
                .collect::<Vec<_>>();
            let score = [score[0], score[1], score[2]];

            result.put_pixel(x, y, Rgb(score));
        }
    }

    result
}

//pub(crate) fn to_pixels(img: &Image) -> Vec<f32> {
//    img.as_bytes().iter().map(|x| *x as f32/256.).collect::<Vec<_>>()
//}

fn is_white(img: &Image) -> bool {
    let mut wc = 0;
    let mut count = 0;
    for (_x, _y, px) in img.enumerate_pixels() {
        if px.0.iter().all(|x| *x == 255) {
            wc += 1;
        }
        count += 1
    }
    // println!("{wc} {count}");
    wc * 2 > count
}

pub(crate) fn has_lost(img: &Image, gameover: &ImageA) -> bool {
    let gameover_img = get_pixel_range(img, (147..271, 193..209));
    match_template(&gameover_img, gameover)
        .pixels()
        .filter(|x| x.0[0] == 0.0)
        .next()
        .is_some()
        && !is_white(&gameover_img)
}
pub(crate) fn num_lives(img: &Image, life: &ImageA) -> isize {
    let life_img = get_pixel_range(img, (497..610, 274..289));
    match_template(&life_img, life)
        .pixels()
        .filter(|x| x.0[0] == 0.0)
        .count() as isize
}

pub(crate) fn get_score(img: &Image, nums: &[ImageA]) -> Option<isize> {
    let mut score = [20; 8];

    let score_img = get_pixel_range(img, (449..577, 97..113));
    if is_white(&score_img) {
        return None;
    }
    //score_img.save("score check/score.png").unwrap();
    for (i, n) in nums.iter().enumerate() {
        let matching = match_template(&score_img, n)
            .pixels()
            .map(|p| p.0[0])
            .collect::<Vec<_>>();

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
            if matching[s * 16] == 0.0 {
                // For some reason 6 doesnt match pixel perfect, this is zero for all others.
                score[s] = i as u8;
            }
        }
    }

    //eprintln!("{score:?}");

    if score.iter().find(|x| **x == 20).is_some() {
        None
    } else {
        Some(
            score
                .iter()
                .rev()
                .enumerate()
                .rev()
                .map(|(i, v)| 10isize.pow(i as u32) * *v as isize)
                .sum(),
        )
    }
}

static ENIGO: Lazy<Mutex<enigo::Enigo>> = Lazy::new(|| {
    Mutex::new({
        let mut e = enigo::Enigo::new();
        e.set_delay(0);
        e
    })
});

pub(crate) fn do_keys(buttons: &[bool]) {
    let mut e = ENIGO.lock().unwrap();

    if buttons[0] {
        e.key_down(Key::Layout('z'));
    } else {
        e.key_up(Key::Layout('z'));
    }
    if buttons[1] {
        e.key_down(Key::Layout('x'));
    } else {
        e.key_up(Key::Layout('x'));
    }
    if buttons[2] {
        e.key_down(Key::UpArrow);
    } else {
        e.key_up(Key::UpArrow);
    }
    if buttons[3] {
        e.key_down(Key::DownArrow);
    } else {
        e.key_up(Key::DownArrow);
    }
    if buttons[4] {
        e.key_down(Key::LeftArrow);
    } else {
        e.key_up(Key::LeftArrow);
    }
    if buttons[5] {
        e.key_down(Key::RightArrow);
    } else {
        e.key_up(Key::RightArrow);
    }
}

pub(crate) fn get_keys() -> [bool; 6] {
    let d = device_query::DeviceState::new();
    let keys = d.query_keymap();
    [
        Keycode::Z,
        Keycode::X,
        Keycode::Up,
        Keycode::Down,
        Keycode::Left,
        Keycode::Right,
    ]
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
#[allow(dead_code)]
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

fn get_mode(pos: (i32, i32), nums: &[ImageA]) -> Mode {
    let ss = screenshot(pos);

    if ss.get_pixel(420, 69).0 == [119, 119, 153] {
        // Checks a background pixel for blueish color
        Mode::Select
    } else if ss.get_pixel(420, 69).0 == [255, 255, 0] {
        // Checks a text pixel for yellow color
        Mode::Start
    } else if (244..=254).all(|y| ss.get_pixel(202, y).0 == [255, 255, 255]) {
        // Checks the vertical line of the R in Return
        Mode::Paused(PauseState::Return)
    } else if (260..=269).all(|y| ss.get_pixel(236, y).0 == [255, 255, 255]) {
        // Checks the vertical line of the I in I was just kidding. Sorry.
        Mode::Paused(PauseState::Quit)
    } else if (244..=254).all(|y| ss.get_pixel(141, y).0 == [255, 255, 255]) {
        // Checks the vertical line of the I in Yes, I'll quit.
        Mode::Paused(PauseState::Kidding)
    } else if (260..=270).all(|y| ss.get_pixel(213, y).0 == [255, 255, 255]) {
        // Checks the vertial line of the t in Quit
        Mode::Paused(PauseState::Really)
    } else if get_score(&ss, nums).is_some() {
        Mode::Game
    } else {
        Mode::Between
    }
}

pub(crate) fn start(pos: (i32, i32), nums: &[ImageA]) {
    let mut e = enigo::Enigo::new();
    while get_mode(pos, nums) == Mode::Game {
        e.key_click(Key::Layout('z'));
    }
    while get_mode(pos, nums) != Mode::Game {
        e.key_click(Key::Layout('z'));
    }
}

#[allow(dead_code)]
pub(crate) fn reset(pos: (i32, i32), nums: &[ImageA]) {
    // let mut e = enigo::Enigo::new();
    //e.key_click(Key::Escape)

    start(pos, nums);
}

#[allow(dead_code)]
pub(crate) fn get_enc_batch<R: Rng>(n: usize, r: &mut R) -> Vec<Image> {
    let files = fs::read_dir("images/encoder_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            if y.file_type().unwrap().is_file() {
                Some(y.path())
            } else {
                None
            }
        })
        .collect();
    to_ret
        .choose_multiple(r, n)
        .into_iter()
        .map(|f| image::open(f).unwrap().into_rgb8())
        .collect()
}
pub(crate) fn get_actor_iter() -> ActImgSeq {
    let files = fs::read_dir("images/actor_critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".png") {
                Some(y.path().to_str().unwrap().to_string())
            } else {
                None
            }
        })
        .collect();
    ActImgSeq::new(to_ret)
}
pub(crate) struct ActImgSeq {
    pub(crate) files: Vec<String>,
    pub(crate) i: usize,
}
#[allow(dead_code)]
impl ActImgSeq {
    fn new(files: Vec<String>) -> Self {
        ActImgSeq { files, i: 0 }
    }

    pub(crate) fn shuffle<R: Rng>(&mut self, r: &mut R) {
        self.files[self.i..].shuffle(r);
    }
    pub(crate) fn reset(&mut self) {
        self.i = 0;
    }
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }
    pub(crate) fn peek(&mut self) -> Option<<&mut Self as Iterator>::Item> {
        let mut s = self;
        let v = s.next();
        s.i -= 1;
        v
    }
    pub(crate) fn split_test(self, test_fraction: usize) -> (Self, Self) {
        let (mut train, mut test) = (vec![], vec![]);
        for a in self.files.chunks(test_fraction) {
            test.push(a[0].clone());
            train.extend_from_slice(&a[1..]);
        }
        (Self { files: train, i: 0 }, Self { files: test, i: 0 })
    }
}
impl Iterator for &mut ActImgSeq {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let name = self.files.get(self.i)?;
        self.i += 1;
        let Some(img) = image::open(name).ok() else {
            eprintln!("Error on file `{name}`");
            return self.next();
        };
        let img = to_pixels(&img.into_rgb8());
        Some(img)
    }
}
#[allow(dead_code)]
pub(crate) fn get_crit_seq<R: Rng>(n: usize, r: &mut R) -> Vec<(Image, Vec<f32>, f32)> {
    let files = fs::read_dir("images/critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".seq") {
                Some(y.path())
            } else {
                None
            }
        })
        .collect();
    // to_ret
    //     .choose_multiple(r, n)
    //     .into_iter()
    //     .map(|f| image::open(f).unwrap().into_rgb8())
    //     .collect()
    let seq = to_ret.choose(r).unwrap();
    let slen = std::fs::read_to_string(seq)
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let sp = r.gen_range(0..(slen - n));
    let seq_name = seq
        .to_str()
        .unwrap()
        .split("/")
        .last()
        .unwrap()
        .split(".")
        .next()
        .unwrap();

    (sp..(sp + n))
        .map(|x| format!("images/critic_data/{seq_name}_{x}"))
        .map(|n| {
            let json = std::fs::read_to_string(format!("{n}.json")).unwrap();
            let (key, score): (Vec<f32>, f32) = serde_json::from_str(&json).unwrap();
            let img = image::open(format!("{n}.png")).unwrap().into_rgb8();
            (img, key, score)
        })
        .collect()
}
pub(crate) struct CritSeq {
    files: Vec<String>,
    i: usize,
}
impl CritSeq {
    fn new(files: Vec<String>) -> Self {
        CritSeq { files, i: 0 }
    }

    pub(crate) fn shuffle<R: Rng>(&mut self, r: &mut R) {
        self.files[self.i..].shuffle(r);
    }
    pub(crate) fn reset(&mut self) {
        self.i = 0;
    }
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }

    pub(crate) fn split_test(self, test_fraction: usize) -> (Self, Self) {
        let (mut train, mut test) = (vec![], vec![]);
        for a in self.files.chunks(test_fraction) {
            test.push(a[0].clone());
            train.extend_from_slice(&a[1..]);
        }
        (Self { files: train, i: 0 }, Self { files: test, i: 0 })
    }
}
impl Iterator for &mut CritSeq {
    type Item = (Vec<f32>, Vec<f32>, f32);

    fn next(&mut self) -> Option<Self::Item> {
        // if self.i > 100 {
        //     return None;
        // }
        let name = self.files.get(self.i)?;
        self.i += 1;
        let seq_name = name.split(".").next().unwrap();
        let json = std::fs::read_to_string(format!("{seq_name}.done.json")).unwrap();
        // println!("{json} {seq_name}");
        let Ok((key, score)): Result<(Vec<f32>, f32), _> = serde_json::from_str(&json) else {
            return None;
        };
        let key = to_flat(&key);
        let Some(img) = image::open(format!("{seq_name}.png")).ok() else {
            return self.next();
        };
        let img = to_pixels(&img.into_rgb8());
        Some((img, key, score))
    }
}
#[allow(dead_code)]
pub(crate) fn get_crit_batch<R: Rng>(n: usize, r: &mut R) -> Vec<(Image, Vec<f32>, f32)> {
    let files = fs::read_dir("images/actor_critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".done.json")
            {
                Some(y.path())
            } else {
                None
            }
        })
        .collect();
    // to_ret
    //     .choose_multiple(r, n)
    //     .into_iter()
    //     .map(|f| image::open(f).unwrap().into_rgb8())
    //     .collect()
    let seq = to_ret.choose_multiple(r, n);
    seq.filter_map(|n| {
        let seq_name = n.to_str().unwrap().split(".").next().unwrap();
        let json = std::fs::read_to_string(format!("{seq_name}.done.json")).unwrap();
        // println!("{json} {seq_name}");
        let Ok((key, score)): Result<(Vec<f32>, f32), _> = serde_json::from_str(&json) else {
            return None;
        };
        let img = image::open(format!("{seq_name}.png")).unwrap().into_rgb8();
        Some((img, key, score))
    })
    .collect()
}
pub(crate) fn get_crit_iter() -> CritSeq {
    let files = fs::read_dir("images/actor_critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".done.json")
            {
                Some(y.path().to_str().unwrap().to_string())
            } else {
                None
            }
        })
        .collect();
    CritSeq::new(to_ret)
}
pub(crate) struct QSeq {
    files: Vec<String>,
    i: usize,
}
impl QSeq {
    fn new(files: Vec<String>) -> Self {
        QSeq { files, i: 0 }
    }

    pub(crate) fn shuffle<R: Rng>(&mut self, r: &mut R) {
        self.files[self.i..].shuffle(r);
    }
    pub(crate) fn reset(&mut self) {
        self.i = 0;
    }
    pub(crate) fn len(&self) -> usize {
        self.files.len()
    }

    pub(crate) fn split_test(self, test_fraction: usize) -> (Self, Self) {
        let (mut train, mut test) = (vec![], vec![]);
        for a in self.files.chunks(test_fraction) {
            test.push(a[0].clone());
            train.extend_from_slice(&a[1..]);
        }
        (Self { files: train, i: 0 }, Self { files: test, i: 0 })
    }
}
impl Iterator for &mut QSeq {
    // Image, keys,  score, next image
    type Item = (Vec<f32>, Vec<f32>, f32, Option<Vec<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        // if self.i > 100 {
        //     return None;
        // }
        let name = self.files.get(self.i)?;
        self.i += 1;
        let seq_name = name.split(".").next().unwrap();
        let json = std::fs::read_to_string(format!("{seq_name}.json")).unwrap();
        // println!("{json} {seq_name}");
        let Ok((keys, score)): Result<(Vec<f32>, f32), _> = serde_json::from_str(&json) else {
            return None;
        };
        // let key = to_flat(&key);
        let Some(img) = image::open(format!("{seq_name}.png")).ok() else {
            return self.next();
        };
        let img = to_pixels(&img.into_rgb8());
        // println!("{seq_name}");

        // let i = seq_name.split("_");
        // for j in i {
        // println!("{j}, {} {:?}", j.len(), j.parse::<usize>());
        // }
        let mut i = seq_name.split("_").collect::<Vec<_>>();
        let l = i.last_mut().unwrap();
        let n = (l.parse::<usize>().unwrap() + 1).to_string();
        *l = &n;
        let mut next_name = i.into_iter().fold("".to_string(), |x, y| x + "_" + &y);
        next_name.remove(0);
        // println!("{seq_name} {next_name}");
        let next = image::open(format!("{next_name}.png"))
            .ok()
            .map(|x| to_pixels(&x.into_rgb8()));
        // println!("{:?}", next.as_ref().map(|x| x.len()));
        Some((img, to_flat(&keys), score, next))
    }
}
pub(crate) fn get_q_iter() -> QSeq {
    let files = fs::read_dir("images/actor_critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".json") {
                Some(y.path().to_str().unwrap().to_string())
            } else {
                None
            }
        })
        .collect();
    QSeq::new(to_ret)
}
#[allow(dead_code)]
pub(crate) fn process_crit_(gamma: f32) {
    let files = fs::read_dir("images/critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".seq") {
                Some(y.path())
            } else {
                None
            }
        })
        .collect();
    // to_ret
    //     .choose_multiple(r, n)
    //     .into_iter()
    //     .map(|f| image::open(f).unwrap().into_rgb8())
    //     .collect()
    for seq in to_ret {
        println!("{}", seq.display());
        let slen = std::fs::read_to_string(&seq)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let seq_name = seq
            .to_str()
            .unwrap()
            .split("/")
            .last()
            .unwrap()
            .split(".")
            .next()
            .unwrap();

        let outp = (0..slen)
            .map(|x| format!("images/critic_data/{seq_name}_{x}"))
            .map(|n| {
                let json = std::fs::read_to_string(format!("{n}.json")).unwrap();
                let (key, score): (Vec<f32>, f32) = serde_json::from_str(&json).unwrap();
                (format!("{n}"), key, score)
            })
            .collect::<Vec<_>>();
        let mut key_vec = vec![];
        let mut out_vec = vec![];
        let mut name_vec = vec![];
        {
            let mut acc = 0.0f32;
            let mut i = 0;
            for (file_name, key, score) in outp.into_iter().rev() {
                print!("{i}/{slen} \r");
                i += 1;
                out_vec.push((acc + 1.0).ln());
                key_vec.push(key.into_iter().map(|x| x as f32).collect::<Vec<_>>());
                name_vec.push(file_name);
                acc *= gamma;
                acc += score;
            }
        }
        println!();
        for (_, (data, name)) in key_vec
            .into_iter()
            .zip(out_vec.into_iter())
            .zip(name_vec.into_iter())
            .enumerate()
        {
            std::fs::write(
                format!("{name}.done.json"),
                serde_json::to_string(&data).unwrap(),
            )
            .unwrap();
        }
    }
    // process_crit_act(gamma);
}
pub(crate) fn process_crit(gamma: f32) {
    let files = fs::read_dir("images/actor_critic_data").unwrap();
    let to_ret: Vec<_> = files
        .filter_map(|x| {
            let y = x.unwrap();
            // println!("{}", y.path().to_str().unwrap());
            if y.file_type().unwrap().is_file() && y.path().to_str().unwrap().contains(".seq") {
                Some(y.path())
            } else {
                None
            }
        })
        .collect();
    // to_ret
    //     .choose_multiple(r, n)
    //     .into_iter()
    //     .map(|f| image::open(f).unwrap().into_rgb8())
    //     .collect()
    for seq in to_ret {
        println!("{}", seq.display());
        let slen = std::fs::read_to_string(&seq)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let seq_name = seq
            .to_str()
            .unwrap()
            .split("/")
            .last()
            .unwrap()
            .split(".")
            .next()
            .unwrap();

        let outp = (0..slen)
            .map(|x| format!("images/actor_critic_data/{seq_name}_{x}"))
            .map(|n| {
                let json = std::fs::read_to_string(format!("{n}.json")).unwrap();
                let (key, score): (Vec<f32>, f32) = serde_json::from_str(&json).unwrap();
                (format!("{n}"), key, score)
            })
            .collect::<Vec<_>>();
        let mut key_vec = vec![];
        let mut out_vec = vec![];
        let mut name_vec = vec![];
        {
            let mut acc = 0.0f32;
            let mut i = 0;
            for (file_name, key, score) in outp.into_iter().rev() {
                print!("{i}/{slen} \r");
                i += 1;
                out_vec.push((acc + 1.0).ln());
                key_vec.push(key.into_iter().map(|x| x as f32).collect::<Vec<_>>());
                name_vec.push(file_name);
                acc *= gamma;
                acc += score;
            }
        }
        println!();
        for (_, (data, name)) in key_vec
            .into_iter()
            .zip(out_vec.into_iter())
            .zip(name_vec.into_iter())
            .enumerate()
        {
            std::fs::write(
                format!("{name}.done.json"),
                serde_json::to_string(&data).unwrap(),
            )
            .unwrap();
        }
    }
}

pub(crate) fn analyze_crit() {
    // let mut r = thread_rng();
    let dataset = {
        let files = fs::read_dir("images/actor_critic_data").unwrap();
        let to_ret: Vec<_> = files
            .filter_map(|x| {
                let y = x.unwrap();
                // println!("{}", y.path().to_str().unwrap());
                if y.file_type().unwrap().is_file()
                    && y.path().to_str().unwrap().contains(".done.json")
                {
                    Some(y.path())
                } else {
                    None
                }
            })
            .collect();
        // to_ret
        //     .choose_multiple(r, n)
        //     .into_iter()
        //     .map(|f| image::open(f).unwrap().into_rgb8())
        //     .collect()
        let seq = to_ret.iter();
        seq.filter_map(|n| {
            let seq_name = n.to_str().unwrap().split(".").next().unwrap();
            let json = std::fs::read_to_string(format!("{seq_name}.done.json")).unwrap();
            // println!("{json} {seq_name}");
            let Ok((_, score)): Result<(Vec<f32>, f64), _> = serde_json::from_str(&json) else {
                return None;
            };
            Some(score)
        })
        .collect::<Vec<_>>()
    };
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for i in dataset.iter() {
        sum += i;
        sumsq += i * i;
    }
    let mean = sum / (dataset.len() as f64);
    let exsq = sumsq / (dataset.len() as f64);
    let var = exsq - mean * mean;
    let sd = var.sqrt();
    println!("Mean: {mean}");
    println!("SD: {sd}");
    let mut num_outlier = 0;
    let mut max = f64::MIN;
    let mut min = f64::MAX;
    for i in dataset.iter() {
        if (i - mean).abs() > sd * 4. {
            num_outlier += 1;
            // println!("{i}");
        }
        if i > &max {
            max = *i;
        }
        if i < &min {
            min = *i;
        }
    }
    println!(
        "Outlier (4sd): {}",
        (num_outlier as f64) / (dataset.len() as f64)
    );
    println!("Min {min} max {max}");
}
pub(crate) fn reset_from_gameover() {
    thread::sleep(Duration::from_millis(4000));
    for _ in 0..2 {
        // println!("Rlset");
        do_keys(&[true, false, false, false, false, false]);
        thread::sleep(Duration::from_millis(100));
        // println!("Rlset");
        do_keys(&[false, false, false, false, false, false]);
        thread::sleep(Duration::from_millis(100));
    }
    // println!("Reset");
    do_keys(&[false, false, false, true, false, false]);
    thread::sleep(Duration::from_millis(100));
    do_keys(&[false, false, false, false, false, false]);
    thread::sleep(Duration::from_millis(100));
    // println!("Reset");
    do_keys(&[true, false, false, false, false, false]);
    thread::sleep(Duration::from_millis(100));
    do_keys(&[false, false, false, false, false, false]);
    // println!("Reset");
    // thread::sleep(Duration::from_millis(3000));
}

pub(crate) fn do_dialog(img: &Image, dia: &ImageA) {
    let score_img = get_pixel_range(img, (42..58, 321..329));

    let skip = match_template(&score_img, dia)
        .pixels()
        .filter(|x| x.0[0] == 0.0)
        .next()
        .is_some();
    if skip {
        do_keys(&[false, false, false, false, false, false]);
        thread::sleep(Duration::from_millis(33));
        do_keys(&[true, false, false, false, false, false]);
        thread::sleep(Duration::from_millis(33));
        do_keys(&[false, false, false, false, false, false]);
        thread::sleep(Duration::from_millis(33));
    }
}

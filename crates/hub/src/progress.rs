//! A progress meter for operations measured in gigabytes.
//!
//! Every step of fetching a 21 GB checkpoint is slow enough to look like a
//! hang: the download obviously, but also the SHA-256 pass over the result and
//! the re-hash of a partial file before a resume. All three report through this.
//!
//! Redraws are throttled to 10 Hz — a write per 8 MB chunk would spend real time
//! on escape codes — and the meter degrades to periodic lines when stdout is not
//! a terminal, so piping to a file does not produce a megabyte of `\r`.

use std::io::Write;
use std::time::{Duration, Instant};

extern "C" {
    fn isatty(fd: i32) -> i32;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

#[repr(C)]
#[derive(Default)]
struct Winsize {
    row: u16,
    col: u16,
    xpixel: u16,
    ypixel: u16,
}

/// macOS/BSD value. A failed ioctl just falls back to 80 columns.
const TIOCGWINSZ: u64 = 0x4008_7468;

fn term_width() -> usize {
    let mut ws = Winsize::default();
    let ok = unsafe { ioctl(1, TIOCGWINSZ, &mut ws as *mut Winsize) } == 0;
    if ok && ws.col >= 40 {
        ws.col as usize
    } else {
        80
    }
}

pub struct Progress {
    label: String,
    total: u64,
    done: u64,
    start: Instant,
    last_draw: Instant,
    /// Bytes and instant at the previous speed sample, for a smoothed rate.
    mark: (u64, Instant),
    speed: f64,
    /// When bytes last actually arrived, so a stall is visible rather than
    /// looking like a slow download.
    last_progress: Instant,
    tty: bool,
    finished: bool,
}

impl Progress {
    pub fn new(label: impl Into<String>, total: u64) -> Self {
        let now = Instant::now();
        let mut p = Self {
            label: label.into(),
            total,
            done: 0,
            start: now,
            last_draw: now - Duration::from_secs(1),
            mark: (0, now),
            speed: 0.0,
            last_progress: now,
            tty: unsafe { isatty(1) } == 1,
            finished: false,
        };
        p.draw(true);
        p
    }

    /// Start partway in, for a resumed transfer. Redraws immediately so the
    /// first frame shows the resumed position rather than a misleading zero.
    pub fn at(mut self, done: u64) -> Self {
        self.done = done;
        self.mark = (done, Instant::now());
        self.draw(true);
        self
    }

    pub fn add(&mut self, n: u64) {
        self.done += n;
        if n > 0 {
            self.last_progress = Instant::now();
        }
        self.draw(false);
    }

    fn sample_speed(&mut self, now: Instant) {
        let dt = now.duration_since(self.mark.1).as_secs_f64();
        if dt >= 0.5 {
            let inst = (self.done - self.mark.0) as f64 / dt;
            // Exponential smoothing: a raw per-tick rate jitters too much to
            // read, and the ETA computed from it is useless.
            self.speed = if self.speed == 0.0 {
                inst
            } else {
                0.7 * self.speed + 0.3 * inst
            };
            self.mark = (self.done, now);
        }
    }

    fn draw(&mut self, force: bool) {
        let now = Instant::now();
        if !force && now.duration_since(self.last_draw) < Duration::from_millis(100) {
            return;
        }
        self.last_draw = now;
        self.sample_speed(now);

        if !self.tty {
            // Not a terminal: one line every few seconds, no cursor tricks.
            if force || now.duration_since(self.start).as_secs() % 5 == 0 {
                println!(
                    "{}  {}/{}  {}",
                    self.label,
                    bytes(self.done),
                    bytes(self.total),
                    rate(self.speed)
                );
                std::io::stdout().flush().ok();
            }
            return;
        }

        let frac = if self.total > 0 {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let stalled = now.duration_since(self.last_progress) > Duration::from_secs(5);
        let tail = format!(
            " {:>5.1}%  {}/{}  {}  {}",
            frac * 100.0,
            bytes(self.done),
            bytes(self.total),
            if stalled { "stalled".into() } else { rate(self.speed) },
            eta(self.total.saturating_sub(self.done), self.speed),
        );

        // Give the bar whatever is left after the label and the numbers.
        let width = term_width();
        let bar_w = width
            .saturating_sub(self.label.chars().count() + tail.chars().count() + 4)
            .clamp(8, 40);
        let filled = (frac * bar_w as f64).round() as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_w - filled);

        print!("\r\x1b[2K{} [{}]{}", self.label, bar, tail);
        std::io::stdout().flush().ok();
    }

    /// Draw one final line and move off it, so the summary is not overwritten.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let elapsed = self.start.elapsed().as_secs_f64();
        let avg = if elapsed > 0.0 {
            self.done as f64 / elapsed
        } else {
            0.0
        };
        if self.tty {
            print!("\r\x1b[2K");
        }
        println!(
            "{}  {} in {}  ({} avg)",
            self.label,
            bytes(self.done),
            clock(elapsed as u64),
            rate(avg)
        );
        std::io::stdout().flush().ok();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // An error path that abandons the meter should not leave the cursor
        // parked mid-bar.
        if !self.finished && self.tty {
            println!();
        }
    }
}

pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1000.0 && u < UNITS.len() - 1 {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

fn rate(bps: f64) -> String {
    if bps <= 0.0 {
        return "—".into();
    }
    format!("{}/s", bytes(bps as u64))
}

fn eta(remaining: u64, bps: f64) -> String {
    if bps <= 1.0 {
        return "eta —".into();
    }
    format!("eta {}", clock((remaining as f64 / bps) as u64))
}

fn clock(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

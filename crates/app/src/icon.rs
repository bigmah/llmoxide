//! The app icon: a padlock in oxide orange (`assets/icon.svg`).

const PNG: &[u8] = include_bytes!("../assets/icon-512.png");

/// Show the icon in the Dock. A bare binary has no bundle to take it from,
/// so it is set on the running application. Must run on the main thread
/// once AppKit is up, which is anywhere in the UI.
#[cfg(target_os = "macos")]
pub fn set_dock_icon() {
    use objc2::AllocAnyThread;
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::{MainThreadMarker, NSData};

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let data = NSData::with_bytes(PNG);
    if let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) {
        unsafe { NSApplication::sharedApplication(mtm).setApplicationIconImage(Some(&image)) };
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_dock_icon() {}

/// The window icon, for the title bar and taskbar. macOS ignores it.
#[cfg(not(target_os = "macos"))]
pub fn window_icon() -> Option<winit::window::Icon> {
    let rgba = image::load_from_memory_with_format(PNG, image::ImageFormat::Png)
        .ok()?
        .into_rgba8();
    let (w, h) = rgba.dimensions();
    winit::window::Icon::from_rgba(rgba.into_raw(), w, h).ok()
}

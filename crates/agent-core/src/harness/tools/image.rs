//! Image format detection and bounded decoding using image/png.

use super::read::{ReadImageProcessor, ReadImageProcessorOptions, ReadImageProcessorResult};
use base64::Engine;
use image::ImageFormat;
use std::io::Cursor;

pub(crate) fn guess_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    match image::guess_format(buffer).ok()? {
        ImageFormat::Jpeg => Some("image/jpeg"),
        ImageFormat::Png => Some("image/png"),
        ImageFormat::Gif => Some("image/gif"),
        ImageFormat::WebP => Some("image/webp"),
        ImageFormat::Bmp => Some("image/bmp"),
        _ => None,
    }
}

/// Identify supported image headers, excluding animated PNG and malformed headers.
pub fn detect_supported_image_mime_type(buffer: &[u8]) -> Option<&'static str> {
    let mime = guess_supported_image_mime_type(buffer)?;
    if mime == "image/png" {
        let reader = png::Decoder::new(Cursor::new(buffer)).read_info().ok()?;
        if reader.info().animation_control.is_some() {
            return None;
        }
    }
    let format = image::guess_format(buffer).ok()?;
    image::ImageReader::with_format(Cursor::new(buffer), format)
        .into_dimensions()
        .ok()?;
    Some(mime)
}

pub fn encode_base64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Built-in processor: validate pixels, optionally resize, and convert BMP to PNG.
/// A custom ReadImageProcessor still overrides this implementation.
pub struct DefaultImageProcessor;

#[async_trait::async_trait]
impl ReadImageProcessor for DefaultImageProcessor {
    async fn process(
        &self,
        bytes: &[u8],
        mime_type: &str,
        options: ReadImageProcessorOptions,
    ) -> ReadImageProcessorResult {
        let bytes = bytes.to_vec();
        let mime = mime_type.to_owned();
        match tokio::task::spawn_blocking(move || {
            process_image(bytes, mime, options.auto_resize_images)
        })
        .await
        {
            Ok(Ok(result)) => result,
            Ok(Err(message)) => ReadImageProcessorResult::Err { message },
            Err(error) => ReadImageProcessorResult::Err {
                message: error.to_string(),
            },
        }
    }
}

fn process_image(
    bytes: Vec<u8>,
    mime_type: String,
    resize: bool,
) -> Result<ReadImageProcessorResult, String> {
    let format = image::guess_format(&bytes).map_err(|error| error.to_string())?;
    let mut reader = image::ImageReader::with_format(Cursor::new(&bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let mut image = reader.decode().map_err(|error| error.to_string())?;
    let original_dimensions = (image.width(), image.height());
    let needs_resize = resize && (image.width() > 1600 || image.height() > 1600);
    let mut hints = Vec::new();
    if needs_resize {
        image = image.thumbnail(1600, 1600);
        hints.push(format!(
            "Resized {}x{} to {}x{}",
            original_dimensions.0,
            original_dimensions.1,
            image.width(),
            image.height()
        ));
    }
    let (bytes, mime_type) = if needs_resize || format == ImageFormat::Bmp {
        let mut output = Cursor::new(Vec::new());
        image
            .write_to(&mut output, ImageFormat::Png)
            .map_err(|error| error.to_string())?;
        (output.into_inner(), "image/png".to_owned())
    } else {
        (bytes, mime_type)
    };
    Ok(ReadImageProcessorResult::Ok {
        data: encode_base64(&bytes),
        mime_type,
        hints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image_bytes(format: ImageFormat) -> Vec<u8> {
        let image = image::DynamicImage::new_rgb8(2, 2);
        let mut buffer = Cursor::new(Vec::new());
        image.write_to(&mut buffer, format).unwrap();
        buffer.into_inner()
    }
    #[test]
    fn detects_real_formats_and_rejects_truncated_headers() {
        for (format, mime) in [
            (ImageFormat::Png, "image/png"),
            (ImageFormat::Jpeg, "image/jpeg"),
            (ImageFormat::Gif, "image/gif"),
            (ImageFormat::WebP, "image/webp"),
            (ImageFormat::Bmp, "image/bmp"),
        ] {
            let bytes = image_bytes(format);
            assert_eq!(detect_supported_image_mime_type(&bytes), Some(mime));
            assert_eq!(detect_supported_image_mime_type(&bytes[..3]), None);
        }
    }
    #[test]
    fn rejects_animated_png() {
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(1, 0).unwrap();
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&[0, 0, 0])
            .unwrap();
        assert_eq!(detect_supported_image_mime_type(&bytes), None);
    }
    #[test]
    fn converts_bmp_and_rejects_corrupt_pixels() {
        let result =
            process_image(image_bytes(ImageFormat::Bmp), "image/bmp".to_owned(), true).unwrap();
        assert!(
            matches!(result, ReadImageProcessorResult::Ok { mime_type, .. } if mime_type == "image/png")
        );
        assert!(process_image(b"GIF89a".to_vec(), "image/gif".to_owned(), false).is_err());
    }
    #[test]
    fn base64_matches_standard_alphabet() {
        assert_eq!(encode_base64(b"hello"), "aGVsbG8=");
    }
}

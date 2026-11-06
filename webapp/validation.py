"""Upload validation. Everything here works on bytes in memory: an uploaded image is never written to disk.

Checks, in order (each failure maps to a friendly error page):
  1. a file was sent                                    -> 400
  2. the file name is sanitized, and its extension is on the allow-list   -> 415
  3. the size is within the limit (also enforced by MAX_CONTENT_LENGTH)   -> 413
  4. the content is really an allowed image type and decodes completely   -> 415 / 400
"""
import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_PIXELS = 40_000_000  # guards against "decompression bombs" (tiny file, enormous canvas)


class UploadError(Exception):
    def __init__(self, status: int, title: str, message: str):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    filename: str     # sanitized, safe to store and display
    sha256: str
    width: int
    height: int
    format: str


def sanitize_filename(raw: str | None) -> str:
    """Strip directories, odd characters and leading dots; fall back to a neutral name."""
    return secure_filename(raw or "") or "upload"


def validate_upload(file, max_bytes: int) -> ValidatedImage:
    if file is None or not getattr(file, "filename", ""):
        raise UploadError(400, "No file selected", "Choose an image file first, then press Analyse.")

    filename = sanitize_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
        raise UploadError(415, "Unsupported file type",
                          f"Files ending in “{ext or 'nothing'}” are not accepted. Please upload one of: {allowed}.")

    data = file.stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadError(413, "File too large", f"The limit is {max_bytes / 1024 / 1024:.0f} MB. "
                                                 "Please upload a smaller image.")
    if not data:
        raise UploadError(400, "Empty file", "The file you sent contains no data.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (width, height) = im.format, im.size
            if fmt not in ALLOWED_FORMATS:
                raise UploadError(415, "Unsupported image format",
                                  f"The file content is {fmt}, which is not accepted, whatever its name says.")
            if width * height > MAX_PIXELS:
                raise UploadError(400, "Image dimensions too large",
                                  f"{width}×{height} pixels is more than this service will process.")
            im.verify()                      # cheap structural check of the headers and chunks
        with Image.open(io.BytesIO(data)) as im:
            im.load()                        # full decode: catches truncated or corrupted pixel data
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError(400, "This image cannot be read",
                          "The file looks like an image but cannot be decoded; it may be corrupted or incomplete. "
                          "Please try another file.") from None

    return ValidatedImage(data, filename, hashlib.sha256(data).hexdigest(), width, height, fmt)


def thumbnail_data_uri(data: bytes, size: int = 360) -> str:
    """Small preview embedded in the result page. Built in memory and sent straight back, never stored."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, im)
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

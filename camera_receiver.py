"""FastAPI receiver for Raspberry Pi camera image uploads."""

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURE_DIR = BASE_DIR / "captured_images"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "JPEG": (".jpg", {"image/jpeg", "image/jpg"}),
    "PNG": (".png", {"image/png"}),
}
CHUNK_SIZE = 64 * 1024


def create_app(
    storage_dir: Path | str = DEFAULT_CAPTURE_DIR,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> FastAPI:
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be greater than zero")

    capture_dir = Path(storage_dir)
    application = FastAPI(title="DermaAI Camera Receiver", version="1.0.0")

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_, __):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "An image file is required in the 'file' multipart field.",
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request, error: HTTPException):
        return JSONResponse(
            status_code=error.status_code,
            content={"success": False, "error": error.detail},
            headers=error.headers,
        )

    @application.get("/health")
    async def health() -> dict:
        return {"success": True, "status": "healthy"}

    @application.post("/upload-image")
    async def upload_image(file: Annotated[UploadFile, File(...)]) -> dict:
        temporary_path: Path | None = None

        try:
            capture_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=capture_dir,
                prefix=".upload-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                size_bytes = 0
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Image exceeds the {max_upload_bytes}-byte upload limit.",
                        )
                    temporary_file.write(chunk)

            if size_bytes == 0:
                raise HTTPException(status_code=400, detail="The uploaded image is empty.")

            _, extension = _validate_image(
                temporary_path,
                file.content_type,
            )
            image_id = f"IMG-{uuid.uuid4().hex}"
            captured_at = datetime.now(timezone.utc)
            timestamp = captured_at.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            stored_filename = (
                f"{image_id}_{captured_at:%Y%m%dT%H%M%S%fZ}{extension}"
            )
            destination = capture_dir / stored_filename
            os.replace(temporary_path, destination)
            temporary_path = None

            return {
                "success": True,
                "image_id": image_id,
                "timestamp": timestamp,
                "filename": stored_filename,
                "upload_status": "success",
            }
        except HTTPException:
            raise
        except OSError as error:
            LOGGER.exception("Image upload failed due to a filesystem error.")
            raise HTTPException(
                status_code=500,
                detail="The server could not store the uploaded image.",
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            await file.close()

    return application


def _validate_image(path: Path, content_type: str | None) -> tuple[str, str]:
    try:
        with Image.open(path) as image:
            image_format = image.format
            dimensions = image.size
            image.verify()

        if image_format not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image format. Upload a JPEG or PNG image.",
            )

        expected_content_types = ALLOWED_IMAGE_TYPES[image_format][1]
        normalized_content_type = (
            content_type.lower().split(";", 1)[0].strip()
            if content_type
            else None
        )
        if normalized_content_type and normalized_content_type not in expected_content_types | {
            "application/octet-stream"
        }:
            raise HTTPException(
                status_code=415,
                detail="The uploaded content type does not match the image format.",
            )

        with Image.open(path) as image:
            image.load()

        if dimensions[0] <= 0 or dimensions[1] <= 0:
            raise HTTPException(status_code=400, detail="The image dimensions are invalid.")

        return image_format, ALLOWED_IMAGE_TYPES[image_format][0]
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid, readable image.",
        ) from error


app = create_app(
    storage_dir=os.environ.get("DERMAAI_CAPTURE_DIR", str(DEFAULT_CAPTURE_DIR)),
    max_upload_bytes=int(
        os.environ.get("DERMAAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
    ),
)

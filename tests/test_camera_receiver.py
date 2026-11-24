import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from camera_receiver import create_app


def make_image(image_format: str = "JPEG") -> bytes:
    image = Image.new("RGB", (24, 24), color=(70, 120, 160))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class CameraReceiverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.capture_dir = Path(self.temporary_directory.name) / "captured_images"
        self.client = TestClient(
            create_app(storage_dir=self.capture_dir, max_upload_bytes=1024)
        )

    def tearDown(self) -> None:
        self.client.close()
        self.temporary_directory.cleanup()

    def test_saves_valid_jpeg_and_returns_id_and_timestamp(self) -> None:
        response = self.client.post(
            "/upload-image",
            files={"file": ("sample.jpg", make_image(), "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])
        self.assertEqual(result["upload_status"], "success")
        self.assertTrue(result["image_id"].startswith("IMG-"))
        self.assertTrue(result["timestamp"].endswith("Z"))
        saved_path = self.capture_dir / result["filename"]
        self.assertTrue(saved_path.is_file())
        self.assertEqual(saved_path.read_bytes(), make_image())

    def test_health_endpoint_reports_api_ready(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "status": "healthy"})

    def test_accepts_png_and_creates_unique_ids(self) -> None:
        first = self.client.post(
            "/upload-image",
            files={"file": ("first.jpg", make_image("PNG"), "image/png")},
        )
        second = self.client.post(
            "/upload-image",
            files={"file": ("second.jpg", make_image("PNG"), "image/png")},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.json()["image_id"], second.json()["image_id"])
        self.assertEqual(len(list(self.capture_dir.iterdir())), 2)

    def test_missing_image_returns_structured_error(self) -> None:
        response = self.client.post("/upload-image")

        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.json()["success"])

    def test_unsupported_format_is_rejected(self) -> None:
        gif_image = make_image("GIF")
        response = self.client.post(
            "/upload-image",
            files={"file": ("sample.gif", gif_image, "image/gif")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.json()["success"])
        self.assertEqual(list(self.capture_dir.iterdir()), [])

    def test_corrupted_image_is_rejected(self) -> None:
        response = self.client.post(
            "/upload-image",
            files={"file": ("broken.jpg", b"\xff\xd8\xffbroken", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])
        self.assertEqual(list(self.capture_dir.iterdir()), [])

    def test_oversized_image_is_rejected(self) -> None:
        large_app = create_app(
            storage_dir=self.capture_dir,
            max_upload_bytes=32,
        )
        with TestClient(large_app) as client:
            response = client.post(
                "/upload-image",
                files={"file": ("large.jpg", make_image(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 413)
        self.assertFalse(response.json()["success"])
        self.assertEqual(list(self.capture_dir.iterdir()), [])

    def test_content_type_must_match_image_format(self) -> None:
        response = self.client.post(
            "/upload-image",
            files={"file": ("sample.png", make_image(), "image/png")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.json()["success"])
        self.assertEqual(list(self.capture_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

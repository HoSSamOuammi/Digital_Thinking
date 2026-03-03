from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import create_app


class StudioAppTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.app = create_app()
        self.upload_dir = Path(self.tempdir.name) / "uploads"
        self.generated_dir = Path(self.tempdir.name) / "generated"
        self.preview_dir = self.generated_dir / "previews"
        self.admins_dir = Path(self.tempdir.name) / "Admins"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.admins_dir.mkdir(parents=True, exist_ok=True)

        self.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=self.upload_dir,
            GENERATED_FOLDER=self.generated_dir,
            PREVIEW_FOLDER=self.preview_dir,
            ADMINS_FOLDER=self.admins_dir,
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def _generated_files(self):
        return [path for path in self.generated_dir.iterdir() if path.is_file()]

    def _csrf_token(self) -> str:
        with self.client.session_transaction() as current_session:
            token = current_session.get("_csrf_token")
            if not token:
                token = "test-csrf-token"
                current_session["_csrf_token"] = token
        return token

    def test_core_pages_render(self):
        for path in ("/", "/gallery", "/team", "/generative", "/data-art", "/media-tools"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

    def test_team_page_uses_admin_photos_when_present(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (80, 100), color="#7a8f6a").save(image_bytes, format="PNG")
        (self.admins_dir / "hossam.png").write_bytes(image_bytes.getvalue())

        response = self.client.get("/team")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Hossam", body)
        self.assertIn("Khadija", body)
        self.assertIn("Admins/hossam.png", body)

    def test_generative_post_creates_artwork_and_keeps_overlay_state(self):
        response = self.client.post(
            "/generative",
            data={
                "csrf_token": self._csrf_token(),
                "series": "constellation",
                "palette": "sunset",
                "custom_palette": "",
                "number_of_shapes": "120",
                "size_variation": "1.1",
                "density": "0.9",
                "line_density": "1.0",
                "canvas_width": "960",
                "canvas_height": "640",
                "background": "aurora",
                "seed": "123456",
                "animation": "on",
                "overlay_shapes": '[{"shape":"circle","x":0.3,"y":0.4,"size":24,"color":"#ffffff"}]',
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self._generated_files())
        body = response.get_data(as_text=True)
        self.assertIn('"shape":"circle"', body.replace(" ", ""))

    def test_generative_preview_api_returns_preview_url(self):
        response = self.client.post(
            "/api/generative-preview",
            json={
                "series": "mosaic",
                "palette": "sunset",
                "custom_palette": "",
                "number_of_shapes": 120,
                "size_variation": 1.0,
                "density": 0.8,
                "line_density": 1.0,
                "canvas_width": 960,
                "canvas_height": 640,
                "background": "aurora",
                "seed": 123456,
                "animation": False,
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("preview_url", payload)
        self.assertTrue(any(self.preview_dir.iterdir()))

    def test_data_art_post_uses_single_preprocessed_frame(self):
        response = self.client.post(
            "/data-art",
            data={
                "csrf_token": self._csrf_token(),
                "data_style": "radial",
                "focus_column": "auto",
                "colormap": "magma",
                "smoothing_window": "8",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self._generated_files())

    def test_image_processing_post_creates_output(self):
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color="#335c67").save(image_bytes, format="PNG")
        image_bytes.seek(0)

        response = self.client.post(
            "/media-tools",
            data={
                "csrf_token": self._csrf_token(),
                "panel": "image",
                "image_effect": "glitch",
                "rotate_degrees": "45",
                "pixel_size": "8",
                "kmeans_colors": "5",
                "glitch_shift": "12",
                "image_file": (image_bytes, "sample.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self._generated_files())
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_gallery_paginates_outputs(self):
        for index in range(15):
            (self.generated_dir / f"image_{index}.png").write_bytes(b"png")
        for index in range(9):
            (self.generated_dir / f"audio_{index}.wav").write_bytes(b"wav")

        response = self.client.get("/gallery?image_page=2&audio_page=2")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Images 2 / 2", body)
        self.assertIn("Audio 2 / 2", body)

    def test_media_tools_reports_audio_unavailable_reason(self):
        with patch("app.get_audio_status", return_value={"available": False, "reason": "ffmpeg missing"}):
            response = self.client.get("/media-tools")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ffmpeg missing", response.get_data(as_text=True))

    def test_form_posts_require_csrf_token(self):
        response = self.client.post("/generative", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("The request could not be verified", response.get_data(as_text=True))

    def test_preview_api_requires_csrf_token(self):
        response = self.client.post("/api/generative-preview", json={"series": "mosaic"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_generative_failures_do_not_expose_internal_exception_details(self):
        with patch("app.create_generative_art", side_effect=RuntimeError("sensitive-debug-detail")):
            response = self.client.post(
                "/generative",
                data={"csrf_token": self._csrf_token()},
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Could not generate artwork right now.", body)
        self.assertNotIn("sensitive-debug-detail", body)

    def test_uploaded_dataset_is_removed_after_processing(self):
        dataset = io.BytesIO(b"value_a,value_b\n1,2\n3,4\n")
        response = self.client.post(
            "/data-art",
            data={
                "csrf_token": self._csrf_token(),
                "data_style": "all",
                "focus_column": "auto",
                "colormap": "magma",
                "smoothing_window": "4",
                "dataset_file": (dataset, "sample.csv"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_generated_output_cleanup_keeps_only_recent_files(self):
        self.app.config["MAX_SAVED_GENERATED_FILES"] = 2
        for index in range(3):
            path = self.generated_dir / f"stale_{index}.png"
            path.write_bytes(b"png")
            os.utime(path, (100 + index, 100 + index))

        response = self.client.post(
            "/generative",
            data={"csrf_token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        remaining = sorted(path.name for path in self._generated_files())
        self.assertEqual(len(remaining), 2)
        self.assertTrue(any(name.startswith("generative_") for name in remaining))


if __name__ == "__main__":
    unittest.main()

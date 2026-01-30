import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def fake_image():
    img = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes

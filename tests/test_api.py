def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict(client, fake_image):
    response = client.post("/predict", files={"file": ("test.jpg", fake_image, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "season" in data
    assert "probability" in data

def test_dataset(client):
    response = client.get("/test_dataset")
    assert response.status_code == 200
    data = response.json()
    assert "cm_image" in data
    assert "accuracy" in data

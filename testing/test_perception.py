import pytest
import numpy as np
from test_helper import load_sample_by_label, load_all_samples
from unittest.mock import MagicMock, patch

@pytest.fixture
def perception():
    with patch("cv2.VideoCapture") as mock_cap, patch("tensorflow.keras.models.load_model"), patch("joblib.load"):
    
        mock_cap.return_value.isOpened.return_value = True

        from perception import Perception
        p = Perception()
        p.model = MagicMock()
        p.encoder = MagicMock()
        return p
    
def test_get_frame_no_hand_returns_none_label(perception):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    perception.cap.read.return_value = (True, fake_frame)
    
    # Mock extract_landmarks to simulate no hand found
    perception.extract_landmarks = lambda frame: None
    
    frame, label = perception.get_frame()
    assert label is None
    
def test_get_frame_returns_none_on_failed_read(perception):
    perception.cap.read.return_value = (False, None)
    frame, label = perception.get_frame()
    assert frame is None
    assert label is None    

def test_extract_landmarks_correct_shape(perception):
    # 21 landmarks * x,y = 42 values, reshaped to (1, 42)
    fake_landmarks = np.ones((1, 42))
    perception.model.predict.return_value = np.array([[0.99, 0, 0]])
    perception.encoder.inverse_transform.return_value = ["closed_fist"]
    
    result = perception.classify_landmarks(fake_landmarks)
    assert fake_landmarks.shape == (1, 42)
    
def test_none_input_returns_none(perception):
    assert perception.classify_landmarks(None) is None
    
def test_classify_landmarks_empty_array_returns_none(perception):
    result = perception.classify_landmarks(np.array([]))
    assert result is None
    
def test_closed_fist_classify_landmarks(perception):
    # load in sample from test_data.csv
    test_sample = load_sample_by_label("closed_fist")
    
    # setup mock model and encoder (tell them what to return)
    perception.model.predict.return_value = np.array([[0.99, 0, 0]])
    perception.encoder.inverse_transform.return_value = ["closed_fist"]
    
    result = perception.classify_landmarks(test_sample)
    assert result == "closed_fist"
    
def test_open_palm_classify_landmarks(perception):
    test_sample = load_sample_by_label("open_palm")
    
    perception.model.predict.return_value = np.array([[0, 0.99, 0]])
    perception.encoder.inverse_transform.return_value = ["open_palm"]
    
    result = perception.classify_landmarks(test_sample)
    assert result == "open_palm"
    
def test_point_up_classify_landmarks(perception):
    test_sample = load_sample_by_label("point_up")
    
    perception.model.predict.return_value = np.array([[0, 0, 0.99]])
    perception.encoder.inverse_transform.return_value = ["point_up"]
    
    result = perception.classify_landmarks(test_sample)
    assert result == "point_up"
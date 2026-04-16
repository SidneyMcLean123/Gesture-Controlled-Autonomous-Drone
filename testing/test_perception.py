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
    
def test_none_input_returns_none(perception):
    assert perception.classify_landmarks(None) is None
    
def test_closed_fist_classify_landmarks(perception):
    # load in sample from test_data.csv
    test_sample = load_sample_by_label("closed_fist")
    
    # setup mock model and encoder (tell them what to return)
    perception.model.predict.return_value = np.array([[0.99, 0, 0, 0, 0, 0, 0]])
    perception.encoder.inverse_transform.return_value = ["closed_fist"]
    
    result = perception.classify_landmarks(test_sample)
    assert result == "closed_fist"
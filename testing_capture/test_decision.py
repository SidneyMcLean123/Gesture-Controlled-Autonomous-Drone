import pytest
import math
from decision import Decision, GESTURE_BUFFER_LEN, MIN_FRAMES, CONFIDENCE

@pytest.fixture

# Pytest will automatically fill each 'decision' arguement with a brand new Decision(), buffer is fresh every time
def decision():
    return Decision()

# Basic mapping tests
def test_open_palm_maps_to_hover(decision):
    for _ in range(MIN_FRAMES):
        result = decision.update("open_palm")
    assert result == "HOVER"

def test_closed_fist_maps_to_land(decision):
    for _ in range(MIN_FRAMES):
        result = decision.update("closed_fist")
    assert result == "LAND"

def test_point_up_maps_to_takeoff(decision):
    for _ in range(MIN_FRAMES):
        result = decision.update("point_up")
    assert result == "TAKEOFF"

# Buffer tests
def test_returns_none_before_min_frames(decision):
    for _ in range(MIN_FRAMES - 1):
        result = decision.update("open_palm")
    assert result is None
    
def test_returns_intent_after_min_frames(decision):
    for _ in range(MIN_FRAMES):
        result = decision.update("open_palm")
    assert result == "HOVER"

# Alternate gestures so nothing reaches confidence threshold   
def test_returns_none_below_confidence_threshold(decision):
    for _ in range(MIN_FRAMES):
        decision.update("open_palm")
        result = decision.update("closed_fist")
    assert result is None

@pytest.mark.parametrize("n", range(int(MIN_FRAMES * CONFIDENCE)))
# Early buffer domination test
def test_fires_after_majority_switch(decision, n):
    for _ in range(n):
        decision.update("closed_fist")
    
    # Add exactly enough to reach the 
    result = None
    
    for _ in range((MIN_FRAMES - n)):
        result = decision.update("open_palm")
    
    assert result == "HOVER"

# @pytest.mark.parametrize("n", range(GESTURE_BUFFER_LEN))
# # Early buffer domination test
# def test_fires_after_majority_switch(decision, n):
#     for _ in range(n + MIN_FRAMES):
#         decision.update("closed_fist")
    
#     # Add exactly enough open_palm to dominate the buffer get >= confidence
#     result = None
    
#     for _ in range(math.ceil(1.5 * (n + MIN_FRAMES))):
#         result = decision.update("open_palm")
    
#     assert result == "HOVER"
    
    

# Test most_common != self.last_stable
def test_does_not_repeat_same_intent(decision):
    # Get a stable gesture decision
    for _ in range(MIN_FRAMES):
        decision.update("closed_fist")
    
    # Keep sending the same gesture -> should not fire again
    for _ in range(MIN_FRAMES):
        result = decision.update("closed_fist")
    assert result is None
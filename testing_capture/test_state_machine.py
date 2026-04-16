import pytest
from state_machine import StateMachine, State

@pytest.fixture
def sm():
    return StateMachine()

# Helper function to navigate to states
def got_to_state(sm, *intents):
    for intent in intents:
        sm.transition(intent)
    return sm

# Initial state
def test_initial_state_is_idle(sm):
    assert sm.get_state() == State.IDLE
    
# Transitions
def test_idle_to_takeoff(sm):
    result = sm.transition("TAKEOFF")
    assert result == State.TAKEOFF

def test_takeoff_to_hover(sm):
    sm.transition("TAKEOFF")
    result = sm.transition("HOVER")
    assert result == State.HOVER
    
def test_hover_to_land(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    result = sm.transition("LAND")
    assert result == State.LAND
    
# Emergency stop from every valid state 
@pytest.mark.parametrize("setup_intents", [
    ["TAKEOFF"],
    ["TAKEOFF", "HOVER"],
    ["TAKEOFF", "HOVER", "LAND"],
])

def test_emergency_stop_from_any_state(sm, setup_intents):
    for intent in setup_intents:
        sm.transition(intent)
    result = sm.transition("EMERGENCY_STOP")
    assert result == State.EMERGENCY_STOP
    
# Invalid intents
def test_invalid_intent_returns_None_in_IDLE(sm):
    result = sm.transition("GO_CRAZY")
    assert result is None
    
def test_invalid_intent_does_not_change_state_in_IDLE(sm):
    result = sm.transition("GO_CRAZY")
    assert sm.get_state() == State.IDLE
    
def test_invalid_intent_returns_None_in_TAKEOFF(sm):
    sm.transition("TAKEOFF")
    result = sm.transition("GO_CRAZY")
    assert result is None
    
def test_invalid_intent_does_not_change_state_in_TAKEOFF(sm):
    sm.transition("TAKEOFF")
    result = sm.transition("GO_CRAZY")
    assert sm.get_state() == State.TAKEOFF
    
def test_invalid_intent_returns_None_in_HOVER(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    result = sm.transition("GO_CRAZY")
    assert result is None
    
def test_invalid_intent_does_not_change_state_in_HOVER(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    result = sm.transition("GO_CRAZY")
    assert sm.get_state() == State.HOVER

def test_invalid_intent_returns_None_in_LAND(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    sm.transition("LAND")
    result = sm.transition("GO_CRAZY")
    assert result is None
    
def test_invalid_intent_does_not_change_state_in_LAND(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    sm.transition("LAND")
    result = sm.transition("GO_CRAZY")
    assert sm.get_state() == State.LAND
    
def test_invalid_intent_returns_None_in_EMERGENCY_STOP(sm):
    got_to_state(sm, "TAKEOFF")
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("GO_CRAZY")
    assert result is None

def test_invalid_intent_does_not_change_state_in_EMERGENCY_STOP(sm):
    got_to_state(sm, "TAKEOFF")
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("GO_CRAZY")
    assert sm.get_state() == State.EMERGENCY_STOP
    
# Invalid transitions
def test_cannot_hover_from_idle(sm):
    result = sm.transition("HOVER")
    assert result is None
    assert sm.get_state() == State.IDLE  # state unchanged

def test_cannot_land_from_idle(sm):
    result = sm.transition("LAND")
    assert result is None
    assert sm.get_state() == State.IDLE

def test_cannot_land_from_takeoff(sm):
    sm.transition("TAKEOFF")
    result = sm.transition("LAND")
    assert result is None
    assert sm.get_state() == State.TAKEOFF

def test_cannot_idle_from_takeoff(sm):
    sm.transition("TAKEOFF")
    result = sm.transition("IDLE")
    assert result is None
    assert sm.get_state() == State.TAKEOFF
    
def test_cannot_idle_from_hover(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    result = sm.transition("IDLE")
    assert result is None
    assert sm.get_state() == State.HOVER
    
def test_cannot_takeoff_from_hover(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    result = sm.transition("TAKEOFF")
    assert result is None
    assert sm.get_state() == State.HOVER
    
def test_cannot_takeoff_from_land(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    sm.transition("LAND")
    result = sm.transition("TAKEOFF")
    assert result is None
    assert sm.get_state() == State.LAND
    
def test_cannot_hover_from_land(sm):
    sm.transition("TAKEOFF")
    sm.transition("HOVER")
    sm.transition("LAND")
    result = sm.transition("HOVER")
    assert result is None
    assert sm.get_state() == State.LAND

def test_cannot_transition_to_idle_from_emergency_stop(sm):
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("IDLE")
    assert result is None
    assert sm.get_state() == State.EMERGENCY_STOP

def test_cannot_transition_to_takeoff_from_emergency_stop(sm):
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("TAKEOFF")
    assert result is None
    assert sm.get_state() == State.EMERGENCY_STOP

def test_cannot_transition_to_hover_from_emergency_stop(sm):
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("HOVER")
    assert result is None
    assert sm.get_state() == State.EMERGENCY_STOP

def test_cannot_transition_to_land_from_emergency_stop(sm):
    sm.transition("EMERGENCY_STOP")
    result = sm.transition("LAND")
    assert result is None
    assert sm.get_state() == State.EMERGENCY_STOP

# def test_land_to_idle(sm):
#     sm.transition("TAKEOFF")
#     sm.transition("HOVER")
#     sm.transition("LAND")
#     result = sm.transition("IDLE")
#     assert result == State.IDLE




from enum import Enum
from typing import Optional

class State(Enum):
    IDLE            = 0
    TAKEOFF         = 1
    HOVER           = 2
    LAND            = 3
    EMERGENCY_STOP  = 4

# Only these transitions are legal
TRANSITIONS = {
    State.IDLE:           [State.TAKEOFF],
    State.TAKEOFF:        [State.HOVER, State.EMERGENCY_STOP],
    State.HOVER:          [State.LAND, State.EMERGENCY_STOP],
    State.LAND:           [State.IDLE, State.EMERGENCY_STOP],
    State.EMERGENCY_STOP: []  
}

INTENT_TO_STATE = {
    "TAKEOFF":        State.TAKEOFF,
    "HOVER":          State.HOVER,
    "LAND":           State.LAND,
    "EMERGENCY_STOP": State.EMERGENCY_STOP,
}

class StateMachine:
    def __init__(self):
        self.state = State.IDLE

    def transition(self, intent: str) -> Optional[State]:
        target = INTENT_TO_STATE.get(intent)

        if target is None:
            print(f"Unknown intent: {intent}")
            return None

        if target not in TRANSITIONS[self.state]:
            print(f"Invalid transition: {self.state.name} -> {target.name}")
            return None

        print(f"STATE: {self.state.name} -> {target.name}")
        self.state = target
        return self.state

    def get_state(self) -> State:
        return self.state
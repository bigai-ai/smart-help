

# maintain the state of the agent
class Agent():
    def __init__(self):
        self.pick_up_obj_id = None
        self.action_history = []

    def reset(self):
        self.pick_up_obj_id = None
        self.action_history = []
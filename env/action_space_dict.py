from gymnasium import spaces
import random
import action as actions
import numpy as np
from gymnasium.spaces import Box


# add a mask
# set obj_num to max
class MyActionSpace(spaces.Dict):
    def __init__(self, low, high, dtype):
        super().__init__(
            {
                "action": Box(low[0], high[0], dtype=dtype),
                "tar_index": Box(low[1], high[1], dtype=dtype),
                # "agent_id": Box(low[2], high[2], dtype=dtype),
            }
        )

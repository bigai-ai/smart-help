import math
import random
from ai2thor.controller import Controller
from env.symbolic_env import SymbolicEnv
from constants import AgentType

config = {
    'agents_num': 1,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 1,
        "scene": 'FloorPlan28',
        "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local_1/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 1, 
        "quality": "Very Low", 
        "platform": "CloudRendering", 
    },
    # 'task': ['move_to', 'Apple|-00.93|+01.15|+00.95']
    # todo id change?
    # 'task': ['PickUp', 'Tomato|+00.17|+00.97|-00.28'],
    # 'task': ['PutOn', 'Tomato|+00.17|+00.97|-00.28', 'Sink|+00.00|+00.89|-01.44'],
    'task': ["MakeBreakfast"],
    # 'task': ['CookTomato']
    # 'task': ['PutOn', 'Tomato|+00.17|+00.97|-00.28', 'Sink|+00.00|+00.89|-01.44'],
}


# TODO: Combine two classes
class ActionNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
    def is_fully_expanded(self):
        return len(self.children) == goal_num

class ObjectNode:
    def __init__(self, parent=None, obj=None):
        self.parent = parent
        self.obj = obj
        self.children = []
        self.visits = 0
        self.value = 0
    def is_fully_expanded(self):
        return len(self.children) == goal_num


def ucb_score(parent_visits, child_value, child_visits):
    exploration_weight = 1.41  # sqrt(2) for UCB1
    if child_visits == 0:
        return float('inf')
    return child_value / child_visits + exploration_weight * math.sqrt(math.log(parent_visits) / child_visits)

def select_child(node):
    ucb_scores = [ucb_score(node.visits, child.value, child.visits) for child in node.children]
    return node.children[ucb_scores.index(max(ucb_scores))]

def expand(node, tree_type):
    if tree_type == 'action':
        unexplored_actions = [goal for goal in range(goal_num) if goal not in [child.action for child in node.children]]
        action = random.choice(unexplored_actions)
        child_node = ActionNode(parent=node, action=action)
        node.children.append(child_node)
        return child_node
    elif tree_type == 'obj':
        unexplored_objs = [obj for obj in range(obj_num) if obj not in [child.obj for child in node.children]]
        obj = random.choice(unexplored_objs)
        child_node = ObjectNode(parent=node, obj=obj)
        node.children.append(child_node)
        return child_node

def estimate_value(env):
    # TODO: Reward is sparse
    value = 0
    subtasks = env.subtask_list
    for subtask in subtasks:
        if env.check_sub_task(subtask):
            value += 1
    if env.check_task():
        value += 10

def simulate(node_action, node_obj, env):
    # TODO: Depth
    max_simulation_depth = 10 

    current_env = env
    current_env.step({"goal": node_action.action, "tar_index": node_obj.obj})

    for d in range(max_simulation_depth):
        # TODO: Better selection policy
        action = random.randint(0, goal_num)
        obj = random.randint(0, obj_num)

        current_env.step({"goal": action, "tar_index": obj})
        if current_env.check_task():     
            break 

    # Evaluate
    return estimate_value(current_env) 

def backprop(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent

def mcts_search(root_action, root_obj, env, num_simulations=1000):
    # TODO: num_simulations
    for _ in range(num_simulations):
        tmp_env = env
        # Action: Select and expand
        node_action = root_action
        while not node_action.is_fully_expanded() and node_action.children:
            node_action = select_child(node_action)
        if not node_action.is_fully_expanded():
            node_action = expand(node_action, 'action')

        # Object
        node_obj = root_obj
        while not node_obj.is_fully_expanded() and node_obj.children:
            node_obj = select_child(node_obj)
        if not node_obj.is_fully_expanded():
            node_obj = expand(node_obj, 'obj')

        value = simulate(node_action, node_obj, tmp_env)
        backprop(node_action, value)
        backprop(node_obj, value)

    best_action = max(root_action.children, key=lambda child: child.visits).action
    best_obj = max(root_obj.children, key=lambda child: child.visits).obj

    return best_action, best_obj


env = SymbolicEnv(config=config)
goal_num = len(env.goal_list)
obj_num = len(env.object_index2name)
action_seq = []

while not env.check_task() and len(action_seq) < 15:
    # TODO: Update existing tree, instead of regrowing
    root_action = ActionNode()
    root_obj = ObjectNode()

    # MCTS
    best_action, best_obj = mcts_search(root_action, root_obj, env)
    env.step({"goal": best_action, "tar_index": best_obj})
    action_seq.append({"goal": best_action, "tar_index": best_obj})

print(action_seq)


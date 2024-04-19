import math
import random
import time
from ai2thor.controller import Controller
from env.symbolic_env import SymbolicEnv
from env.goal import goal_list as action_list
from env.parse import parse_sub_task
from constants import AgentType

#random.seed(2023)
sample_prob = 0.9

config = {
    'agents_num': 1,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [1,1,1,1,1,1]},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 1,
        "scene": 'FloorPlan28',
        "local_executable_path": "/home/lfan/ai2thor/ai2thor/unity/builds/thor-Linux64-local/thor-Linux64-local",
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


class Node:
    def __init__(self, parent=None, action=None, obj=None, obj2=None):
        self.parent = parent
        self.action = action
        self.obj = obj
        self.obj2 = obj2
        self.children = []
        self.visits = 0
        self.value = 0
    def is_fully_expanded(self):
        return len(self.children) == goal_num * obj_num


def ucb_score(parent_visits, child_value, child_visits):
    exploration_weight = 1.41  # sqrt(2) for UCB1
    if child_visits == 0:
        return float('inf')
    return child_value / child_visits + exploration_weight * math.sqrt(math.log(parent_visits) / child_visits)

def select_child(node, eps = sample_prob):
    ucb_scores = [ucb_score(node.visits, child.value, child.visits) for child in node.children]

    # Use rule at a certain prob
    if random.random() > eps:
        goal = parse_sub_task(env.subtask_list[0])[0]
        #print(f'Selecting goal {goal}')
        if goal[0] == 'Put':
            pair = [action_list.index(goal[0]), env.object_name2index[goal[2]]]
        else:
            pair = [action_list.index(goal[0]), env.object_name2index[goal[1]]]
        for child in node.children:
            if child.action == pair[0] and child.obj == pair[1]:
                return child

    # Or select child accoding to ucb score    
    return node.children[ucb_scores.index(max(ucb_scores))]

def expand(node, eps = sample_prob):
    unexplored_pairs = []
    explored_pairs = [[child.action, child.obj] for child in node.children]

    for action in range(goal_num):
        # TODO: Special processing for 'Put' because it needs 2 obj
        for obj in range(obj_num):
            if [action, obj, None] not in explored_pairs:
                unexplored_pairs.append([action, obj, None])
    
    # Use rule at a certain prob
    if random.random() > eps:
        goal = parse_sub_task(env.subtask_list[0])[0]
        #print(f'Selecting goal {goal}')
        if goal[0] == 'Put':
            pair = [action_list.index(goal[0]), env.object_name2index[goal[2]]]
        else:
            pair = [action_list.index(goal[0]), env.object_name2index[goal[1]]]
        if pair in explored_pairs:
            pair = random.choice(unexplored_pairs) 
    # Or expand a random node
    else:
        pair = random.choice(unexplored_pairs) 
        
    child_node = Node(parent=node, action=pair[0], obj=pair[1])
    node.children.append(child_node)
    return child_node


def estimate_value(env):
    # TODO: Reward is sparse
    value = 0
    subtasks = env.subtask_list
    
    # Subtasks must be completed sequentially
    if env.check_sub_task(subtasks[0], agent_id=0):
        print(f'Simulation: Subtask {subtasks[0]} done')
        value += 100
    if env.check_task():
        value += 1000

    return value

def simulate(node, env):
    # Rule out invalid actions
    if not env.is_action_valid({"action": node.action, "tar_index": node.obj}):
        return 0
    
    # Init
    max_simulation_depth = 5 
    value = 0 
    if len(env.subtask_list) == 0:
        return 0
    curr_subtask = env.subtask_list[0]

    env.step({"goal": node.action, "tar_index": node.obj})
    # Rule out invalid actions
    if not env.controller.last_event.metadata['lastActionSuccess']:
        return 0
    
    #print(f'Simulating: action {node.action}-{action_list[node.action]}, \
       #obj {node.obj}-{env.object_index2name[node.obj]}')
    
    if env.check_sub_task(curr_subtask, agent_id=0):
        print(f'Simulation: Subtask {curr_subtask} done')
        return 100
    
    # Rollout
    for d in range(max_simulation_depth):
        action = random.randint(0, goal_num - 1)
        obj = random.randint(0, obj_num - 1)

        # TODO: Rule out invalid actions
        while not env.is_action_valid({"action": action, "tar_index": obj}):
            action = random.randint(0, goal_num - 1)
            obj = random.randint(0, obj_num - 1)

        env.step({"goal": action, "tar_index": obj})
        # Evaluate
        if env.check_sub_task(curr_subtask, agent_id=0):
            print(f'Simulation: Subtask {curr_subtask} done')
            value += (100 / ((d+2)*2)) # Discount by simulation steps
            break

    return value

def backprop(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent

def mcts_search(root, num_simulations=1000):
    # TODO: num_simulations
    for _ in range(num_simulations):
        env.reset()
        for action in action_seq:
            env.step(action)

        # Select and expand
        node = root
        parent_seq = []
        while node.is_fully_expanded() and node.children:
            parent_seq.append(node)
            node = select_child(node)
        if not node.is_fully_expanded():
            parent_seq.append(node)
            node = expand(node)
   
        for parent in parent_seq[1:]:
            env.step({"goal": parent.action, "tar_index": parent.obj})

        value = simulate(node, env)
        backprop(node, value)

    best_node = max(root.children, key=lambda child: child.visit)

    return best_node

# Debug: Print tree
from collections import deque
def print_tree(root):
    if root is None:
        return
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if node.action is not None and node.value > 0:
            print(f'Node: Action: {node.action}-{action_list[node.action]}, \
            Obj {node.obj}-{env.object_index2name[node.obj]}, Value {node.value}') 
        for child in node.children:
            queue.append(child)



env = SymbolicEnv(config=config)
goal_num = len(action_list)
obj_num = len(env.object_index2name)
print(f'Initiating env with {goal_num} goals and {obj_num} objs')
print(env.subtask_list)
print(env.goal_list)
action_seq = []

start_time = time.time()
root = Node()
curr_root = root

while not env.check_task() and len(action_seq) < 8:
    # MCTS
    best_node = mcts_search(curr_root)
    best_action, best_obj = best_node.action, best_node.obj

    # Take best action and record action_seq
    env.step({"goal": best_action, "tar_index": best_obj})
    action_seq.append({"goal": best_action, "tar_index": best_obj})
    print(f'Taking action {best_action}-{action_list[best_action]} \
        with obj {best_obj}-{env.object_index2name[best_obj]}')

    # Continue search from subtree
    curr_root = best_node

sim_time = time.time() - start_time

print_tree(root)
for pair in action_seq:
    action = pair['goal']
    obj = pair['tar_index']
    print(f'Took action {action}-{action_list[action]} \
        with obj {obj}-{env.object_index2name[obj]}')
if env.check_task():
    print("Task complete!")
else:
    print("Task failed")
print(f'Simulation time: {sim_time}')


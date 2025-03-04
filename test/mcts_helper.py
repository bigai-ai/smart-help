import math
import random
import time
from ai2thor.controller import Controller
from env.symbolic_env import SymbolicEnv
from env.goal import goal_list as action_list
from env.subtask import subtask_list, SUBTASK_NUM
#from action import action_list
from env.parse import parse_sub_task
from env.type import sample_type

#random.seed(2023)
sample_prob = 0.0



config = {
    'agents_num': 2,
    # 'agents_type': {0: AgentType.AGENT_WITH_FULL_CAPABILITIES, 1: AgentType.AGENT_WITH_TOGGLE_ISSUES},
    'agents_type': {0: [1,1,1,1,1,1], 1: sample_type()},
    'main_agent_id': 1,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 2,
        "scene": 'FloorPlan28',
        "local_executable_path": "/home/zhihao/下载/thor-Linux64-local/thor-Linux64-local",
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
    def __init__(self, parent=None, action=None, obj=None, env_obj_num=0):
        self.parent = parent
        self.action = action
        self.obj = obj
        self.children = []
        self.visits = 0
        self.value = 0
        self.env_obj_num = env_obj_num
    def is_fully_expanded(self):
        return len(self.children) == GOAL_NUM * self.env_obj_num


def ucb_score(parent_visits, child_value, child_visits):
    exploration_weight = 1.41  # sqrt(2) for UCB1
    if child_visits == 0:
        return float('inf')
    return child_value / child_visits + exploration_weight * math.sqrt(math.log(parent_visits) / child_visits)

def select_child(node, env, eps = sample_prob):
    ucb_scores = [ucb_score(node.visits, child.value, child.visits) for child in node.children]

    # Use rule at a certain prob
    if random.random() > eps and len(env.subtask_list) > 0:
        goal = parse_sub_task(env.subtask_list[0])[0]
        #print(f'Selecting goal {goal}')
        if goal[0] == 'Put' and goal[2] is not None:
            pair = [action_list.index(goal[0]), env.object_name2index[goal[2]]]
        else:
            pair = [action_list.index(goal[0]), env.object_name2index[goal[1]]]
        for child in node.children:
            if child.action == pair[0] and child.obj == pair[1]:
                return child

    # Or select child accoding to ucb score    
    return node.children[ucb_scores.index(max(ucb_scores))]

def expand(node, env, eps = sample_prob):
    unexplored_pairs = []
    explored_pairs = [[child.action, child.obj] for child in node.children]
    env_obj_num = len(env.object_index2name)

    for action in range(GOAL_NUM):
        for obj in range(env_obj_num):
            if [action, obj, None] not in explored_pairs:
                unexplored_pairs.append([action, obj, None])
    
    # Use rule at a certain prob
    if random.random() > eps and len(env.subtask_list) > 0:
        #print(env.subtask_list[0])
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
        
    child_node = Node(parent=node, action=pair[0], obj=pair[1], env_obj_num=env_obj_num)
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
    if not env.is_action_valid({"goal": node.action, "tar_index": node.obj}):
        return 0
    
    # Init
    max_simulation_depth = 5 
    value = 0 
    if len(env.subtask_list) == 0:
        return 0
    curr_subtask = env.subtask_list[0]
    if curr_subtask == ['In', 'Potato', 'Microwave']:
        node.action = 2
        node.obj = 24

    env.step({"goal": node.action, "tar_index": node.obj})
    #print(env.agents[0].pick_up_obj_id)
    # Rule out invalid actions
    if not env.controller.last_event.metadata['lastActionSuccess']:
        #print(env.agents[0].pick_up_obj_id)
        #print('Action failed')
        return 0
    
    #print(f'Simulating: action {node.action}-{action_list[node.action]}, \
       #obj {node.obj}-{env.object_index2name[node.obj]}')
    
    if env.check_sub_task(curr_subtask, agent_id=0):
        #print(f'Simulation: Subtask {curr_subtask} done')
        return 100
    
    # Rollout
    env_obj_num = len(env.object_index2name)
    for d in range(max_simulation_depth):
        action = random.randint(0, GOAL_NUM - 1)
        obj = random.randint(0, env_obj_num - 1)

        # TODO: Rule out invalid actions
        while not env.is_action_valid({"goal": action, "tar_index": obj}):
            action = random.randint(0, GOAL_NUM - 1)
            obj = random.randint(0, env_obj_num - 1)

        env.step({"goal": action, "tar_index": obj})
        # Evaluate
        if env.check_sub_task(curr_subtask, agent_id=0):
            #print(f'Simulation: Subtask {curr_subtask} done')
            value += (100 / ((d+2)*2)) # Discount by simulation steps
            break

    return value

def backprop(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        node = node.parent

def mcts_search(root, env, action_seq = [], subtask = None, agent_type = None, num_simulations=100, sample_prob=sample_prob):
    # TODO: num_simulations
    for _ in range(num_simulations):
        env.reset_to_init()
        for action in action_seq:
            env.step(action)
            #print(f'Previous step: {action}')
        if subtask is not None:
            env.subtask_list = [subtask]
        #if type is not None:
        #    env._agent_type[0] = agent_type[0]
        env_obj_num = len(env.object_index2name)

        # Select and expand
        node = root
        parent_seq = []
        while node.is_fully_expanded() and node.children:
            parent_seq.append(node)
            node = select_child(node, env, sample_prob)
        if not node.is_fully_expanded():
            parent_seq.append(node)
            node = expand(node, env, sample_prob)
   
        for parent in parent_seq[1:]:
            env.step({"goal": parent.action, "tar_index": parent.obj})
            #print(f'Parent step: action {parent.action}, obj {parent.obj}')

        value = simulate(node, env)
        backprop(node, value)

    best_node = max(root.children, key=lambda child: child.visits)
    #print_tree(env, root)
    return best_node

# Debug: Print tree
from collections import deque
def print_tree(env, root):
    if root is None:
        return
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if node.action is not None:
            print(f'Node: Action: {node.action}-{action_list[node.action]}, \
            Obj {node.obj}-{env.object_index2name[node.obj]}, Value {node.value}') 
        for child in node.children:
            queue.append(child)

GOAL_NUM = len(action_list)
def run_mcts(env, action_seq = [], inferred_subtask=None, inferred_type=None, num_sim=100, sample_prob=0.5):
        # Wait if unknown
        if inferred_subtask[0] == "Unknown":
            return {"goal": 0, "tar_index": 0} # Wait if unknown

        # Init tree
        env_obj_num = len(env.object_index2name)
        root = Node(env_obj_num=env_obj_num)

        # Random subtask
        if inferred_subtask is None:
            subtask_pairs = []
            for state in range(SUBTASK_NUM):
                for obj in range(env_obj_num):
                    if subtask_list[state] == "On" or subtask_list[state] == "In":
                        for obj2 in range(env_obj_num):
                            subtask_pairs.append([subtask_list[state], \
                                                  env.object_index2name[obj], env.object_index2name[obj2]])
                    else:
                        subtask_pairs.append([subtask_list[state], \
                                              env.object_index2name[obj], None])
            inferred_subtask = random.choice(subtask_pairs) 

        # MCTS
        best_node = mcts_search(root, env, action_seq, inferred_subtask, inferred_type, num_sim, sample_prob)
        best_action, best_obj = best_node.action, best_node.obj

        output_action = {"goal": best_action, "tar_index": best_obj}
        return output_action

if __name__ == '__main__':

    env = SymbolicEnv(config=config)
    GOAL_NUM = len(action_list)
    OBJ_NUM = len(env.object_index2name)
    
    
    print(f'Initiating env with {GOAL_NUM} goals and {OBJ_NUM} objs')
    print(env.subtask_list)
    print(env.goal_list)
    action_seq = []

    # Call this function in test_mcts.py

    start_time = time.time()
    # Init tree
    root = Node()
    curr_root = root

    while not env.check_task() and len(action_seq) < 5:
        # MCTS
        best_node = mcts_search(curr_root, env, action_seq = [], subtask = env.subtask_list[0])
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


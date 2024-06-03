# Smart Help: Strategic Opponent Modeling for Proactive and Adaptive Robot Assistance in Households
 
This is the official implementation of the paper [*Smart Help: Strategic Opponent Modeling for Proactive and Adaptive Robot Assistance in Households*](https://arxiv.org/abs/2404.09001). 

Our main contributions can be summarized as follows:

We propose a novel Smart Help challenge that aims at learning to provide both proactive and adaptive help to diverse human users (especially vulnerable groups) based on inferred goals and capabilities.

To the best of our knowledge, we contribute the first 3D realistic home environment built upon AI2-THOR, that focuses on assisting the vulnerable group with daily household tasks in an online and interactive manner. 

We provide a benchmark model with a joint goal and capability inference, bottleneck reasoning, and helping policy improvement. Strict experiments and the proposed holistic evaluations validate the efficacy of our model.

## Setup

### Install Requirements 
```bash
pip install -r requirements.txt
```

### Download Checkpoints

The checkpoints and dataset can be found in https://huggingface.co/datasets/bigai/SmartHelp/tree/main

## Code Explanation

### Environment

The environments can be found in the folder env. 

Symbolic_env.py provide an environment with symbolic observation. For a step, your input should be: 
{
    "goal": action_index(int), 
    "tar_index": obj_index(int)
}

Here the goal represents the intentional action mentioned in the paper. 

The full action list can be found in env/goal.py, and the tar_index can be found in object_name2index.json. 

### Observation Space

The symbolic observation space is: 

[
    [obj_index, parent_receptacle_index, isPickedUp, isOpen, isCooked, isToggled, is_visible(by the agent 1), is_visible(by the agent 2), object_height, obj_weight, obj_position(x, y, z), obj_distance(from the agent), 1(indicate that this line is an object)],<br>
    ...(totally object * 30), 
    [agent_pos(x, y, z), agent_rotation(x, y, z), agent_action_index, agent_action_obj, agent_action_success, agent_picked_up_id],<br>, 
    ...(totally agent * 2)
]

### Training

The main.py is used to train the BaseModel, and the main_finetune.py is used to train the BaseModel-PL. 

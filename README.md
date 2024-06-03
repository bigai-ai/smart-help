# Smart Help: Strategic Opponent Modeling for Proactive and Adaptive Robot Assistance in Households
 
This is the official implementation of the paper [*Smart Help: Strategic Opponent Modeling for Proactive and Adaptive Robot Assistance in Households*](https://arxiv.org/abs/2404.09001). 

## Abstract

Despite the significant demand for assistive technology among vulnerable groups (e.g., the elderly, children, and the disabled) in daily tasks, research into advanced AI-driven assistive solutions that genuinely accommodate their diverse needs remains sparse. Traditional human-machine interaction tasks often require machines to simply help without nuanced consideration of human abilities and feelings, such as their opportunity for practice and learning, sense of self-improvement, and self-esteem. Addressing this gap, we define a pivotal and novel challenge Smart Help, which aims to provide proactive yet adaptive support to human agents with diverse disabilities and dynamic goals in various tasks and environments. To establish this challenge, we leverage AI2-THOR to build a new interactive 3D realistic household environment for the Smart Help task. We introduce an innovative opponent modeling module that provides a nuanced understanding of the main agent's capabilities and goals, in order to optimize the assisting agent's helping policy. Rigorous experiments validate the efficacy of our model components and show the superiority of our holistic approach against established baselines. Our findings illustrate the potential of AI-imbued assistive robots in improving the well-being of vulnerable groups.

<img width="895" alt="image" src="https://github.com/bigai-ai/smart-help/assets/18099927/86e1930e-c38f-4734-a182-3e30ffeea560">

Our main **contributions** can be summarized as follows:

1）We propose a novel Smart Help challenge that aims at learning to provide both proactive and adaptive help to diverse human users (especially vulnerable groups) based on inferred goals and capabilities.

2）To the best of our knowledge, we contribute the first 3D realistic home environment built upon AI2-THOR, that focuses on assisting the vulnerable group with daily household tasks in an online and interactive manner. 

3）We provide a benchmark model with a joint goal and capability inference, bottleneck reasoning, and helping policy improvement. Strict experiments and the proposed holistic evaluations validate the efficacy of our model.

## Dataset

Visit the following huggingface site to download our dataset: https://huggingface.co/datasets/bigai/SmartHelp/tree/main

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

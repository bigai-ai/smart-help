from env.single_env_symbolic import Single_Env_Symbolic


config = {
    'agents_num': 1,
    'agents_type': 'rllib',
    'main_agent_id': 0,
    'mode': 'train',
    'controller_kwargs': {
        "agentCount": 1,
        "scene": 'FloorPlan2',
        "local_executable_path": "/home/zhihao/A2SP/thor-Linux64-local/thor-Linux64-local",
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "visibilityDistance": 6, 
    },
    # 'task': ['move_to', 'Apple|-00.93|+01.15|+00.95']
    # todo id change?
    # 'task': ['get', 'Tomato|+00.17|+00.97|-00.28'],
    'task': ['put_on', 'Tomato|+00.17|+00.97|-00.28', 'Sink|+00.00|+00.89|-01.44'],
}

for i in range(30):
    scene_name = "FloorPlan%d" % (i + 1)
    config['controller_kwargs']['scene'] = scene_name
    Single_Env_Symbolic(config=config)

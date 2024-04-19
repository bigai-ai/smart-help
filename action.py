
# Place 'Wait' first because the empty action in obs is 0
action_list = [
    'Wait',
    'MoveAhead',
    'RotateRight',
    'RotateLeft',
    'LookUp',
    'LookDown',
    'PickUp',
    'Drop',
    'Put',
    'Open',
    'Close',
    'Slice',
    'ToggleOn',
    'ToggleOff',
    'Stand',
    'Crouch',
    'Teleport',
]

def MoveAhead(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='MoveAhead',
        moveMagnitude=0.25,
        agentId=agent_id,
        # disableRendering = disableRendering ,
        # forceAction=True
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def RotateRight(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='RotateRight',
        degrees=90,
        agentId=agent_id,
        # disableRendering = disableRendering ,
        forceAction=True
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def RotateLeft(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='RotateLeft',
        degrees=90,
        agentId=agent_id,
        # disableRendering = disableRendering ,
        forceAction=True
    ) 
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def LookUp(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='LookUp',
        degrees=30,
        agentId=agent_id,
        # disableRendering = disableRendering ,
        forceAction=True
    ) 
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def LookDown(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='LookDown',
        degrees=30,
        agentId=agent_id,
        # disableRendering = disableRendering ,
        forceAction=True
    ) 
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

# distance limited 
def PickUp(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    # print('[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]', selected_obj_id)
    # print(selected_obj_id)
    event = env.controller.step(
        action='PickupObject',
        objectId=selected_obj_id, 
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    # print('++++++++++++++++++++++++++++++++++')
    if env.controller.last_event.metadata['lastActionSuccess']:
        env.agents[agent_id].pick_up_obj_id = selected_obj_id
    #     print('------------------------------success------------')
    else:
        # print('=========================', env.controller.last_event.metadata['errorMessage'])
        return False
        
    return event 

def Drop(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    
    event = env.controller.step(
        action='DropHandObject',
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        env.agents[agent_id].pick_up_obj_id = None
    return event

def Put(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    event = env.controller.step(
        action='PutObject',
        objectId = selected_obj_id, 
        # receptacleObjectId = selected_obj_id[1],
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    # print("for debug action Put", event.metadata['errorMessage'])
    if env.controller.last_event.metadata['lastActionSuccess']:
        env.agents[agent_id].pick_up_obj_id = None
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Stand(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute Stand Action
    '''
    # if env.agents_state[agent_id]['standing'] == True:
    #     return False
    event = env.controller.step(
        action="Stand",
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    # if env.last_event.events[agent_id].metadata['lastActionSuccess']:
    #     assert(
    #         env.last_event.events[agent_id].metadata['lastAction'] == 'Stand'
    #     ), f'last action is {env.last_event.events[agent_id].metadata["lastAction"]}. Wrong! The expected action "Stand" is not performed!'
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Crouch(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute Crouch Action
    '''
    # if env.agents_state[agent_id]['standing'] == False:
    #     return False
    event = env.controller.step(
        action="Crouch",
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    # if env.last_event.events[agent_id].metadata['lastActionSuccess']:
    #     assert(
    #         env.last_event.events[agent_id].metadata['lastAction'] == 'Crouch'
    #     ), f'last action is {env.last_event.events[agent_id].metadata["lastAction"]}. Wrong! The expected action "Crouch" is not performed!'
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Open(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute Open Action
    In the future, these two ways of representation may be unified. 
    '''
    if "Blinds" in selected_obj_id:
        return False
    event = env.controller.step(
        action='OpenObject',
        objectId=selected_obj_id,
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    
    # if env.last_event.events[agent_id].metadata['lastActionSuccess']:
    #     assert(
    #         env.last_event.events[agent_id].metadata['lastAction'] == 'OpenObject'
    #     ), f'last action is {env.last_event.events[agent_id].metadata["lastAction"]}. Wrong! The expected action "OpenObject" is not performed!'
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Close(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute Close Action
    In the future, these two ways of representation may be unified. 
    '''
    if "Blinds" in selected_obj_id:
        return False
    event = env.controller.step(
        action='CloseObject',
        objectId=selected_obj_id,
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Slice(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute Slice Action
    In the future, these two ways of representation may be unified. 
    '''
    event = env.controller.step(
        action='SliceObject',
        objectId=selected_obj_id,
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def ToggleOn(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute ToggleOn Action
    In the future, these two ways of representation may be unified. 
    '''
    event = env.controller.step(
        action='ToggleObjectOn',
        objectId=selected_obj_id,
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering ,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def ToggleOff(env, selected_obj_id, agent_id, disableRendering = True) -> bool:
    '''
    Execute ToggleOff Action
    In the future, these two ways of representation may be unified. 
    '''
    event = env.controller.step(
        action='ToggleObjectOff',
        objectId=selected_obj_id,
        forceAction=True,
        agentId=agent_id,
        # disableRendering = disableRendering,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event 

def Wait(env, agent_id, _, disableRendering = True) -> bool:
    '''
    Execute Wait Action
    '''
    return False

def Teleport(env, target_pose, agent_id, disableRenderting = True):
    '''
    Execute Teleport Action
    '''
    event = env.controller.step(
        action='Teleport',
        position=dict(x=target_pose[0], y=target_pose[1], z=target_pose[2]),
        rotation=dict(x=target_pose[6], y=target_pose[3], z=0),
        horizon=target_pose[4],
        standing=target_pose[5],
        agentId=agent_id,
        forceAction=True,
        # disableRendering = disableRendering ,
    )
    if env.controller.last_event.metadata['lastActionSuccess']:
        pass
    else:
        return False
    return event
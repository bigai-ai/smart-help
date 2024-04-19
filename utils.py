import numpy as np
import random
import math
from ai2thor.controller import Controller
from typing import Optional, List, Dict
import copy
from constants import (
    AgentType, 
    SceneType, 
    VISIBILITY_DISTANCE,
    SCREEN_SIZE,
    EXCLUDED_RECEPTACLES_TYPE_IN_DATASET, 
    OPENABLE_RECEPTACLES_TYPE, 
    PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE, 
    COOK_RECEPTACLES, 
    CLEANABLE_OBJECTS_TYPE
)

def distance_2D(position_1: dict, position_2: dict) -> float:
    '''
    This function is used to compute the 2D distance between two position.
    The coordinate of the position is described as (x, z) and the distacne is the Euclidean Distance. 
    '''
    dist = math.sqrt(sum([(position_1[key] - position_2[key]) ** 2 for key in ["x", "z"]]))
    return dist

def distance_3D(position_1: dict, position_2: dict) -> float:
    '''
    This function is used to compute the 3D distance between two position.
    The coordinate of the position is described as (x, y, z) and the distacne is the Euclidean Distance. 
    '''
    dist = math.sqrt(sum([(position_1[key] - position_2[key]) ** 2 for key in ["x", "y", "z"]]))
    return dist

def find_closest_position(obj_position: dict, reachable_positions: list, except_pos: list):
    '''
        - obj_position: the position of a special object, it should contain x and z at least.
        - reachable_positions: all the reachable positions in the scene of the agent, 
            it is usually obtained directly from the AI2THOR controller. 
    '''
    dist_2_pos = {}
    for position in reachable_positions:
        if (round(position['x'], 2), round(position['z'], 2)) in except_pos:
            # print((round(position['x'], 2), round(position['z'], 2)))
            # print(except_pos)
            continue
        dist = distance_2D(obj_position, position)
        dist_2_pos[dist] = position
    closest_candidates = sorted(dist_2_pos.items(), key=lambda x: x[0])[0 : 4]
    closest_pos = random.choice(closest_candidates)[1]
    return closest_pos

def distance_min_coordinate(position_1: dict, position_2: dict) -> float:
    dist = min([abs(position_1[key] - position_2[key]) for key in ['x', 'y', 'z']])
    return dist


def find_closest_object(target_obj_pos, candidate_objs_id_2_pos):
    dist_2_candidates = {}
    for candidate, position in candidate_objs_id_2_pos.items():
        dist = distance_min_coordinate(target_obj_pos, position)
        dist_2_candidates[dist] = candidate
    closest_candidate = sorted(dist_2_candidates.items(), key=lambda x: x[0])[0][1]
    return closest_candidate
    
def find_parent_receptacle(obj_pos: Dict, parent_receptacles: List[str] = None):
    if parent_receptacles == None:
        parent_receptacle = None
    elif len(parent_receptacles) > 1:
        candidate_obj_2_positions = {}
        for receptacle in parent_receptacles:
            receptacle_pos = {
                'x': float(receptacle.split('|')[1]),
                'y': float(receptacle.split('|')[2]),
                'z': float(receptacle.split('|')[3]),
            }
            candidate_obj_2_positions[receptacle] = receptacle_pos
        parent_receptacle = find_closest_object(obj_pos, candidate_obj_2_positions)
    else:
        parent_receptacle = parent_receptacles[0]
    return parent_receptacle

def find_parent_receptacle_plus(
    obj_id: str, 
    parent_receptacles: List[str],
    objects: List, 
    obj_id_2_id: Dict,
    disabled_objs_id: Optional[List] = []
):  
    if parent_receptacles == None:
        return None
    elif len(parent_receptacles) == 1:
        return parent_receptacles[0]
    else: 
        obj_full_state = objects[obj_id_2_id[obj_id]]
        obj_bbx_max = obj_full_state['axisAlignedBoundingBox']['cornerPoints'][0]
        obj_bbx_min = obj_full_state['axisAlignedBoundingBox']['cornerPoints'][-1]
        obj_pos = obj_full_state['position']
        
        score_2_parent_receptacles_id = {}
        for _recep_obj_id in parent_receptacles:
            # if _recep_obj_id not in disabled_objs_id:
            _recep_obj_full_state = objects[obj_id_2_id[_recep_obj_id]]
            if _recep_obj_full_state['axisAlignedBoundingBox']['cornerPoints'] == None:
                continue
            _recep_obj_bbx_max = _recep_obj_full_state['axisAlignedBoundingBox']['cornerPoints'][0]
            _recep_obj_bbx_min = _recep_obj_full_state['axisAlignedBoundingBox']['cornerPoints'][-1]
            # _pos_center = _obj_full_state['axisAlignedBoundingBox']['center']
            _pos_center = _recep_obj_full_state['position']
            pseudoIoU_3D = compute_pseudoIoU_3D(
                obj_bbx_max, obj_bbx_min, _recep_obj_bbx_max, _recep_obj_bbx_min
            )
            distance = distance_min_coordinate(obj_pos, _pos_center)
            score = distance - pseudoIoU_3D
            score_2_parent_receptacles_id[score] = _recep_obj_id
        try:
            return sorted(score_2_parent_receptacles_id.items(), key=lambda x: x[0])[0][1]
        except:
            print(parent_receptacles)
            print(disabled_objs_id)
            print(score_2_parent_receptacles_id)
            return None
        


def compute_pseudoIoU_3D(bbx_max_1, bbx_min_1, bbx_max_2, bbx_min_2):
    x_len = min(bbx_max_1[0], bbx_max_2[0]) - max(bbx_min_1[0], bbx_min_2[0])
    y_len = min(bbx_max_1[1], bbx_max_2[1]) - max(bbx_min_1[1], bbx_min_2[1])
    z_len = min(bbx_max_1[2], bbx_max_2[2]) - max(bbx_min_1[2], bbx_min_2[2])

    V = (bbx_max_1[0] - bbx_min_1[0]) * (bbx_max_1[1] - bbx_min_1[1]) * (bbx_max_1[2] - bbx_min_1[2]) 

    pseudoIoU = x_len * y_len * z_len / V
    if pseudoIoU > 0:
        return pseudoIoU
    else:
        return 0


def compute_face_2_pos(objId: str, obj_pos: dict, reachable_positions: list) -> dict:
    '''
    This function will not directly teleport the agent to the position, 
    alternatively computes the pose and correction of the agent accuragely . 
        - objId: the Id of the object
        - obj_pos: the position of the object
        - reachable_positions: all the reachable positions in the scene of the agent, 
            it is usually obtained directly from the AI2THOR controller. 
    The return of the function is: 
        - the pose of the agent: position (x, y, z), rotation, is_standing, horizon.
        - the correction of the pose, which used to correct the agent' s rotation
    '''
    obj_x, obj_y, obj_z = obj_pos['x'], obj_pos['y'], obj_pos['z']
    agent_pos = find_closest_position(obj_pos, reachable_positions)
    agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']
    # controller.step(
    #     action='Teleport',
    #     position=position,
    #     rotation=dict(x=0, y=0, z=0),
    #     horizon=0.0,
    #     standing=True,
    # )  
    # compute the rotation that turns the agent face to the object
    yaw_appropriate = round((np.arctan2(-agent_x + obj_x, -agent_z + obj_z) / np.pi * 180) % 360, 1)
    # rotate the agent
    # need to rotate right
    # controller.step(action="RotateRight", degrees=yaw)

    # modification and correction
    yaw_modified = 90.0 * round(yaw_appropriate / 90.0)
    correction = yaw_appropriate - yaw_modified     

    # look down by 30 degrees
    if (round(obj_y, 1) < 1.2):
        horizon = 30
    elif (round(obj_y, 1) < 0.6):
        horizon = 60
    # look up by 30 degrees
    elif (round(obj_y, 1) > 1.8):
        horizon = -30
    else:
        horizon = 0
    # move backward to have wider view
    if 'Drawer' in objId:
        # move back operation
        direction = round(yaw_appropriate / 90.0)
        if direction == 0:
            agent_z -= 0.25
        elif direction == 1:
            agent_x -= 0.25
        elif direction == 2:
            agent_z += 0.25
        elif direction == 3:
            agent_x += 0.25 
        if dict(x=agent_x, y=agent_y, z=agent_z) not in reachable_positions:
            agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']  
    pose = {
        'x': agent_x,
        'y': agent_y,
        'z': agent_z,
        'rotation': yaw_modified,
        'standing': True,
        'horizon': horizon,
    }
    return pose, correction

# remove correction
def compute_face_2_pos_rough(objId: str, obj_pos: dict, reachable_positions: list) -> dict:
    '''
    This function will not directly teleport the agent to the position, 
    alternatively computes the pose and correction of the agent roughly . 
        - objId: the Id of the object
        - obj_pos: the position of the object
        - reachable_positions: all the reachable positions in the scene of the agent, 
            it is usually obtained directly from the AI2THOR controller. 

    The return of the function is: 
        - the pose of the agent: position (x, y, z), rotation, is_standing, horizon.
    Note that:
        - the coordinates of the position are all multiples of 0.25 (e.g., 0.25, 0.5, 0.75)
        - the rotation is the multiple of 90 (e.g., 0, 90, 180, 270)
        - the horizon is the multiple of 30 (e.g., 30, -30) and the positive means looking up, 
            while the negative means looking down. 
    '''
    obj_x, obj_y, obj_z = obj_pos['x'], obj_pos['y'], obj_pos['z']
    agent_pos = find_closest_position(obj_pos, reachable_positions)
    agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']
    # controller.step(
    #     action='Teleport',
    #     position=position,
    #     rotation=dict(x=0, y=0, z=0),
    #     horizon=0.0,
    #     standing=True,
    # )  
    # compute the rotation that turns the agent face to the object
    yaw_appropriate = round((np.arctan2(-agent_x + obj_x, -agent_z + obj_z) / np.pi * 180) % 360, 1)
    yaw_modified = 90.0 * round(yaw_appropriate / 90.0)

    # look down by 30 degrees
    if (round(obj_y, 1) < 1.2):
        horizon = 30
    elif (round(obj_y, 1) < 0.6):
        horizon = 60
    # look up by 30 degrees
    elif (round(obj_y, 1) > 1.8):
        horizon = -30
    else:
        horizon = 0
    # move backward to have wider view
    if 'Drawer' in objId:
        # move back operation
        direction = round(yaw_appropriate / 90.0)
        if direction == 0:
            agent_z -= 0.25
        elif direction == 1:
            agent_x -= 0.25
        elif direction == 2:
            agent_z += 0.25
        elif direction == 3:
            agent_x += 0.25 
        if dict(x=agent_x, y=agent_y, z=agent_z) not in reachable_positions:
            agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']  
    pose = {
        'x': agent_x,
        'y': agent_y,
        'z': agent_z,
        'rotation': yaw_modified,
        'standing': True,
        'horizon': horizon,
    }
    return pose

def compute_face_2_pos_rough_plus(obj_id: str, obj_pos: dict, reachable_positions: list, except_pos: list) -> dict:
    '''
    This function will not directly teleport the agent to the position, 
    alternatively computes the pose and correction of the agent roughly . 
        - objId: the Id of the object
        - obj_pos: the position of the object
        - reachable_positions: all the reachable positions in the scene of the agent, 
            it is usually obtained directly from the AI2THOR controller. 

    The return of the function is: 
        - the pose of the agent: position (x, y, z), rotation, is_standing, horizon.
    Note that:
        - the coordinates of the position are all multiples of 0.25 (e.g., 0.25, 0.5, 0.75)
        - the rotation is the multiple of 90 (e.g., 0, 90, 180, 270)
        - the horizon is the multiple of 30 (e.g., 30, -30) and the positive means looking up, 
            while the negative means looking down. 
    '''
    obj_x, obj_y, obj_z = obj_pos['x'], obj_pos['y'], obj_pos['z']
    agent_pos = find_closest_position(obj_pos, reachable_positions, except_pos)
    agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']
    # controller.step(
    #     action='Teleport',
    #     position=position,
    #     rotation=dict(x=0, y=0, z=0),
    #     horizon=0.0,
    #     standing=True,
    # )  
    # compute the rotation that turns the agent face to the object
    yaw_appropriate = round((np.arctan2(-agent_x + obj_x, -agent_z + obj_z) / np.pi * 180) % 360, 1)
    yaw_modified = 90.0 * round(yaw_appropriate / 90.0)

    standing = True 
    # crouch
    if (round(obj_y, 1) < 0.7):
        standing = False
        horizon = 30
    # look down by 30 degrees
    elif (round(obj_y, 1) < 1.2):
        horizon = 30
    # look up by 30 degrees
    elif (round(obj_y, 1) > 1.8):
        horizon = -30
    else:
        horizon = 0
    # move backward to have wider view
    if 'Drawer' in obj_id:
        # move back operation
        direction = round(yaw_appropriate / 90.0)
        if direction == 0:
            agent_z -= 0.25
        elif direction == 1:
            agent_x -= 0.25
        elif direction == 2:
            agent_z += 0.25
        elif direction == 3:
            agent_x += 0.25 
        if dict(x=agent_x, y=agent_y, z=agent_z) not in reachable_positions:
            agent_x, agent_y, agent_z = agent_pos['x'], agent_pos['y'], agent_pos['z']  
    pose = {
        'x': agent_x,
        'y': agent_y,
        'z': agent_z,
        'rotation': yaw_modified,
        'standing': standing,
        'horizon': horizon,
    }
    return pose


def teleport_face_2_pos(
    controller: Controller, 
    obj_id: str, 
    obj_pos: dict, 
    reachable_positions: list
):
    pose = compute_face_2_pos_rough_plus(obj_id, obj_pos, reachable_positions)
    controller.step(
        action="Teleport",
        position=dict(x=pose['x'], y=pose['y'], z=pose['z']),
        rotation=dict(x=0, y=pose['rotation'], z=0),
        horizon=pose['horizon'],
        standing=pose['standing']
    )


def judge_visible_1(obj_id: str, recep_id: str, observation: dict, 
                controller: Controller, agent_id: int, scene_state: dict, 
                visible_distance: float, width: int, height: int):
    obj_visible = False
    recep_visible = False
    agent_info = controller.last_event.events[agent_id].metadata['agent']
    agent_2D_pos = {
        'x': round(agent_info['position']['x'], 2),
        'z': round(agent_info['position']['z'], 2),
    }
    if obj_id in observation.keys():
        obj_2D_pos = {
            'x': round(scene_state[obj_id]['obj_pos']['x'], 2),
            'z': round(scene_state[obj_id]['obj_pos']['z'], 2)
        }
        distance = distance_2D(agent_2D_pos, obj_2D_pos)
        if distance < visible_distance:
            IOU = observation[obj_id][0.1*width: 0.9*width, 0.1*height: 0.9*height]
            ratio = np.count_nonzero(IOU) / np.count_nonzero(observation[obj_id])
            if ratio > 0.6:
                obj_visible = True
    if recep_id in observation.keys():
        recep_2D_pos = {
            'x': round(scene_state[recep_id]['obj_pos']['x'], 2),
            'z': round(scene_state[recep_id]['obj_pos']['z'], 2)
        }
        distance = distance_2D(agent_2D_pos, recep_2D_pos)
        if distance < visible_distance:
            IOU = observation[recep_id][0.1*width: 0.9*width, 0.1*height: 0.9*height]
            ratio = np.count_nonzero(IOU) / np.count_nonzero(observation[recep_id])
            if ratio > 0.6:
                recep_visible = True
    return obj_visible, recep_visible

def judge_visible(
    observation_instance_masks: dict, agent_3D_pos: dict, 
    scene_state: dict, width: int = SCREEN_SIZE, height: int = SCREEN_SIZE, 
    visible_distance: float = VISIBILITY_DISTANCE, threshold: float = 0.6
) -> List[bool]:
    '''
    agent_3D_pos = {
        'x': x_coordinate,
        'y': y_coordinate,
        'z': z_coordinate,
    }
    '''
    objs_visible = []
    for obj_id, obj_state in scene_state.items():
        if isinstance(obj_state, dict) == False:
                continue
        obj_visible = False
        if obj_id in observation_instance_masks.keys():
            obj_3D_pos = {
                'x': round(scene_state[obj_id]['obj_pos']['x'], 2),
                'y': round(scene_state[obj_id]['obj_pos']['y'], 2),
                'z': round(scene_state[obj_id]['obj_pos']['z'], 2),
            }
            distance = distance_3D(agent_3D_pos, obj_3D_pos)
            if distance < visible_distance:
                # IoU = observation_instance_masks[obj_id][int(0.1*width): int(0.9*width), int(0.1*height): int(0.9*height)]
                # ratio = np.count_nonzero(IoU) / np.count_nonzero(observation_instance_masks[obj_id])
                # if ratio > threshold:
                #     obj_visible = True
                if np.count_nonzero(observation_instance_masks[obj_id]) > 0:
                    obj_visible = True
        objs_visible.append(obj_visible)
    return objs_visible


def disable_objs_of_receptacle(
    controller: Controller, of_receptacle_objs_id: List[str], 
    excluded_objs_id: Optional[List[str]] = [],
    excluded_objs_type: Optional[List[str]] = [],
):
    disabled_objs_id = []
    if (of_receptacle_objs_id != None and of_receptacle_objs_id != []):
        for obj_id in of_receptacle_objs_id:
            if (
                (obj_id_2_obj_type(obj_id) not in excluded_objs_type)
                and (obj_id not in excluded_objs_id)
            ):
                controller.step(
                    action="DisableObject",
                    objectId=obj_id
                )
                disabled_objs_id.append(obj_id)
    return disabled_objs_id

def obj_id_2_obj_type(obj_id: str) -> str:
    obj_id_split = obj_id.split('|')    
    if len(obj_id_split) == 4:   # for normal obj_id, e.g., Sink|+00.16|+00.82|-01.80, Bread|-00.96|+00.77|+01.10
        return obj_id_split[0]
    elif len(obj_id_split) == 5: # for special obj_id, e.g., Bread|-00.96|+00.77|+01.10|BreadSliced_1, Sink|-01.80|+00.92|+03.66|SinkBasin
        return obj_id_split[-1].split('_')[0]

def scene_name_2_scene_type(scene_name: str) -> str:
    import re
    scene_id = int(re.findall("\d+",scene_name)[0])
    if scene_id in list(range(1, 31)):
        return 'kitchen'
    elif scene_id in list(range(201, 231)):
        return 'living_room'
    elif scene_id in list(range(301, 331)):
        return 'bedroom'
    elif scene_id in list(range(401, 431)):
        return 'bathroom'
    else:
        raise NotImplementedError

def agent_type_2_str(agent_type: AgentType) -> str:
    if agent_type == AgentType.AGENT_WITH_FULL_CAPABILITIES:
        return 'full capabilities'
    elif agent_type == AgentType.AGENT_WITH_CARELESS_MIND:
        return 'careless mind'
    elif agent_type == AgentType.AGENT_WITH_PICKUP_ISSUES:
        return 'manipulation issues'
    elif agent_type == AgentType.AGENT_WITH_VISIBILITY_ISSUES:
        return 'visibility issues'
    elif agent_type == AgentType.AGENT_WITH_MOBILITY_ISSUES:
        return 'mobility issues'
    elif agent_type == AgentType.AGENT_HELPER:
        return 'helper with full capabilities'



if __name__ == '__main__':
    from ai2thor.platform import CloudRendering

    controller = Controller(
        agentCount = 2,
        scene='FloorPlan1',
        platform=CloudRendering,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )
   

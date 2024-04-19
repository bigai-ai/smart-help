from env.task import PICKUP_ABLE_OBJECT, OPEN_ABLE_OBJECT, PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE
from random import sample
from env.utils import find_parent_receptacle_plus


def parse(task):
    sub_task_list = parse_task(task)
    goal_list = []
    for sub_task in sub_task_list:
        goal_list += parse_sub_task(sub_task)
    return goal_list


# Task: EatSomething
def parse_task(task, env):
    # for now, we only have subtask
    sub_task_list = []
    if task[0] == "MakeBreakfast":
        sub_task_list.append(['Get', "Potato"])
        sub_task_list.append(["ToggleOff", "Microwave"])
        sub_task_list.append(["Open", "Microwave"])
        sub_task_list.append(["In", "Potato", "Microwave"])
        sub_task_list.append(["Close", "Microwave"])
        sub_task_list.append(["ToggleOn", "Microwave"])
        sub_task_list.append(["ToggleOff", "Microwave"])
        # sub_task_list.append(["Open", "Microwave"])
    elif task[0] == "ArrangeRoom":
        # sample several object to put in/on
        obj_list = sample(PICKUP_ABLE_OBJECT, 3)
        for i in range(3):
            container = sample(PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE[obj_list[i]], 1)[0]
            obj = obj_list[i]
            parent_recp_list = env.metadata()['objects'][env.find_index(obj)]['parentReceptacles']
            if parent_recp_list is None or len(parent_recp_list) == 0:
                pass
            else:
                parent_recp = find_parent_receptacle_plus(env.find_id(obj), parent_recp_list, env.metadata()['objects'], env.object_id2index)
                if env.metadata()['objects'][env.find_index(parent_recp)]['openable']:
                    sub_task_list.append(["Open", parent_recp[:parent_recp.find('|')]])
            sub_task_list.append(["Get", obj_list[i]])
            if container in OPEN_ABLE_OBJECT:
                sub_task_list.append(["Open", container])
            sub_task_list.append(["In", obj_list[i], container])
            if container in OPEN_ABLE_OBJECT:
                sub_task_list.append(["Close", container])
    elif task[0] == "MakeCoffee":
        recept_obj_list = env.metadata()['objects'][env.find_index("CoffeeMachine")]['receptacleObjectIds']
        if recept_obj_list is not None and len(recept_obj_list) > 0:
            obj = recept_obj_list[0][:recept_obj_list[0].index("|")]
            sub_task_list.append(["ToggleOn", "CoffeeMachine"])
            sub_task_list.append(["ToggleOff", "CoffeeMachine"])
            # sub_task_list.append(["Get", "Mug"])
        else:
            sub_task_list.append(["Get", "Mug"])
            sub_task_list.append(["On", "Mug", "CoffeeMachine"])
            sub_task_list.append(["ToggleOn", "CoffeeMachine"])
            sub_task_list.append(["ToggleOff", "CoffeeMachine"])
            # sub_task_list.append(["Get", "Mug"])
    if len(sub_task_list) == 0:
        sub_task_list = [task]
    return sub_task_list


def parse_sub_task(subtask):
    goal_list = []
    if subtask[0] == 'PutOn' or subtask[0] == "On":
        sth_1 = subtask[1]
        sth_2 = subtask[2]
        goal_list.append(['PickUp', sth_1])
        goal_list.append(['Put', sth_1, sth_2])
    elif subtask[0] == 'PickUp' or subtask[0] == "Get":
        sth_1 = subtask[1]
        goal_list.append(['PickUp', sth_1])
    elif subtask[0] == 'In':
        goal_list.append(['PickUp', subtask[1]])
        goal_list.append(['Put', subtask[1], subtask[2]])
    elif subtask[0] == 'ToggleOn':
        goal_list.append(['ToggleOn', subtask[1]])
    elif subtask[0] == 'ToggleOff':
        goal_list.append(['ToggleOff', subtask[1]])
    elif subtask[0] == "Cook":
        goal_list.append(["ToggleOn", "Microwave"])
    elif subtask[0] == "Open":
        goal_list.append(["Open", subtask[1]])
    elif subtask[0] == "Close":
        goal_list.append(["Close", subtask[1]])
    return goal_list



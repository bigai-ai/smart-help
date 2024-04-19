import random
import json

def sample_task(task=None):
    # 分几类sample
    # Micro wave
    # 单个task已经足够复杂了
    seed = random.random()

    # return ["ArrangeRoom"]

    # for data gen

    # if seed < 0.5:
    #     return ["MakeBreakfast"]
    # else:
    #     return ["MakeCoffee"]

    if task is not None:
        return [task]

    if seed < 0.2:
        return ["MakeBreakfast"]
    elif seed < 0.8:
        return ["ArrangeRoom"]
    else:
        return ["MakeCoffee"]
    
# PICKUP_ABLE_OBJECT = ["Apple", "Bowl", "Bread", "ButterKnife",
#                        "Cup", "DishSponge", "Egg", "Fork", "Knife", "Lettuce", "Mug",
#                          "Pan", "PepperShaker", "Plate", "Pot", "Potato", "SaltShaker",
#                            "SoapBottle", "Spatula", "Spoon", "Tomato"]

PICKUP_ABLE_OBJECT = ["Apple", "Bowl", "Bread", "ButterKnife",
                       "Cup", "DishSponge", "Egg", "Fork", "Knife", "Lettuce", "Plate", "Potato"
                       , "Spatula", "Spoon", "Tomato"]

# PUT_ABLE_OBJECT = [
#     "Bowl", "Cabinet", "CounterTop", 
#       "Fridge", "GarbageCan", "Pan", "Plate",
#         "Pot", "Sink", "StoveBurner", "Toaster"
# ]
    
PUT_ABLE_OBJECT = [
      "Fridge", "Pan", "Plate",
        "Pot", "StoveBurner", "Toaster"
]

OPEN_ABLE_OBJECT = ["Cabinet", "Fridge"]

# pickupable object type to receptacles' type
PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE = {"Apple": ["Pan", "Fridge", "Plate"], "Bowl": ["Fridge"], "Bread": ["Fridge"], "ButterKnife": ["Pot", "Pan", "Plate"], "Cup": ["Fridge"], "DishSponge": ["Pot", "Pan", "Plate"], "Egg": ["Pot", "Pan", "Fridge", "Plate"], "Fork": ["Pot", "Pan", "Plate", "Fridge"], "Knife": ["Pot", "Pan", "Plate"], "Lettuce": ["Pot", "Pan", "Fridge", "Plate"], "Plate": ["Fridge"], "Potato": ["Pot", "Pan", "Fridge", "Plate"], "Spatula": ["Pot", "Pan", "Plate"], "Spoon": ["Pot", "Pan", "Plate"], "Tomato": ["Fridge", "Plate"]}

OBJECT_CAN_EAT = [
    "Potato", "Apple", "Bread", "Egg", "Lettuce", "Tomato", 
]

OBJECT_CAN_DRINK = [
    "Mug", "Cup"
]

tmp_dict = {}
for obj in PICKUP_ABLE_OBJECT:
    tmp_dict[obj] = []
    for receptacle in PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE[obj]:
        if receptacle in PUT_ABLE_OBJECT:
            tmp_dict[obj].append(receptacle)
    if len(tmp_dict[obj]) == 0:
        print(obj)
json.dump(tmp_dict, open("./tmp_dict.json", "w"))

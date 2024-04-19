from code import interact
from enum import Enum, unique
from typing import Set
from ai2thor.platform import CloudRendering
    
import os 
from pathlib import Path
import json 
import stringcase

from collections import OrderedDict
from typing import TypeVar, Generic, Tuple, Optional, Union, Dict, List, Any

import torch
from gym.spaces.dict import Dict as SpaceDict
import torch.nn as nn


# Built-in Type
@unique
class SceneType(Enum):
    KITCHENS = 0
    LIVINGROOMS = 1
    BEDROOMS = 2
    BATHROOMS = 3

@unique
class AgentType(Enum):
    AGENT_WITH_FULL_CAPABILITIES = 1
    AGENT_WITH_CARELESS_MIND = 3
    AGENT_WITH_PICKUP_ISSUES = 4
    AGENT_WITH_OPEN_ISSUES = 5
    AGENT_WITH_TOGGLE_ISSUES = 6
    AGENT_WITH_VISIBILITY_ISSUES = 7
    AGENT_WITH_MOBILITY_ISSUES = 8

@unique
class Temperature(Enum):
    Cold = 0
    RoomTemp = 1
    Hot = 2

temperature2index = {
    "Cold": 0,
    "RoomTemp": 1,
    "Hot": 2
}

@unique
class Mode(Enum):
    SYMBOLIC = 0
    VISION = 1

# Basic Settings
MAX_STEPS = 300
AGENT_TYPE_NUMBER = len(AgentType)  
# THOR_COMMIT_ID = 'eb93d0b6520e567bac8ad630462b5c0c4cea1f5f'
THOR_COMMIT_ID = 'f0825767cd50d69f666c7f282e54abfe58f1e917'
ROTATE_DEGREES = 90
CAMERA_ROTATE_DEGREES = 30
OBJECT_BASE_PROPERTIES_NUMBER = 6

# Hyper-parameters
MAX_OBJS_NUM_FROM_OBSERVATION = 100
SYMBOLIC_OBJECT_EMBED_LEN = 13
SYMBOLIC_AGENT_EMBED_LEN = 9

VISIBILITY_DISTANCE = 1.0
GRID_SIZE = 0.25
SCREEN_SIZE = 1200

TRAIN_PROCESS = 5
PROPERTIES_SELECTED = [True, True, True, True, True, True, False, True] # 1 main agent, 1 helper agent

# directory
STARTER_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data"
)

# subtasks
SUBTASKS_TYPE_2_ID = {
    'RearrangeObject': 0,
    'CleanCutlery':1 ,
    'SliceFruit': 2,
    'MakeCoffee': 3,
    'ToastBread': 4,
    'CookEgg': 5,
}
SUBTASKS_TYPE_NUM = len(SUBTASKS_TYPE_2_ID)
# goals
GOALS_TYPE_2_ID = {
    'Open': 0,
    'Close': 1,
    'PickUp': 2,
    'PutOn': 3,
    'PutIn': 4,
    'Slice': 5,
    'ToggleOn': 6,
    'ToggleOff': 7
}
GOALS_TYPE_NUM = len(GOALS_TYPE_2_ID)


DETAILED_GOALS_TYPE = [
    # clean cutlery
    'PickUp=Knife',
    'PickUp=Fork',
    'PutOn=Knife=SinkBasin',
    'PutOn=Fork=SinkBasin',
    'Open=Drawer',
    'Close=Drawer',
    'ToggleOn=Faucet',
    'ToggleOff=Faucet',
    # toast bread
    'Slice=Bread',
    'PickUp=BreadSliced',
    'PutOn=BreadSliced=Toaster',
    'ToggleOn=Toaster',
    'ToggleOff=Toaster',
    # Make coffee
    'PickUp=Mug', 
    'PutOn=Mug=CoffeeMachine', 
    'ToggleOn=CoffeeMachine', 
    'ToggleOff=CoffeeMachine', 
    'Open=Cabinet', 
    'Close=Cabinet',
    # cook
    'PickUp=Pan', 
    'PutOn=Pan=StoveBurner', 
    'PickUp=Egg', 
    'PutOn=Egg=Pan', 
    'Slice=Egg', 
    'ToggleOn=StoveKnob', 
    'ToggleOff=StoveKnob', 
    'Open=Cabinet', 
    'Close=Cabinet', 
    'Open=Fridge', 
    'Close=Fridge',
    # rearrange
    'PickUp=Potato', 
    'PutIn=Potato=Microwave', 
    'Close=Microwave', 
    'PickUp=Mug', 
    'PutOn=Mug=CoffeeMachine', 
    'Open=Microwave', 
    'Open=Cabinet', 
    'Close=Cabinet', 
    'Open=Fridge', 
    'Close=Fridge', 
    'Open=Drawer', 
    'Close=Drawer', 
    'Open=Safe', 
    'Close=Safe',
    # slice
    'PickUp=Apple', 
    'PutOn=Apple=CounterTop', 
    'Slice=Apple', 
    'PickUp=Lettuce', 
    'PutOn=Lettuce=CounterTop', 
    'Slice=Lettuce', 
    'Open=Fridge', 
    'Close=Fridge', 
    'PutOn=Lettuce=SideTable', 
    'PutOn=Apple=SideTable', 
    'PutOn=Lettuce=DiningTable', 
    'PutOn=Apple=DiningTable', 
    'Open=Cabinet', 
    'Close=Cabinet',
]

DETAILED_GOALS_TYPE_2_ID = {
    goal_type: id for id, goal_type in enumerate(DETAILED_GOALS_TYPE)
}

DETAILED_GOALS_TYPE_NUM = len(DETAILED_GOALS_TYPE)

# rotation and horizon information in AI2THOR
'''
Rotation information
rotation = 0 --- z > 0
rotation = 180 --- z < 0
rotation = 90 --- x > 0
rotatioin = 270 --- x < 0
RotateRight increases rotation while RotateLeft decreases rotation.

Horizon information
look up -> horizon decreases
look down -> horizon increases
'''

# objects with properites in kitchens
ALL_PICKUPABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'AluminumFoil', 'Apple', 'Book', 'Bottle', 'Bowl', 'Bread', 'ButterKnife', 
    'CellPhone', 'CreditCard', 'Cup', 'DishSponge', 'Egg', 'Fork', 'Kettle', 
    'Knife', 'Ladle', 'Lettuce', 'Mug', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil',
    'PepperShaker', 'Plate', 'Pot', 'Potato', 'SaltShaker', 'SoapBottle', 'Spatula', 
    'Spoon', 'SprayBottle', 'Statue', 'Tomato', 'Vase', 'WineBottle',

    'BreadSliced',  'EggCracked', 'AppleSliced',  'LettuceSliced', 'PotatoSliced', 'TomatoSliced' # sliced objects are added
]

USED_PICKUPABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'Potato', 'Mug',    # rearrange, coffee
    'Fork', 'Knife',    # clean
    'Apple', 'Lettuce', # slice
    'BreadSliced',      # toast
    'Egg', 'Pan'        # cook
]

ALL_RECEPTACLES_TYPE_IN_KITCHENS = [
    'Bowl', 'Cabinet', 'Chair', 'CoffeeMachine', 'CounterTop', 'Cup', 
    'DiningTable', 'Drawer', 'Floor', 'Fridge', 'GarbageCan', 'Microwave', 
    'Mug', 'Pan', 'Plate', 'Pot', 'Safe', 'Shelf', 'SideTable', 'Sink', 
    'SinkBasin', 'Stool', 'StoveBurner', 'Toaster'
]
USED_RECEPTACLES_TYPE_IN_KITCHENS = [
    'CoffeeMachine', 'Microwave',               # rearrange, coffee
    'SinkBasin',                                # clean
    'CounterTop', 'DiningTable', 'SideTable',   # slice
    'Toaster',                                  # toast
    'Pan', 'StoveBurner'                        # cook
]

ALL_OPENABLE_RECEPTACLES_TYPE_IN_KITCHENS = [
    'Blinds', 'Book', 'Cabinet', 'Drawer', 'Fridge', 'Kettle', 'Microwave', 'Safe'
]
USED_OPENABLE_RECEPTACLES_TYPE_IN_KITCHENS = [
    'Cabinet', 'Drawer', 'Fridge', 'Microwave', 'Safe'
]

ALL_SLICEABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'Apple', 'Bread', 'Egg', 'Lettuce', 'Potato', 'Tomato'
]
USED_SLICEABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'Apple', 'Lettuce',     # slice
    'Bread',                # toast
    'Egg',                  # cook
]

ALL_TOGGLEABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'CellPhone', 'CoffeeMachine', 'Faucet', 'LightSwitch', 'Microwave', 'StoveKnob', 'Toaster'
]
USED_TOGGLEABLE_OBJECTS_TYPE_IN_KITCHENS = [
    'Faucet',           # clean
    'CoffeeMachine',    # coffee
    'Toaster',          # toast
    'StoveKnob',        # cook
]

ALL_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS = {
    'pickupable': ALL_PICKUPABLE_OBJECTS_TYPE_IN_KITCHENS,
    'receptacle': ALL_RECEPTACLES_TYPE_IN_KITCHENS,
    'openable': ALL_OPENABLE_RECEPTACLES_TYPE_IN_KITCHENS,
    'sliceable': ALL_SLICEABLE_OBJECTS_TYPE_IN_KITCHENS,
    'toggleable': ALL_TOGGLEABLE_OBJECTS_TYPE_IN_KITCHENS
}

USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS = {
    'pickupable': USED_PICKUPABLE_OBJECTS_TYPE_IN_KITCHENS,
    'receptacle': USED_RECEPTACLES_TYPE_IN_KITCHENS,
    'openable': USED_OPENABLE_RECEPTACLES_TYPE_IN_KITCHENS,
    'sliceable': USED_SLICEABLE_OBJECTS_TYPE_IN_KITCHENS,
    'toggleable': USED_TOGGLEABLE_OBJECTS_TYPE_IN_KITCHENS
}

USED_OBJECTS_TYPE_IN_KITCHENS = set(
    [
        *USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['pickupable'],
        *USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['receptacle'],
        *USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['openable'],
    ]
)

ALL_OBJECTS_TYPE_WITH_NONE_IN_KITCHENS = [
    'None', 'AluminumFoil', 'Apple', 'Blinds', 'Book', 'Bottle', 'Bowl', 'Bread', 
    'ButterKnife', 'Cabinet', 'CellPhone', 'Chair', 'CoffeeMachine', 'CounterTop', 
    'CreditCard', 'Cup', 'Curtains', 'DiningTable', 'DishSponge', 'Drawer', 'Egg', 
    'Faucet', 'Floor', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HousePlant', 
    'Kettle', 'Knife', 'Ladle', 'Lettuce', 'LightSwitch', 'Microwave', 'Mirror', 
    'Mug', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Plate', 'Pot', 
    'Potato', 'Safe', 'SaltShaker', 'Shelf', 'ShelvingUnit', 'SideTable', 'Sink', 
    'SinkBasin', 'SoapBottle', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool', 
    'StoveBurner', 'StoveKnob', 'Toaster', 'Tomato', 'Vase', 'Window', 'WineBottle',
    'BreadSliced',  'EggCracked', 'AppleSliced',  'LettuceSliced', 'PotatoSliced', 'TomatoSliced' # sliced objects are added
]

ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN = len(ALL_OBJECTS_TYPE_WITH_NONE_IN_KITCHENS)

ALL_OBJECTS_TYPE_2_ID_IN_KITCHENS = {
    obj_type: id for id, obj_type in enumerate(ALL_OBJECTS_TYPE_WITH_NONE_IN_KITCHENS)
}
# ALL_OBJECTS_TYPE_2_ID_IN_KITCHENS = {
    # "None": 0,
    # "AluminumFoil": 1,
    # "Apple": 2,
    # "Blinds": 3,
    # "Book": 4,
    # "Bottle": 5,
    # "Bowl": 6,
    # "Bread": 7,
    # "ButterKnife": 8,
    # "Cabinet": 9,
    # "CellPhone": 10,
    # "Chair": 11,
    # "CoffeeMachine": 12,
    # "CounterTop": 13,
    # "CreditCard": 14,
    # "Cup": 15,
    # "Curtains": 16,
    # "DiningTable": 17,
    # "DishSponge": 18,
    # "Drawer": 19,
    # "Egg": 20,
    # "Faucet": 21,
    # "Floor": 22,
    # "Fork": 23,
    # "Fridge": 24,
    # "GarbageBag": 25,
    # "GarbageCan": 26,
    # "HousePlant": 27,
    # "Kettle": 28,
    # "Knife": 29,
    # "Ladle": 30,
    # "Lettuce": 31,
    # "LightSwitch": 32,
    # "Microwave": 33,
    # "Mirror": 34,
    # "Mug": 35,
    # "Pan": 36,
    # "PaperTowelRoll": 37,
    # "Pen": 38,
    # "Pencil": 39,
    # "PepperShaker": 40,
    # "Plate": 41,
    # "Pot": 42,
    # "Potato": 43,
    # "Safe": 44,
    # "SaltShaker": 45,
    # "Shelf": 46,
    # "ShelvingUnit": 47,
    # "SideTable": 48,
    # "Sink": 49,
    # "SinkBasin": 50,
    # "SoapBottle": 51,
    # "Spatula": 52,
    # "Spoon": 53,
    # "SprayBottle": 54,
    # "Statue": 55,
    # "Stool": 56,
    # "StoveBurner": 57,
    # "StoveKnob": 58,
    # "Toaster": 59,
    # "Tomato": 60,
    # "Vase": 61,
    # "Window": 62,
    # "WineBottle": 63,
    # "BreadSliced": 64,
    # "EggCracked": 65,
    # "AppleSliced": 66,
    # "LettuceSliced": 67,
    # "PotatoSliced": 68,
    # "TomatoSliced": 69
# }



# objects with properites in living rooms
PICKUPABLE_OBJECTS_TYPE_IN_LIVINGROOMS = [
    'KeyChain', 'Pencil', 'WateringCan', 'Laptop', 'TissueBox', 'Watch', 'Vase', 'Pen', 
    'RemoteControl', 'Newspaper', 'CreditCard', 'Candle', 'Book', 'CellPhone', 'Boots', 
    'Statue', 'Pillow'
]
RECEPTACLES_TYPE_IN_LIVINGROOMS = [
    'SideTable', 'Floor', 'ArmChair', 'Desk', 'Plate', 'Drawer', 'TVStand', 'DogBed', 'Sofa', 
    'Safe', 'Stool', 'Dresser', 'Shelf', 'GarbageCan', 'Box', 'CoffeeTable', 'Bowl', 
    'DiningTable', 'Chair', 'Ottoman', 'Cabinet'
]
OPENABLE_RECEPTACLES_TYPE_IN_LIVINGROOMS = ['Safe', 'Box', 'Cabinet', 'Drawer']

# objects with properites in bedrooms
PICKUPABLE_OBJECTS_TYPE_IN_BEDROOMS = [
    'TissueBox', 'Vase', 'CreditCard', 'CellPhone', 'Pillow', 'Pencil', 'Watch', 'Pen', 
    'AlarmClock', 'RemoteControl', 'TableTopDecor', 'Boots', 'Statue', 'BasketBall', 'CD', 
    'Dumbbell', 'KeyChain', 'Laptop', 'Cloth', 'BaseballBat', 'TennisRacket', 'Book', 'TeddyBear'
]
RECEPTACLES_TYPE_IN_BEDROOMS = [
    'SideTable', 'Floor', 'Footstool', 'ArmChair', 'Desk', 'LaundryHamper', 'Drawer', 'DogBed', 
    'Sofa', 'ShelvingUnit', 'Safe', 'Bed', 'Stool', 'Dresser', 'Shelf', 'GarbageCan', 'Mug', 
    'Box', 'CoffeeTable', 'Bowl', 'Chair', 'CounterTop', 'Cabinet'
]
OPENABLE_RECEPTACLES_TYPE_IN_BEDROOMS = ['Safe', 'Drawer', 'LaundryHamper', 'Box', 'Cabinet']

# objects with properites in bathrooms
PICKUPABLE_OBJECTS_TYPE_IN_BATHROOMS = [
    'TissueBox', 'Cloth', 'PaperTowelRoll', 'SoapBar', 'Towel', 'DishSponge', 'Candle', 
    'SoapBottle', 'Plunger', 'SprayBottle', 'ScrubBrush', 'HandTowel', 'ToiletPaper'
]
RECEPTACLES_TYPE_IN_BATHROOMS = [
    'HandTowelHolder', 'SideTable', 'Floor', 'Footstool', 'Toilet', 'Dresser', 'Bathtub', 'Sink', 
    'Drawer', 'TowelHolder', 'ToiletPaperHanger', 'Shelf', 'GarbageCan', 'CounterTop', 'Cabinet'
]
OPENABLE_RECEPTACLES_TYPE_IN_BATHROOMS = ['Drawer', 'Toilet', 'Cabinet']



# all actions' names
ALL_ACTIONS_NAME = {
    'navigation_actions': ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown", "Stand", "Crouch"],   
    'interaction_actions': ["PickUp", "Put", "Open", "Close", "Slice", "ToggleOn", "ToggleOff"],  
    'function_actions': ["Wait", "Done"],     
}

# Action SPACE
PICKUP_ACTIONS = list(
    sorted(
        [
            f'PickUp={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['pickupable']
        ]
    )
)
PUT_ACTIONS = list(
    sorted(
        [
            f'Put={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['receptacle']
        ]
    )
)
OPEN_ACTIONS = list(
    sorted(
        [
            f'Open={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['openable']
        ]
    )
)
CLOSE_ACTIONS = list(
    sorted(
        [
            f'Close={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['openable']
        ]
    )
)
SLICE_ACTIONS = list(
    sorted(
        [
            f'Slice={stringcase.capitalcase(obj_type)}' 
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['sliceable']
        ]
    )
)
TOGGLEON_ACTIONS = list(
    sorted(
        [
            f'ToggleOn={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['toggleable']
        ]
    )
)
TOGGLEOFF_ACTIONS = list(
    sorted(
        [
            f'ToggleOff={stringcase.capitalcase(obj_type)}'
            for obj_type in USED_OBJECTS_TYPE_WITH_PROPERTIES_IN_KITCHENS['toggleable']
        ]
    )
)

USED_ACTION_SPACE = [
    'Wait', 
    *ALL_ACTIONS_NAME['navigation_actions'],
    *PICKUP_ACTIONS,
    *PUT_ACTIONS,
    *OPEN_ACTIONS,
    *CLOSE_ACTIONS,
    *SLICE_ACTIONS,
    *TOGGLEON_ACTIONS,
    *TOGGLEOFF_ACTIONS,
    # 'Done',
]
    
USED_ACTION_SPACE_LEN = len(USED_ACTION_SPACE)

USED_ACTION_NAME_2_ID = {
    action_name: id for id, action_name in enumerate(USED_ACTION_SPACE)
}
# USED_ACTION_NAME_2_ID = {
    # "Wait": 0,
    # "MoveAhead": 1,
    # "RotateRight": 2,
    # "RotateLeft": 3,
    # "LookUp": 4,
    # "LookDown": 5,
    # "Stand": 6,
    # "Crouch": 7,
    # "PickUp=Apple": 8,
    # "PickUp=BreadSliced": 9,
    # "PickUp=Egg": 10,
    # "PickUp=Fork": 11,
    # "PickUp=Knife": 12,
    # "PickUp=Lettuce": 13,
    # "PickUp=Mug": 14,
    # "PickUp=Pan": 15,
    # "PickUp=Potato": 16,
    # "Put=CoffeeMachine": 17,
    # "Put=CounterTop": 18,
    # "Put=DiningTable": 19,
    # "Put=Microwave": 20,
    # "Put=Pan": 21,
    # "Put=SideTable": 22,
    # "Put=SinkBasin": 23,
    # "Put=StoveBurner": 24,
    # "Put=Toaster": 25,
    # "Open=Cabinet": 26,
    # "Open=Drawer": 27,
    # "Open=Fridge": 28,
    # "Open=Microwave": 29,
    # "Close=Cabinet": 30,
    # "Close=Drawer": 31,
    # "Close=Fridge": 32,
    # "Close=Microwave": 33,
    # "Slice=Apple": 34,
    # "Slice=Bread": 35,
    # "Slice=Egg": 36,
    # "Slice=Lettuce": 37,
    # "ToggleOn=CoffeeMachine": 38,
    # "ToggleOn=Faucet": 39,
    # "ToggleOn=StoveKnob": 40,
    # "ToggleOn=Toaster": 41,
    # "ToggleOff=CoffeeMachine": 42,
    # "ToggleOff=Faucet": 43,
    # "ToggleOff=StoveKnob": 44,
    # "ToggleOff=Toaster": 45
# }


# pickupable object type to receptacles' type
PICKUPABLE_OBJECTS_TYPE_AND_COMPATIBLE_RECEPTACLES_TYPE = {
    'AlarmClock': [
        'Box', 'Dresser', 'Desk', 'SideTable', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'CounterTop', 'Shelf', 
    ],
    'AluminumFoil': [
        'Box', 'Desk', 'SideTable', 'DiningTable', 'TVStand', 'CoffeeTable', 'CounterTop', 
        'Shelf', 
        ], 
    'Apple': [
        'Pot', 'Pan', 'Bowl', 'Microwave', 'Fridge', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'Desk', 'CounterTop', 'GarbageCan', 'Dresser'
    ], 
    'BaseballBat': [
        'Bed', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'Desk', 'CounterTop'
    ], 
    'BasketBall': [
        'Sofa', 'ArmChair', 'Dresser', 'Desk', 'Bed', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop'
    ], 
    'Book': [
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'Desk', 'Bed', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer'
    ], 
    'Boots': [ 'Floor' ], 
    'Bottle': [
        'Fridge', 'Box', 'Dresser', 'Desk', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'GarbageCan'
    ], 
    'Bowl': [
        'Microwave', 'Fridge', 'Dresser', 'Desk', 'Sink', 'SinkBasin', 'Cabinet', 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf'
    ],
    'Box': [
        'Sofa', 'ArmChair', 'Dresser', 'Desk', 'Cabinet', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Ottoman'
    ],
    'Bread': [
        'Microwave', 'Fridge', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'Desk', 
        'CounterTop', 'GarbageCan', 'Plate'
    ], 
    'ButterKnife': [
        'Pot', 'Pan', 'Bowl', 'Mug', 'Plate', 'Cup', 'Sink', 'SinkBasin', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'Desk', 'CounterTop', 'Drawer'
    ], 
    'CD': [
        'Box', 'Ottoman', 'Dresser', 'Desk', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan', 'Safe', 'Sofa', 'ArmChair'	
    ], 
    'Candle': [
        'Box', 'Dresser', 'Desk', 'Toilet', 'Cart', 'Bathtub', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer'
    ], 
    'CellPhone': [
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'Desk', 'Bed', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'Safe'
    ], 
    'Cloth': [
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'LaundryHamper', 'Desk', 'Toilet', 
        'Cart', 'BathtubBasin', 'Bathtub', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'CreditCard': [
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'Shelf'
    ], 
    'Cup': [ 
        'Microwave', 'Fridge', 'Dresser', 'Desk', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf'
    ],
    'DishSponge': [ 
        'Pot', 'Pan', 'Bowl', 'Plate', 'Box', 'Toilet', 'Cart', 'Cart', 'BathtubBasin', 
        'Bathtub', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'Dumbbell': [
        'Desk', 'SideTable', 'DiningTable', 'CounterTop', 'Floor', 'Shelf', 
    ], 
    'Egg': [ 
        'Pot', 'Pan', 'Bowl', 'Microwave', 'Fridge', 'Plate', 'Sink', 'SinkBasin', 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'GarbageCan'
    ], 
    'Fork': [ 
        'Pot', 'Pan', 'Bowl', 'Mug', 'Plate', 'Cup', 'Sink', 'SinkBasin', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer', 'Fridge'
    ], 
    'Footstool': [ 'Floor' ],
    'HandTowel': [ 'HandTowelHolder' ], 
    'Kettle': [ 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Sink', 
        'SinkBasin', 'Cabinet', 'StoveBurner', 'Shelf'
    ], 
    'KeyChain': [ 
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'Safe'
    ], 
    'Knife': [ 
        'Pot', 'Pan', 'Bowl', 'Mug', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer'
    ], 
    'Ladle': [ 
        'Pot', 'Pan', 'Bowl', 'Plate', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer'
    ], 
    'Laptop': [ 
        'Sofa', 'ArmChair', 'Ottoman', 'Dresser', 'Desk', 'Bed', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop'
    ], 
    'Lettuce': [ 
        'Pot', 'Pan', 'Bowl', 'Fridge', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'GarbageCan'
    ], 
    'Mug': [
        'SinkBasin', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf'
    ],
    'Newspaper': [ 
        'Sofa', 'ArmChair', 'Ottoman', 'Dresser', 'Desk', 'Bed', 'Toilet', 'Cabinet', 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 
        'GarbageCan'
    ], 
    'PaperTowelRoll': [
        'Sofa', 'ArmChair', 'Dresser', 'Desk', 'Bed', 'Toilet', 'Cabinet', 'DiningTable', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'Pan': [ 
        'DiningTable', 'CounterTop', 'TVStand', 'CoffeeTable', 'SideTable', 'Sink', 'SinkBasin', 
        'Cabinet', 'StoveBurner', 'Fridge'
    ],
    'Pen': [ 
        'Mug', 'Box', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'Pencil': [ 
        'Mug', 'Box', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'PepperShaker': [ 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer', 
        'Cabinet', 'Shelf'
    ], 
    'Pillow': [ 'Sofa', 'ArmChair', 'Ottoman', 'Bed' ],
    'Plate': [ 
        'Microwave', 'Fridge', 'Dresser', 'Desk', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf'
    ] ,
    'Plunger': [ 'Cart', 'Cabinet' ],
    'Pot': [ 
        'StoveBurner', 'Fridge', 'Sink', 'SinkBasin', 'Cabinet', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf'
    ],
    'Potato': [ 
        'Pot', 'Pan', 'Bowl', 'Microwave', 'Fridge', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'GarbageCan'
    ], 
    'RemoteControl': [ 
        'Sofa', 'ArmChair', 'Box', 'Ottoman', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer'
    ], 
    'SaltShaker': [ 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer', 
        'Cabinet', 'Shelf'
    ], 
    'ScrubBrush': [
        'Dresser', 'Desk', 'Toilet', 'Cart', 'Bathtub', 'Sink', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'SoapBar': [ 
        'Toilet', 'Cart', 'Bathtub', 'BathtubBasin', 'Sink', 'SinkBasin', 'Cabinet', 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 
        'Drawer', 'GarbageCan'
    ], 
    'SoapBottle': [ 
        'Dresser', 'Desk', 'Toilet', 'Cart', 'Bathtub', 'Sink', 'Cabinet', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'Spatula': [ 
        'Pot', 'Pan', 'Bowl', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer'
    ], 
    'Spoon': [ 
        'Pot', 'Pan', 'Bowl', 'Mug', 'Plate', 'Cup', 'Sink', 'SinkBasin', 'DiningTable', 
        'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Drawer'
    ], 
    'SprayBottle': [ 
        'Dresser', 'Desk', 'Toilet', 'Cart', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'Statue': [ 
        'Box', 'Dresser', 'Desk', 'Cart', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf', 'Safe'
    ], 
    'TableTopDecor': [
        'Bed', 'Sofa', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 
    ], 
    'TeddyBear': [ 
        'Bed', 'Sofa', 'ArmChair', 'Ottoman', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Safe'
    ], 
    'TennisRacket': [ 
        'Dresser', 'Desk', 'Bed', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop'
    ], 
    'TissueBox': [ 
        'Box', 'Dresser', 'Desk', 'Toilet', 'Cart', 'Cabinet', 'DiningTable', 'TVStand', 
        'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan'
    ], 
    'ToiletPaper': [ 
        'Dresser', 'Desk', 'Toilet', 'ToiletPaperHanger', 'Cart', 'Bathtub', 'Cabinet', 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 
        'Drawer', 'GarbageCan'
    ], 
    'Tomato': [ 
        'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Sink', 
        'SinkBasin', 'Pot', 'Bowl', 'Fridge', 'GarbageCan', 'Plate'
    ], 
    'Towel': [ 'TowelHolder' ], 
    'Vase': [ 
        'Box', 'Dresser', 'Desk', 'Cart', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 'Shelf', 'Safe'
    ], 
    'Watch': [ 
        'Box', 'Dresser', 'Desk', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf', 'Drawer', 'Safe'
    ], 
    'WateringCan': [ 
        'Dresser', 'Desk', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 
        'CounterTop', 'Shelf', 'Drawer'
    ], 
    'WineBottle': [ 
        'Fridge', 'Dresser', 'Desk', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 
        'SideTable', 'CounterTop', 'Shelf', 'GarbageCan'
    ]
}

# all receptacles' type
ALL_RECEPTACLES_TYPE = [
    'ArmChair', 'Bathtub', 'Bed', 'Bowl', 'Box', 'Cabinet', 'Chair', 'CoffeeMachine', 
    'CoffeeTable', 'CounterTop', 'Cup', 'Desk', 'DiningTable', 'DogBed', 'Drawer', 
    'Dresser', 'Floor', 'Footstool', 'Fridge', 'GarbageCan', 'HandTowelHolder', 
    'LaundryHamper', 'Microwave', 'Mug', 'Ottoman', 'Pan', 'Plate', 'Pot', 'Safe', 
    'Shelf', 'ShelvingUnit', 'SideTable', 'Sink', 'SinkBasin', 'Sofa', 'Stool', 
    'StoveBurner', 'TVStand', 'Toaster', 'Toilet', 'ToiletPaperHanger', 'TowelHolder'
]

# receptacles' type with large surface
WITH_LARGE_SURFACE_RECEPTACLES_TYPE = [
    'CoffeeTable', 'CounterTop', 'Desk', 'DiningTable', 'SideTable', 
]

# these receptacles are not used
EXCLUDED_RECEPTACLES_TYPE_IN_DATASET = [
    'Bowl', 'Box', 'Cup', 'Floor', 'Footstool', 'Mug', 'Pot'
]

# openable receptacles' type
OPENABLE_RECEPTACLES_TYPE = [
    'Box', 'Cabinet', 'Drawer', 'Fridge', 'LaundryHamper', 'Microwave', 'Safe', 'Toilet'
]

# sliceable objects' type
SLICEABLE_FRUIT_OBJECTS_TYPE = [
    'Apple', 'Lettuce', 'Potato', 'Tomato', 
]

# these objects are used in cleaning tasks
# Note tha 'Fork', 'Knife', 'Spatula', 'ButterKnife' don't have isDirty property.
CLEANABLE_OBJECTS_TYPE = {
    'kitchen': [ 'Bowl', 'Cup', 'Fork', 'Knife' 'Mug', 'Pan', 'Plate', 'Pot' , 'Spatula', 'ButterKnife'],        
    'bathroom': [ 'DishSponge', 'HandTowel', 'ScrubBrush', 'Towel' ],  
}

COOK_TASKS = {
    'Bread': ['Toaster'],
    'Egg': ['Microwave', 'StoveBurner'],
    'Potato': ['Microwave', 'StoveBurner']
}

COOK_RECEPTACLES = [
    ('Plate', 'Microwave'), ('Pan', 'StoveBurner')
]



'''Functions'''

def test_rotation(controller) -> None:
    # rotation = 0 --- z > 0
    # rotation = 180 --- z < 0
    # rotation = 90 --- x > 0
    # rotatioin = 270 --- x < 0
    # RotateRight increases rotation while RotateLeft decreases rotation.
    controller.step(
        action='Teleport',
        position=dict(x=1.0, y=0.9, z=-0.25),
        rotation=dict(x=0, y=0, z=0),
    )
    print('----------------------------------------------------')
    for id in list(controller.last_event.instance_masks.keys()):
        print(id)
    controller.step(
        action='Teleport',
        position=dict(x=1.25, y=0.9, z=0),
        rotation=dict(x=0, y=180, z=0),
    )
    print('----------------------------------------------------')
    for id in list(controller.last_event.instance_masks.keys()):
        print(id)
    controller.step(
        action='Teleport',
        position=dict(x=1.0, y=0.9, z=-0.25),
        rotation=dict(x=0, y=90, z=0),
    )
    print('----------------------------------------------------')
    for id in list(controller.last_event.instance_masks.keys()):
        print(id)
    controller.step(
        action='Teleport',
        position=dict(x=1.25, y=0.9, z=0),
        rotation=dict(x=0, y=270, z=0),
    )
    print('----------------------------------------------------')
    for id in list(controller.last_event.instance_masks.keys()):
        print(id)

    print(controller.last_event.metadata['agent']['rotation']['y'])
    controller.step(
        action='RotateRight'
    )
    print(controller.last_event.metadata['agent']['rotation']['y'])        

def scenes_2_gain(scenes: SceneType) -> int:
    if scenes == SceneType.KITCHENS:
        return 0
    elif scenes == SceneType.LIVINGROOMS:
        return 200
    elif scenes == SceneType.BEDROOMS:
        return 300
    elif scenes == SceneType.BATHROOMS:
        return 400
    else:
        return None

def search_all_receptacles(controller, scenes: SceneType) -> Set[str]:
    receptacles_all = []
    gain = scenes_2_gain(scenes)
    for i in range(gain + 1, gain + 31):
        controller.reset(scene=f'FloorPlan{i}')
        receptacles = [
            obj['objectId'].split('|')[0] for obj in 
            controller.last_event.metadata['objects'] if obj['receptacle'] == True
        ]
        receptacles_all.extend(receptacles)
    receptacles_all = set(receptacles_all)
    return receptacles_all

def search_all_openable_receptacles(controller, scenes: SceneType) -> Set[str]:
    openable_receptacles_all = []
    gain = scenes_2_gain(scenes)
    for i in range(gain + 1, gain + 31):
        controller.reset(scene=f'FloorPlan{i}')
        openable_receptacles = [
            obj['objectId'].split('|')[0] for obj in controller.last_event.metadata['objects'] 
            if obj['receptacle'] == True and obj['openable'] == True
        ]
        openable_receptacles_all.extend(openable_receptacles)
    openable_receptacles_all = set(openable_receptacles_all)
    return openable_receptacles_all

def search_all_cookable_objects(controller, scenes: SceneType) -> Set[str]:
    pickupable_objects_all = []
    gain = scenes_2_gain(scenes)
    for i in range(gain + 1, gain + 31):
        controller.reset(scene=f'FloorPlan{i}')
        pickupable_objects = [
            obj['objectId'].split('|')[0] for obj in controller.last_event.metadata['objects']
            if obj['cookable'] == True and obj['receptacle'] == False
        ]
        pickupable_objects_all.extend(pickupable_objects)
    pickupable_objects_all = set(pickupable_objects_all)
    return pickupable_objects_all

def search_all_toggleable_objects(controller, scenes: SceneType) -> Set[str]:
    toggleable_objects_all = []
    gain = scenes_2_gain(scenes)
    for i in range(gain + 1, gain + 31):
        controller.reset(scene=f'FloorPlan{i}')
        toggleable_objects = [
            obj['objectId'].split('|')[0] for obj in controller.last_event.metadata['objects']
            if obj['toggleable'] == True 
        ]
        toggleable_objects_all.extend(toggleable_objects)
    toggleable_objects_all = set(toggleable_objects_all)
    return toggleable_objects_all

def get_obj_types_in_kitchen(controller):
    all_obj_types_in_kitchen = []
    for scene_id in range(1, 31):
        scene_name = f'FloorPlan{scene_id}'
        controller.reset(scene=scene_name)
        for obj in controller.last_event.metadata['objects']:
            if obj['objectType'] not in all_obj_types_in_kitchen:
                if obj['toggleable'] == True:
                    all_obj_types_in_kitchen.append(obj['objectType'])

    print(sorted(all_obj_types_in_kitchen))



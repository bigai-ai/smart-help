import ai2thor
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Sequence,
)
import networkx as nx
from torch.distributions.utils import lazy_property
import math

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.join(os.path.abspath('.'), '../../'))

from env.utils import (
    distance_3D, 
    find_closest_position, 
    compute_face_2_pos_rough_plus,
    obj_id_2_obj_type,
    find_parent_receptacle_plus
)

from env.constants import (
    AgentType,
    GRID_SIZE,
    USED_ACTION_NAME_2_ID,
    OPENABLE_RECEPTACLES_TYPE,
    STARTER_DATA_DIR
)
import matplotlib.pyplot as plt
AgentLocKeyType = Tuple[float, float, int, int]
AgentPoseKeyType = Tuple[float, float, int, int, bool]


class NavigatorTHOR:
    """Tracks shortest paths in AI2-THOR environments.

    Assumes 90 degree rotations and fixed step sizes.

    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        controller: ai2thor.controller.Controller,
        grid_size: float,
        include_move_left_right: bool = False,
        agent_id: int = 0,
        agent_num = 1,
    ):
        """Create a `ShortestPathNavigatorTHOR` instance.

        # Parameters
        controller : An AI2-THOR controller which represents the environment in which shortest paths should be
            computed.
        grid_size : The distance traveled by an AI2-THOR agent when taking a single navigational step.
        include_move_left_right : If `True` the navigational actions will include `MoveLeft` and `MoveRight`, otherwise
            they wil not.
        """
        self._cached_graphs: Dict[str, nx.DiGraph] = {}

        self._current_scene: Optional[nx.DiGraph] = None
        self._current_graph: Optional[nx.DiGraph] = None
        self.agent_id = agent_id
        self.agent_num = agent_num
        self._grid_size = grid_size
        self.controller = controller

        self._include_move_left_right = include_move_left_right

        self.except_pos = []

    @lazy_property
    def nav_actions_set(self) -> frozenset:
        """Navigation actions considered when computing shortest paths."""
        nav_actions = [
            "LookUp",
            "LookDown",
            "RotateLeft",
            "RotateRight",
            "MoveAhead",
        ]
        if self._include_move_left_right:
            nav_actions.extend(["MoveLeft", "MoveRight"])
        return frozenset(nav_actions)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    def on_reset(self):
        """Function that must be called whenever the AI2-THOR controller is
        reset."""
        self._current_scene = None
    
    @property
    def graph(self) -> nx.DiGraph:
        """A directed graph representing the navigation graph of the current
        scene."""
        
        g = nx.DiGraph()
        
        points = self.reachable_points_with_rotations_and_horizons()
        for p in points:
            self._add_node_to_graph(g, self.get_key(p))

        self._cached_graphs[self.scene_name] = g

        self._current_scene = self.scene_name
        self._current_graph = self._cached_graphs[self.scene_name].copy()
        return self._current_graph

    @property
    def reachable_positions(self, scene_name=None, boundary=None) -> List:
        '''
        return current reachable positions 
        '''
        event = self.controller.step(action="GetReachablePositions", agentId=self.agent_id)
        if not self.last_action_success:
            print("for debug navigator", event.metadata["errorMessage"])
        assert self.last_action_success

        return self.last_event.metadata["actionReturn"]

    def reachable_points_with_rotations_and_horizons(
        self,
    ) -> List[Dict[str, Union[float, int]]]:
        """Get all the reaachable positions in the scene along with possible
        rotation/horizons."""
        # self.controller.step(action="GetReachablePositions")
        # assert self.last_action_success

        # points_slim = self.last_event.metadata["actionReturn"]
        points_slim = self.reachable_positions

        for i in range(self.agent_num):
            if i == self.agent_id:
                continue
            else:
                agent_position = self.controller.last_event.events[i].metadata['agent']['position']

                # print("for debug navigator", agent_position)

                for position_dict in points_slim:
                    if math.sqrt((position_dict['x'] - agent_position['x']) ** 2 + (position_dict['z'] - agent_position['z']) ** 2) < 0.4:
                        points_slim.remove(position_dict)

        return points_slim

    @staticmethod
    def get_key(input_dict: Dict[str, Any], ndigits: int = 2) -> AgentLocKeyType:
        """Return a graph node key given an input agent location dictionary."""
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            # rot = input_dict["rotation"]
            # hor = input_dict["horizon"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            # rot = input_dict["rotation"]["y"]
            # hor = input_dict["cameraHorizon"]

        return (
            round(x, ndigits),
            round(z, ndigits),
            # round_to_factor(rot, 90) % 360,
            # round_to_factor(hor, 30) % 360,
        )

    def _add_from_to_edge(
        self, g: nx.DiGraph, s: AgentLocKeyType, t: AgentLocKeyType,
    ):
        """Add an edge to the graph."""

        g.add_edge(s, t, action='action')

    @lazy_property
    def possible_neighbor_offsets(self) -> Tuple[AgentLocKeyType, ...]:
        """Offsets used to generate potential neighbors of a node."""
        grid_size = round(self._grid_size, 2)
        offsets = []
        for x_diff in [-grid_size, 0, grid_size]:
            for z_diff in [-grid_size, 0, grid_size]:
                if (x_diff != 0) + (z_diff != 0) == 1:
                    offsets.append((x_diff, z_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: AgentLocKeyType):
        """Add a node to the graph along with any adjacent edges."""
        if s in graph:
            return
        if s in self.except_pos:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for x_diff, z_diff in self.possible_neighbor_offsets:
            t = (
                s[0] + x_diff,
                s[1] + z_diff,
                # (s[2] + rot_diff) % 360,
                # (s[3] + horz_diff) % 360,
            )
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    def _check_contains_key(self, key: AgentLocKeyType, add_if_not=True) -> bool:
        """Check if a node key is in the graph.

        # Parameters
        key : The key to check.
        add_if_not : If the key doesn't exist and this is `True`, the key will be added along with
            edges to any adjacent nodes.
        """
        key_in_graph = key in self.graph
        if not key_in_graph:
            print(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)
                if key not in self._cached_graphs[self.scene_name]:
                    self._add_node_to_graph(self._cached_graphs[self.scene_name], key)
        return key_in_graph

    def shortest_state_path(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ) -> Optional[Sequence[AgentLocKeyType]]:
        """Get the shortest path between node keys."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        path = nx.shortest_path(
            G=self.graph, source=source_state_key, target=goal_state_key
        )
        # necessary operation: change horizon from -30 to 330
        path_appropriate = []
        for pos in path:
            path_appropriate.append(
                (round(pos[0], 2), round(pos[1], 2), 
                round(pos[2], 0) % 360, round(pos[3], 0) % 360)
            )
        return path_appropriate

    def shortest_path(
        self, source_pose_key: AgentPoseKeyType, goal_pose_key: AgentPoseKeyType, except_pos
    ) -> Optional[Sequence[AgentPoseKeyType]]:
        self.except_pos = except_pos
        source_state_key = (
            source_pose_key[0], source_pose_key[1],
        )
        goal_state_key = (
            goal_pose_key[0], goal_pose_key[1],
        )
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return abs(x1 - x2) + abs(y1 - y2)

        # print("for debug navigator", (0.5, -0.75) in self.graph)
        # print("for debug navigator", (-0.25, -0.75) in self.graph)

        # G = self.graph

        # fig, ax_ = plt.subplots(1, 1)

        # # Plot the graph
        # nx.draw(G, with_labels=False, node_color='lightblue', font_weight='bold', ax=ax_)

        # # Save the plot to a file
        # plt.savefig("graph.png", format="PNG")
        # print("Astar graph", G)
        # print("for debug", self.last_event.metadata["actionReturn"])

        # print("for debug", source_state_key in self.graph, goal_state_key in self.graph)

        if (source_state_key not in self.graph) or (goal_state_key not in self.graph):
            return None

        try:

            path = nx.astar_path(
                G=self.graph, source=source_state_key, target=goal_state_key, heuristic=dist
            )

        except:
            return None
        # necessary operation: change horizon from -30 to 330
        shortest_path = []
        for pos in path:
            shortest_path.append(
                (round(pos[0], 2), round(pos[1], 2),)
            )
        
        return shortest_path
        
    def action_transitioning_between_keys(self, s: AgentLocKeyType, t: AgentLocKeyType):
        """Get the action that takes the agent from node s to node t."""
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next node key on the shortest path from the source to the
        goal."""
        if source_state_key == goal_state_key:
            raise RuntimeError("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next action along the shortest path from the source to the
        goal."""
        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_pose_path_next_action(
        self, source_pose_key: AgentPoseKeyType, goal_pose_key: AgentPoseKeyType
    ):
        source_state_key = (
            source_pose_key[0], source_pose_key[1],
            # source_pose_key[2], source_pose_key[3]
        )
        goal_state_key = (
            goal_pose_key[0], goal_pose_key[1],
            # goal_pose_key[2], goal_pose_key[3]
        )
        if source_pose_key[4] == goal_pose_key[4]:
            next_action = self.action_transitioning_between_keys(source_state_key, goal_state_key)
        elif source_pose_key[4] == True:    # goal_pose_key[4] == False
            next_action = 'Crouch'
        else:   # source_pose_key[4] == False and goal_pose_key[4] == True
            next_action = 'Stand'
        return next_action
            
    def shortest_path_next_action_multi_target(
        self,
        source_state_key: AgentLocKeyType,
        goal_state_keys: Sequence[AgentLocKeyType],
    ):
        """Get the next action along the shortest path from the source to the
        closest goal."""
        self._check_contains_key(source_state_key)

        terminal_node = (-1.0, -1.0, -1, -1)
        self.graph.add_node(terminal_node)
        for gsk in goal_state_keys:
            self._check_contains_key(gsk)
            self.graph.add_edge(gsk, terminal_node, action=None)

        next_state_key = self.shortest_path_next_state(source_state_key, terminal_node)
        action = self.graph.get_edge_data(source_state_key, next_state_key)["action"]

        self.graph.remove_node(terminal_node)
        return action

    def shortest_path_length(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the path shorest path length between the source and the goal."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")

import sys
sys.path.append("/home/zhihao/A2SP/rllib_A2SP")


from model.oppent_modeling_single_vision_window import Classifier_OppenentModeling_v2
import torch

model = Classifier_OppenentModeling_v2()
model.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/vision/oppent_modeling_vision_775.pth"))

opponent_modeling = model.opponent_modeling

torch.save(opponent_modeling.state_dict(), "/home/zhihao/A2SP/rllib_A2SP/model/vision/oppenent_modeling_vision.pth")

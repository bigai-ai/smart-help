import sys
sys.path.append("/home/zhihao/文档/GitHub/rllib_A2SP")

from model.opp_dep import Classifier_OppenentModeling
import torch

model = Classifier_OppenentModeling()
model.load_state_dict(torch.load("/home/zhihao/文档/GitHub/rllib_A2SP/model/op_309.pth"))

type_encoder = model.type_encoder
object_encoder = model.object_encoder
agent_encoder_without_type = model.agent_encoder
oppenent_modeling = model.opponent_modeling

torch.save(type_encoder.state_dict(), "/home/zhihao/文档/GitHub/rllib_A2SP/model/type_encoder_dep.pth")
torch.save(object_encoder.state_dict(), "/home/zhihao/文档/GitHub/rllib_A2SP/model/object_encoder_dep.pth")
torch.save(agent_encoder_without_type.state_dict(), "/home/zhihao/文档/GitHub/rllib_A2SP/model/agent_encoder_without_type_dep.pth")
torch.save(oppenent_modeling.state_dict(), "/home/zhihao/文档/GitHub/rllib_A2SP/model/oppenent_modeling_dep.pth")
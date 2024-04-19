import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

import sys
import os
sys.path.append("/home/zhihao/文档/GitHub/rllib_A2SP")


from model.agent_encoder_without_type_v2 import AgentEncoderWithoutType
from model.object_encoder_v3 import ObjectEncoder
from model.opponent_modeling_vision_window import OpponentModelingVision
import numpy as np

from env.constants import AGENT_TYPE_NUMBER, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN
from env.subtask import SUBTASK_NUM, subtask_list

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import h5py
from tqdm import tqdm

import glob
from constants import ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN
import wandb
import random
from torchvision import models
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.ops import nms
from torchvision.models.detection.ssd import SSDScoringHead


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)

# 这里输出2是为了取概率
class SSDisPickedUpHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 2 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 2)

class SSDisOpenUpHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 2 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 2)

class SSDisCookedUpHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 2 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 2)

class SSDisToggledUpHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 2 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 2)

class SSDHeightHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 1 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 1)

class SSDWeightHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 1 * anchors, kernel_size=3, padding=1))
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 1)

class CustomSSDHead(nn.Module):
    def __init__(self, in_channels, num_classes, anchor_generator):
        super().__init__()
        # 你可以从SSD模型中复用一部分头部
        self.classification_head = SSDHead(in_channels, anchor_generator, num_classes)
        # 假设你想预测一个额外的weight属性
        self.regression_head = nn.Linear(576, 4)  # 用于预测边界框
        self.attribute_head = nn.Linear(576, 2)  # 用于预测weight和can_open

    def forward(self, x):
        # 计算分类和边界框回归
        dict = self.classification_head(x)
        print(type(dict["cls_logits"]), type(dict["bbox_regression"]))
        print(dict["cls_logits"].shape, dict["bbox_regression"].shape)
        print(len(x))
        print(type(x[0]))
        print(x[0].shape)
        
        # # 计算额外的属性
        # attribute_pred = self.attribute_head(x[0])
        
        return dict
        return dict["cls_logits"], dict["bbox_regression"]

# 替换SSD的backbone为MobileNetV3Small
def create_mobilenetv3_ssd(num_classes, is_pretrained=True):
    # 加载预训练的MobileNetV3模型
    if is_pretrained:
        backbone = mobilenet_v3_small(pretrained=True).features
    else:
        backbone = mobilenet_v3_small(pretrained=False).features

    # MobileNetV3的最后一个卷积层的输出通道数
    # 这是需要根据你的backbone修改的，对于MobileNetV3Small通常是576
    backbone_out_channels = 576

    # 根据SSD框架，需要修改backbone，使其不包括池化层和分类层
    backbone = nn.Sequential(*list(backbone.children())[:-1])

    # 加载SSD模型
    model = ssd300_vgg16(pretrained=False, progress=False, num_classes=91, pretrained_backbone=False)

    # 创建一个新的SSD头部，用于替换默认的SSD头部
    num_anchors = model.anchor_generator.num_anchors_per_location()
    print(num_anchors)
    # new_head = CustomSSDHead([96, 576], num_classes, [4, 4])
    new_head = CustomSSDHead([96], num_classes, num_anchors)
    
    # 替换SSD模型的头部
    model.backbone = backbone
    model.head = new_head

    return model

# 定义物体的类别数，加上背景类别
num_classes = 62 + 1  # 30个物体类别 + 1个背景类别

# 创建模型
model = create_mobilenetv3_ssd(num_classes)

with h5py.File("/home/zhihao/文档/GitHub/rllib_A2SP/dataset_detect/image_1.h5", 'r') as f:
    # 获取对应索引的数据和标签
    image = f['image'][:]
    print(image.shape)
    image = torch.from_numpy(image)
    image = image.float() / 255.0
    image = image.unsqueeze(0)
    model.eval()


    print("forwarding")
    output = model(image)
    # print(output)
    print(output[0]['labels'].shape)

    # 提取第一个图像的预测结果
    prediction = output[0]

    # 置信度阈值和NMS阈值
    confidence_threshold = 0.5
    nms_threshold = 0.4

    # 对每个类别应用置信度阈值和NMS
    scores = prediction['scores']
    labels = prediction['labels']
    boxes = prediction['boxes']

    # 筛选出置信度大于阈值的预测
    indices = scores > confidence_threshold
    scores = scores[indices]
    labels = labels[indices]
    boxes = boxes[indices]

    # 应用NMS并获取保留的索引
    keep_indices = nms(boxes, scores, nms_threshold)

    # 根据NMS后的索引保留结果
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    boxes = boxes[keep_indices]

    # 如果需要限制最大检测数量，可以根据分数排序后选取前N个检测结果
    max_detections = 30
    if len(scores) > max_detections:
        top_scores, top_indices = scores.topk(max_detections)
        scores = top_scores
        labels = labels[top_indices]
        boxes = boxes[top_indices]

    # 最终的检测结果
    detected_boxes = boxes
    detected_labels = labels
    detected_scores = scores

    print(labels)

# 检查模型结构
# print(model)

exit()

# def cross_entropy(tensor1, tensor2):
#     tmp = tensor1 * torch.log(tensor2 + 1e-8)
#     return -torch.sum(tmp)

def criterion(input_matrix, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict):
    batch_size = input_matrix.shape[0]

    subtask_name = input_matrix[:, 30, 0].to(torch.int).cuda()
    tar_index_1 = input_matrix[:, 30, 1].to(torch.int).cuda()
    tar_index_2 = input_matrix[:, 30, 2].to(torch.int).cuda()
    # subtask_name.required_grad = False
    # tar_index_1.required_grad = False
    # tar_index_2.required_grad = False
    # type_name = x[:, 31, 0].int()
    # main_agent id: 1!!!
    main_type = input_matrix[:, 32, 0:6]
    # main_type.required_grad = False

    subtask_predict_index = torch.argmax(subtask_predict, dim=1)
    subtask_predict_success_num = torch.sum(subtask_predict_index == subtask_name)
    tar_index_1_predict_index = torch.argmax(tar_index_1_predict, dim=1)
    tar_index_1_predict_success_num = torch.sum(tar_index_1_predict_index == tar_index_1)
    tar_index_2_predict_index = torch.argmax(tar_index_2_predict, dim=1)
    tar_index_2_predict_success_num = torch.sum(tar_index_2_predict_index == tar_index_2)
    total_num = subtask_name.shape[0]
    subtask_acc = subtask_predict_success_num / total_num
    tar_index_1_acc = tar_index_1_predict_success_num / total_num
    tar_index_2_acc = tar_index_2_predict_success_num / total_num
    height_distance = torch.mean(torch.abs(type_predict[:, 0] - main_type[:, 0]))
    weight_distance = torch.mean(torch.abs(type_predict[:, 1] - main_type[:, 1]))
    open_distance = torch.mean(torch.abs(type_predict[:, 2] - main_type[:, 2]))
    close_distance = torch.mean(torch.abs(type_predict[:, 3] - main_type[:, 3]))
    toggle_on_distance = torch.mean(torch.abs(type_predict[:, 4] - main_type[:, 4]))
    toggle_off_distance = torch.mean(torch.abs(type_predict[:, 5] - main_type[:, 5]))

    goal_name_probablity = torch.zeros(batch_size, SUBTASK_NUM).cuda()
    goal_name_probablity[np.arange(len(subtask_name)), subtask_name] = 1
    tar_index_1_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_1_probability[np.arange(len(tar_index_1)), tar_index_1] = 1
    tar_index_2_probability = torch.zeros(batch_size, ALL_OBJECTS_TYPE_LENGTH_IN_KITCHEN).cuda()
    tar_index_2_probability[np.arange(len(tar_index_2)), tar_index_2] = 1

    total_loss = F.binary_cross_entropy(subtask_predict, goal_name_probablity)
    total_loss = total_loss + F.binary_cross_entropy(tar_index_1_predict, tar_index_1_probability)
    total_loss = total_loss + F.binary_cross_entropy(tar_index_2_predict, tar_index_2_probability) / 3
    total_loss = total_loss + (torch.sum((type_predict - main_type) ** 2)) / 3

    wandb.log(
        {
            "loss1": F.binary_cross_entropy(subtask_predict, goal_name_probablity), 
            "loss2": F.binary_cross_entropy(tar_index_1_predict, tar_index_1_probability),
            "loss3": F.binary_cross_entropy(tar_index_2_predict, tar_index_2_probability) / 3,
            "loss4": (torch.sum((type_predict - main_type) ** 2)) / 3,
        }
    )

    # print("for debug loss", torch.max(subtask_predict), torch.min(subtask_predict), torch.max(tar_index_1_predict), torch.min(tar_index_1_predict), torch.max(tar_index_2_predict), torch.min(tar_index_2_predict), torch.max(type_predict), torch.min(type_predict))
    # print("for debug loss", torch.max(goal_name_probablity), torch.min(goal_name_probablity), torch.max(tar_index_1_probability), torch.min(tar_index_1_probability), torch.max(tar_index_2_probability), torch.min(tar_index_2_probability), torch.max(main_type), torch.min(main_type))
    # print("for debug loss", torch.max(total_loss), torch.min(total_loss))

    return total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, train=True, train_ratio=0.9, use_ratio=1.0):
        self.file_path = file_path
        self.len = 0
        self.files = {}
        for subtask in subtask_list:
            self.files[subtask] = glob.glob(os.path.join(file_path, f"{subtask}/*"))
            self.files[subtask].sort()
            if train:
                self.files[subtask] = self.files[subtask][:int(len(self.files[subtask]) * train_ratio * use_ratio)]
            else:
                self.files[subtask] = self.files[subtask][int(len(self.files[subtask]) * train_ratio * use_ratio):int(len(self.files[subtask]) * use_ratio)]
            self.len = max(self.len, len(self.files[subtask]))
        # self.files = glob.glob(os.path.join(file_path, "*"))
        # self.files.sort()
        # if train:
        #     self.files = self.files[:int(len(self.files) * train_ratio*use_ratio)]
        # else:
        #     self.files = self.files[int(len(self.files) * train_ratio*use_ratio):int(len(self.files)*use_ratio)]

    def __getitem__(self, index):
        subtask = random.choice(subtask_list)
        index = int(index / self.len * len(self.files[subtask]))
        with h5py.File(self.files[subtask][index], 'r') as f:
            # 获取对应索引的数据和标签
            sequence = f['x'][:]
            data = f['data'][:]
            return torch.from_numpy(sequence), torch.from_numpy(data)

    def __len__(self):
        # 数据集的长度为'matrices'数据集的长度
        return self.len

if __name__ == '__main__':
    wandb.init(
        project="OppenentModelingVision"
    )   

    # 创建数据集
    dataset_train = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/data_resample_x/', use_ratio=1)
    dataset_test = TrajectoryDataset('/home/zhihao/文档/GitHub/rllib_A2SP/data_resample_x/', train=False, use_ratio=1)

    batch_size = 32

    # 创建数据加载器
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    print(len(dataset_train))

    # with h5py.File('trajectories.h5', 'r') as f:
    #     # 获取对应索引的数据和标签
    #     sequence = f['data'][0]
    #     print(sequence.shape)
    #     print(f['data'].shape)
    #     print(f['data'].shape[0])

    # count = 1
    # # 现在可以在训练循环中使用dataloader
    # for batch in dataloader:
    #     sequences = batch
    #     print(count)
    #     count += 1
    #     # 使用sequences和labels进行训练

    model = Object_Detection()
    # model.load_state_dict(torch.load("/home/zhihao/A2SP/rllib_A2SP/model/oppent_modeling_single_0.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    epoch_num = 1000
    batch_num = int(len(dataset_train) / batch_size)
    truncate_backprop = 5

    if torch.cuda.is_available():
        model = model.cuda()

    with tqdm(total=epoch_num, desc="epoch") as pbar:
        for epoch in range(epoch_num):
            with tqdm(total=batch_num, desc="batch") as pbar_small:
                for x_batch, data_batch in dataloader:
                    # for debug 
                    # break
                    # model.opponent_modeling.reset_lstm_state(batch_size=batch_size)
                    x_batch = x_batch.squeeze()
                    data_batch = data_batch.squeeze()
                    # batch == 1 的时候会被squeeze掉
                    if len(x_batch.shape) == 4:
                        x_batch = x_batch.unsqueeze(0)
                        data_batch = data_batch.unsqueeze(0)
                    data = x_batch[:]
                    data.squeeze()
                    input_matrix = data_batch[:, 4]
                    input_matrix.squeeze()
                    if torch.cuda.is_available():
                        data = data.cuda()
                        input_matrix = input_matrix.cuda()
                    # 前向传播
                    subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = model(data)

                    # 计算损失
                    total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance = criterion(input_matrix, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict)
                    
                    # 反向传播和参数更新
                    optimizer.zero_grad()
                    total_loss.backward(retain_graph=True)
                    # total_loss.backward()

                    # 进行梯度裁剪
                    # clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    wandb.log(
                        {
                            "loss": total_loss, 
                            "subtask_acc": subtask_acc, 
                            "tar_index_1_acc": tar_index_1_acc,
                            "tar_index_2_acc": tar_index_2_acc,
                            "weight_distance": weight_distance,
                            "height_distance": height_distance,
                            "open_distance": open_distance,
                            "close_distance": close_distance,
                            "toggle_on_distance": toggle_on_distance,
                            "toggle_off_distance": toggle_off_distance,
                        }
                    )
                    pbar_small.update(1)
            with torch.no_grad():
                count = 1
                all_subtask_acc = 0
                all_tar_index_1_acc = 0
                all_tar_index_2_acc = 0
                all_weight_distance = 0
                all_height_distance = 0
                all_open_distance = 0
                all_close_distance = 0
                all_toggle_on_distance = 0
                all_toggle_off_distance = 0
                for x_batch, data_batch in dataloader_test:
                    # model.opponent_modeling.reset_lstm_state(batch_size=batch_size)
                    x_batch = x_batch.squeeze()
                    data_batch = data_batch.squeeze()
                    if len(x_batch.shape) == 4:
                        x_batch.unsqueeze(0)
                        data_batch.unsqueeze(0)
                    data = x_batch[:]
                    data.squeeze()
                    input_matrix = data_batch[:, 4]
                    input_matrix.squeeze()
                    if torch.cuda.is_available():
                        data = data.cuda()
                        input_matrix = input_matrix.cuda()
                    # 前向传播
                    subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict = model(data)

                    # 计算损失
                    total_loss, subtask_acc, tar_index_1_acc, tar_index_2_acc, weight_distance, height_distance, open_distance, close_distance, toggle_on_distance, toggle_off_distance = criterion(input_matrix, subtask_predict, tar_index_1_predict, tar_index_2_predict, type_predict)
                    
                    all_subtask_acc = all_subtask_acc / count * (count - 1) + subtask_acc / count
                    all_tar_index_1_acc = all_tar_index_1_acc / count * (count - 1) + tar_index_1_acc / count
                    all_tar_index_2_acc = all_tar_index_2_acc / count * (count - 1) + tar_index_2_acc / count
                    all_weight_distance = all_weight_distance / count * (count - 1) + weight_distance / count
                    all_height_distance = all_height_distance / count * (count - 1) + height_distance / count
                    all_open_distance = all_open_distance / count * (count - 1) + open_distance / count
                    all_close_distance = all_close_distance / count * (count - 1) + close_distance / count
                    all_toggle_on_distance = all_toggle_on_distance / count * (count - 1) + toggle_on_distance / count
                    all_toggle_off_distance = all_toggle_off_distance / count * (count - 1) + toggle_off_distance / count
                    
                    count += 1
                wandb.log(
                    {
                        "test_subtask_acc": all_subtask_acc, 
                        "test_tar_index_1_acc": all_tar_index_1_acc,
                        "test_tar_index_2_acc": all_tar_index_2_acc,
                        "test_weight_distance": all_weight_distance,
                        "test_height_distance": all_height_distance,
                        "test_open_distance": all_open_distance,
                        "test_close_distance": all_close_distance,
                        "test_toggle_on_distance": all_toggle_on_distance,
                        "test_toggle_off_distance": all_toggle_off_distance,
                    }
                )
            if epoch % 9 == 0:
                torch.save(model.state_dict(), f"/home/zhihao/文档/GitHub/rllib_A2SP/model/oppent_modeling_vision_{epoch}.pth")
            pbar.update(1)
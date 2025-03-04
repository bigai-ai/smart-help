import json
import os
import os.path as osp



SR_list = []
GSR_list = []
CR_list = []
HE_list = []
HN_list = []
helping_num = 0

data_root = 'debug_llm'
dirs = os.listdir(data_root)
tmp_SR_list = []
tmp_GSR_list = []
tmp_CR_list = []
tmp_HE_list = []
tmp_HN_list = []
tmp_helping_num = 0
tmp_need_help_num = 0
rewards = []
episode_len = []
SPL = []

for dir in dirs:
    with open(osp.join(data_root, dir), 'r') as fp:
        data = json.load(fp)
        if True:
            tmp_SR_list.append(data['SR'])
            tmp_GSR_list.append(data['GSR'])
            tmp_CR_list.append(data['CR'])
            SR_list.append(data['SR'])
            GSR_list.append(data['GSR'])
            CR_list.append(data['CR'])
            episode_len.append(len(data['llm_action']))
            rewards.append(data['total_reward'])
            SPL.append(data['SPL'])
            if data['HN'] != -1:
                tmp_helping_num += 1
                helping_num += 1
                tmp_HE_list.append(data['HE'])
                tmp_HN_list.append(data['HN'])
                HE_list.append(data['HE'])
                HN_list.append(data['HN'])
            tmp_need_help_num += 1
print("average_SR : ", sum(tmp_SR_list) / len(tmp_SR_list))
print("average_GSR: ", sum(tmp_GSR_list) / len(tmp_GSR_list))
print("average_CR : ", sum(tmp_CR_list) / len(tmp_CR_list))
print("average_HE : ", sum(tmp_HE_list) / max(len(tmp_HE_list), 1))
print("average_HN : ", sum(tmp_HN_list) / max(len(tmp_HN_list), 1))
print("helping_num : ", tmp_helping_num)
print("need_help_num:", tmp_need_help_num)
print('episode len: ', sum(episode_len) / len(episode_len))
print('reward: ', sum(rewards) / len(rewards))
print('SPL: ', sum(SPL) / len(SPL))

# print("=======END=======")
# print("average_SR : ", sum(SR_list) / len(SR_list))
# print("average_GSR: ", sum(GSR_list) / len(GSR_list))
# print("average_CR : ", sum(CR_list) / len(CR_list))
# print("average_HE : ", sum(HE_list) / max(len(HE_list), 1))
# print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
# print("helping_num : ", helping_num)
# # print("need_help_num:", need_help_num)






# for type_index in range(7):
#     tmp_SR_list = []
#     tmp_GSR_list = []
#     tmp_CR_list = []
#     tmp_HE_list = []
#     tmp_HN_list = []
#     tmp_helping_num = 0
#     # tmp_need_help_num = 0
#     for env_index in range(0, 30):
#         with open(f'test_llm_result/MakeBreakfast_{type_index}_{env_index}.json', 'r') as fp:
#             data = json.load(fp)
#             tmp_SR_list.append(data['SR'])
#             tmp_GSR_list.append(data['GSR'])
#             tmp_CR_list.append(data['CR'])
#             SR_list.append(data['SR'])
#             GSR_list.append(data['GSR'])
#             CR_list.append(data['CR'])
#             if data['HN'] != -1:
#                 tmp_helping_num += 1
#                 helping_num += 1
#                 tmp_HE_list.append(data['HE'])
#                 tmp_HN_list.append(data['HN'])
#                 HE_list.append(data['HE'])
#                 HN_list.append(data['HN'])
#     print(type_index)
#     print("average_SR : ", sum(tmp_SR_list) / len(tmp_SR_list))
#     print("average_GSR: ", sum(tmp_GSR_list) / len(tmp_GSR_list))
#     print("average_CR : ", sum(tmp_CR_list) / len(tmp_CR_list))
#     print("average_HE : ", sum(tmp_HE_list) / max(len(tmp_HE_list), 1))
#     print("average_HN : ", sum(tmp_HN_list) / max(len(tmp_HN_list), 1))
#     print("helping_num : ", tmp_helping_num)
#     # print("need_help_num:", tmp_need_help_num)

# print("=======END=======")
# print("average_SR : ", sum(SR_list) / len(SR_list))
# print("average_GSR: ", sum(GSR_list) / len(GSR_list))
# print("average_CR : ", sum(CR_list) / len(CR_list))
# print("average_HE : ", sum(HE_list) / max(len(HE_list), 1))
# print("average_HN : ", sum(HN_list) / max(len(HN_list), 1))
# print("helping_num : ", helping_num)
# # print("need_help_num:", need_help_num)






import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

# 读取CSV文件
df = pd.read_csv('/home/zhihao/ray_results/e2e/progress.csv')
# df = pd.read_csv('./combined_file.csv')
# df = pd.read_csv('/home/zhihao/ray_results/CustomPPO_Single_Env_Symbolic_2023-09-15_14-35-3874iyt6mb/progress.csv')


def draw_total_loss(df, window_size=20):

    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/learner_stats/total_loss']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('total_loss')
    # plt.grid(True)

    # 添加滑动平均曲线
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_policy_loss(df, window_size=20):
    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/learner_stats/policy_loss']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('policy_loss')
    # plt.grid(True)

    # 添加滑动平均曲线
    window_size = 20  # 滑动窗口的大小
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_vf_loss(df, window_size=20):
    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/learner_stats/vf_loss']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('vf_loss')
    # plt.grid(True)

    # 添加滑动平均曲线
    window_size = 20  # 滑动窗口的大小
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_mean_step(df, window_size=20):
    # 将日期列解析为日期时间格式
    total_loss = df['episode_len_mean']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('episode_len_mean')
    # plt.grid(True)

    # 添加滑动平均曲线
    window_size = 20  # 滑动窗口的大小
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_reward(df, window_size=20):
    # 将日期列解析为日期时间格式
    reward_max = df['episode_reward_max']
    reward_mean = df['episode_reward_mean']
    reward_min = df['episode_reward_min']
    total_loss = df['episode_len_mean']
    data_num = len(reward_mean)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, reward_mean, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('reward')
    # plt.grid(True)

    # 添加滑动平均曲线
    rolling_mean = reward_mean.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_subtask_acc(df, window_size=20):

    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/custom_metrics/subtask_success_rate']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('subtask_success')
    # plt.grid(True)

    # 添加滑动平均曲线
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_tar_index_1(df, window_size=20):

    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/custom_metrics/tar_index_1_rate']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('tar_index_1')
    # plt.grid(True)

    # 添加滑动平均曲线
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

def draw_type(df, window_size=20):

    # 将日期列解析为日期时间格式
    total_loss = df['info/learner/default_policy/custom_metrics/type_rate']
    data_num = len(total_loss)
    index = list(range(data_num))

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(index, total_loss, marker='o', color='b')
    plt.title('training')
    plt.xlabel('epoch')
    plt.ylabel('type')
    # plt.grid(True)

    # 添加滑动平均曲线
    rolling_mean = total_loss.rolling(window=window_size).mean()
    plt.plot(index, rolling_mean, linestyle='-', color='r', label=f'window_size:{window_size}')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    draw_total_loss(df, window_size=8)
    # draw_subtask_acc(df)
    # draw_tar_index_1(df)
    # draw_type(df)
    # draw_policy_loss(df, 20)
    # draw_vf_loss(df, 20)
    # draw_mean_step(df, 3)
    draw_reward(df, window_size=8)
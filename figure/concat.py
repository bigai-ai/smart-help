import pandas as pd

# 读取两个csv文件
df1 = pd.read_csv('/home/zhihao/ray_results/exp_2/progress.csv')
df2 = pd.read_csv('/home/zhihao/ray_results/exp_2.2/progress.csv')

# 垂直整合两个DataFrame
result = pd.concat([df1, df2], ignore_index=True)

# 保存结果到新的csv文件
result.to_csv('combined_file.csv', index=False)
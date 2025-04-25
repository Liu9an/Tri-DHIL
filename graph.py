import matplotlib.pyplot as plt
import numpy as np

# 示例数据
data = {
    'Name': ['NYU', 'UM', 'USM', 'UCLA', 'LEUVEN', 'MAX MUN', 'YALE', 'PITT', 'KKI', 'TRINITY', 'STANFORD', 'CALTECH', 'OLIN', 'SDSU', 'SBL', 'OHSU', 'CMU'],
    'C1': [105, 77, 43, 45, 33, 33, 28, 27, 31, 25, 20, 19, 16, 22, 15, 15, 13],
    'C2': [79, 66, 57, 54, 29, 24, 28, 29, 17, 24, 20, 19, 20, 14, 15, 13, 12],
    'Triplets': [16406, 10021, 4802, 4761, 1852, 1527, 1512, 1510, 1006, 1151, 760, 684, 604, 580, 420, 362, 287],
    'G': [71.19, 71.48, 71.66, 70.76, 71.67, 71.81, 71.26, 71.42, 71.73, 71.51, 71.60, 71.71, 71.58, 71.60, 71.35, 71.71, 71.11]
}

# 数据处理
names = data['Name']
c1 = data['C1']
c2 = data['C2']

# 绘制堆积条形图
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(names))
width = 0.35

rects1 = ax.bar(x, c1, width, label='C1')
rects2 = ax.bar(x, c2, width, bottom=c1, label='C2')

ax.set_xlabel('Name')
ax.set_ylabel('Values')
ax.set_title('Grouped Stacked Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()

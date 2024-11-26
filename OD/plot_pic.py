# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义数据
# data = np.array([
#     0.02215, 0.02815, 0.072912, 0.02815, 0.042916, 0.011998, 0.059991,
#     0.052607, 0.081218, 0.021228, 0.021228, 0.069682, 0.025842, 0.015228,
#     0.29257, 0.053992, 0.039225, 0.019382, 0.01892, 0.022612
# ])
#
# # 定义横坐标标签
# categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#
# # 创建柱状图
# plt.figure(figsize=(12, 6))
# plt.bar(range(len(data)), data, color='skyblue')
#
# # 添加标题和标签
# plt.title('Coreset')
# plt.xlabel('Category')
# plt.ylabel('Value')
#
# # 设置x轴的类别标签
# plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
#
# # 显示图形
# plt.tight_layout()  # 自动调整布局，防止标签重叠
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# 假设你有四个列表
list1 = [0.248, 0.355, 0.404, 0.415, 0.424, 0.434, 0.437, 0.444, 0.443, 0.445, 0.446]
list2 = [0.248, 0.324, 0.381, 0.396, 0.415, 0.427, 0.431, 0.436, 0.440,0.440, 0.441]
list3 = [0.252, 0.324, 0.383,0.397, 0.413, 0.423, 0.422, 0.427, 0.429, 0.432, 0.436]
list4 = [0.253,0.312, 0.340, 0.382,  0.398,  0.412, 0.417, 0.421, 0.421, 0.420,0.422]
list5 = [0.251, 0.339,0.379,  0.392, 0.412, 0.417,0.419, 0.423, 0.427, 0.429, 0.432]
list6=[0.246, 0.319, 0.373, 0.394, 0.409, 0.416, 0.419, 0.423, 0.424, 0.424, 0.425]



# 打印均值
print("Mean of list1:", np.mean(list1))
print("Mean of list2:", np.mean(list2))
print("Mean of list3:", np.mean(list3))
print("Mean of list4:", np.mean(list4))
print("Mean of list5:", np.mean(list5))
print("Mean of list6:", np.mean(list6))

# 生成索引，横坐标
indices = [i for i in range(len(list1))]

# 计算面积
area_list1 = np.trapz(list1, indices)/11
area_list2 = np.trapz(list2, indices)/11
area_list3 = np.trapz(list3, indices)/11
area_list4 = np.trapz(list4, indices)/11
# area_list5 = np.trapz(list5, indices)/11
# area_list6 = np.trapz(list6, indices)/11



print(f"Area under the curve (List1 : {area_list1}")
print(f"Area under the curve (List2 : {area_list2}")
print(f"Area under the curve (List3 : {area_list3}")
print(f"Area under the curve (List4 : {area_list4}")
# print(f"Area under the curve (List5 : {area_list5}")
# print(f"Area under the curve (List6 : {area_list6}")

# 绘制折线图
plt.plot(indices, list1, label='Ours', marker='o', color='red', linestyle='-', linewidth=2)
plt.plot(indices, list2, label='FAL', marker='x', color='blue', linestyle='--', linewidth=2)
plt.plot(indices, list3, label='Random', marker='s', color='green', linestyle='-.', linewidth=2)
plt.plot(indices, list4, label='Coreset', marker='d', color='orange', linestyle=':', linewidth=2)
plt.plot(indices, list5, label='LeastConf', marker='+', color='gray', linestyle=':', linewidth=2)
plt.plot(indices, list6, label='CALD', marker='^', color='purple', linestyle='-', linewidth=2)

# plt.plot(indices, list1, label='TSDAL', marker='o', color='red', linestyle='-', linewidth=2)
# plt.plot(indices, list2, label='w/o Stage1', marker='s', color='green', linestyle='-.', linewidth=2)
# plt.plot(indices, list3, label='w/o Stage2', marker='d', color='orange', linestyle=':', linewidth=2)
# plt.plot(indices, list4, label='w/o Weighting feature representation ', marker='x', color='blue', linestyle='--', linewidth=2)

# 添加标签和标题
plt.xlabel('Round $r$', fontsize=20)
plt.ylabel('mAP@0.5:0.95', fontsize=20)
plt.title('Faster R-CNN on VOC2012', fontsize=20)

plt.legend(fontsize=14)

# 设置刻度字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)

# 显示图像
plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义常量 D
# D = 1.0  # 可以根据需要更改
#
# # 第一个函数: y = 400D(2^x - 1)
# def func1(x, D):
#     return 400 * D * (2**x - 1)
#
# # 第二个函数: y = 400Dx
# def func2(x, D):
#     return 400 * D * x
#
# # 生成 x 值
# x = np.linspace(0, 10, 100)
#
# # 计算 y 值
# y1 = func1(x, D)
# y2 = func2(x, D)
#
# # 绘制图像
# plt.plot(x, y1, label=r'Fleet AL')
# plt.plot(x, y2, label=r'Ours')
# plt.xlabel('n')
# plt.legend()
# plt.title('Communication cost')
# plt.grid(True)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

list1 = [0.248, 0.355, 0.404, 0.415, 0.424, 0.434, 0.437, 0.444, 0.443, 0.445, 0.446]
list2 = [0.248, 0.324, 0.381, 0.396, 0.415, 0.427, 0.431, 0.436, 0.440,0.440, 0.441]
list3 = [0.252, 0.324, 0.383,0.397, 0.413, 0.423, 0.422, 0.427, 0.429, 0.432, 0.436]
list4 = [0.253,0.312, 0.340, 0.382,  0.398,  0.412, 0.417, 0.421, 0.421, 0.420,0.422]
list5 = [0.251, 0.339,0.379,  0.392, 0.412, 0.417,0.419, 0.423, 0.427, 0.429, 0.432]
list6=[0.246, 0.319, 0.373, 0.394, 0.409, 0.416, 0.419, 0.423, 0.424, 0.424, 0.425]



print("Mean of list1:", np.mean(list1))
print("Mean of list2:", np.mean(list2))
print("Mean of list3:", np.mean(list3))
print("Mean of list4:", np.mean(list4))
print("Mean of list5:", np.mean(list5))
print("Mean of list6:", np.mean(list6))

indices = [i for i in range(len(list1))]

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

plt.xlabel('Round $r$', fontsize=20)
plt.ylabel('mAP@0.5:0.95', fontsize=20)
plt.title('Faster R-CNN on VOC2012', fontsize=20)

plt.legend(fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

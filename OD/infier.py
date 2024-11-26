import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import cv2
import faster_rcnn_modify

# np.set_printoptions(threshold=np.inf)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(device)

# 加载模型，确保它是在推理模式
model.eval()

# 加载并预处理输入图片
img_path = '2.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 进行图像的预处理，如转换为tensor并调整大小
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

img_tensor = transform(img_rgb).unsqueeze(0)  # 扩展维度，因为模型期望输入batch格式

# 将图像移到设备上
img_tensor = img_tensor.to(device)

# 进行推理
with torch.no_grad():
    outputs = model(img_tensor)
# 处理模型输出，提取预测边界框、标签和置信度
output = outputs[0]
boxes = output['boxes'].cpu().numpy()
labels = output['labels'].cpu().numpy()
scores = output['scores'].cpu().numpy()
embeddings = output['embedding'].cpu().numpy()
print(scores)

# 仅保留置信度超过阈值的预测结果
threshold = 0.25 # as same as yolo
filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]
filtered_scores = scores[scores > threshold]
filtered_embeddings = embeddings[scores > threshold]

# 可视化推理结果
for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制边界框
    cv2.putText(img, f'Class: {label}, Score: {score:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 添加类别标签和置信度

# 显示结果
cv2.imshow('Inference', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
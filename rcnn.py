import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO labels (shortened)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light'
]

# Load image
image_path = "test.jpg"   # <-- put your image path
image = cv2.imread(image_path)

# Convert BGR → RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Transform
transform = T.Compose([T.ToTensor()])
img = transform(image_rgb)

# Inference
with torch.no_grad():
    predictions = model([img])

# Draw boxes
for box, label, score in zip(
    predictions[0]['boxes'],
    predictions[0]['labels'],
    predictions[0]['scores']
):
    if score > 0.5:
        x1, y1, x2, y2 = map(int, box)
        label_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label_name} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

# Show result
cv2.imshow("Faster R-CNN Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
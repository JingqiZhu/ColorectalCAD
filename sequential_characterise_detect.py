import torch
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os

# Load pretrained YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp3/weights/best.pt')

# Load pretained ResNet model
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 3)
resnet_model.load_state_dict(torch.load('/cs/student/projects3/aibh/2023/jingqzhu/output/best_model.pth'))
resnet_model.eval()

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3365, 0.2471, 0.1854], std=[0.3051, 0.2425, 0.1969]),
])

# Define the root directory containing the subdirectories
root_dir = '/cs/student/projects3/aibh/2023/jingqzhu/data/colonoscopic'

# List of subdirectories to process
subdirs = ['Adenoma', 'Hyperplastic', 'Serrated']

# Loop through each subdirectory
for subdir in subdirs:
    # Get the full path of the subdirectory
    subdir_path = os.path.join(root_dir, subdir)
    
    # Loop through all video files in the subdirectory
    for filename in os.listdir(subdir_path):
        if filename.endswith(".mp4"):  
            # Get the full path of the video file
            input_video_path = os.path.join(subdir_path, filename)

            base_name = os.path.splitext(os.path.basename(input_video_path))[0]

            # Create output file names based on the base name
            output_video_path = f'/cs/student/projects3/aibh/2023/jingqzhu/output/videos/{base_name}_preds.mp4'
            output_txt_path = f'/cs/student/projects3/aibh/2023/jingqzhu/output/labels/{base_name}_preds.txt'

            cap = cv2.VideoCapture(input_video_path)
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (int(cap.get(3)), int(cap.get(4))))

            label_mapping = {0: 'adenoma', 1: 'hyperplastic', 2: 'serrated'}

            output_file = open(output_txt_path, 'w')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                original_height, original_width, _ = frame.shape
                
                resized_frame = cv2.resize(frame, (512, 512))

                results = yolo_model(resized_frame)
                
                frame_info = []

                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
                for det in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    
                    x1 = (x1 / 512) * original_width
                    y1 = (y1 / 512) * original_height
                    x2 = (x2 / 512) * original_width
                    y2 = (y2 / 512) * original_height
                    
                    # Calculate the width and height of the ROI
                    roi_width = x2 - x1
                    roi_height = y2 - y1

                    old_x1, old_x2, old_y1, old_y2 = x1, x2, y1, y2

                    # Expand the height to match the width if the ROI is not square
                    if roi_width > roi_height:
                        extra_height = roi_width - roi_height
                        y1 = max(0, y1 - extra_height / 2)
                        y2 = min(original_height, y2 + extra_height / 2)
                        roi_size = roi_width
                    else:
                        extra_width = roi_height - roi_width
                        x1 = max(0, x1 - extra_width / 2)
                        x2 = min(original_width, x2 + extra_width / 2)
                        roi_size = roi_height

                    # Calculate the center of the original square ROI
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Calculate the new side length (twice the original square size)
                    new_roi_size = roi_size * 2

                    # Recalculate the new coordinates while keeping the center the same
                    new_x1 = max(0, center_x - new_roi_size / 2)
                    new_y1 = max(0, center_y - new_roi_size / 2)
                    new_x2 = min(original_width, center_x + new_roi_size / 2)
                    new_y2 = min(original_height, center_y + new_roi_size / 2)

                    # Extract the expanded ROI
                    roi = frame[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
                    
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    roi_tensor = transform(roi_pil).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = resnet_model(roi_tensor)
                        _, pred = torch.max(output, 1)
                    
                    x1_rel = old_x1 / original_width
                    y1_rel = old_y1 / original_height
                    x2_rel = old_x2 / original_width
                    y2_rel = old_y2 / original_height
                    
                    frame_info.append(f"{frame_number},{x1_rel:.4f},{y1_rel:.4f},{x2_rel:.4f},{y2_rel:.4f},{pred.item()}")
                    
                    label = label_mapping[pred.item()]
                    cv2.putText(frame, label, (int(old_x1), int(old_y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                    cv2.rectangle(frame, (int(old_x1), int(old_y1)), (int(old_x2), int(old_y2)), (255,0,0), 2)
                
                if frame_info:
                    output_file.write(" ".join(frame_info) + "\n")
                
                out.write(frame)

            cap.release()
            out.release()
            output_file.close()

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, matthews_corrcoef, classification_report

# Define the correct labels for each video type
correct_labels = {
    'adenoma': {
        'suffixes': ['30_WL', '31_WL', '10_WL', '19_WL', '15_WL', '16_WL', '24_WL', '17_WL', '20_WL', 
                     '39_WL', '38_WL', '27_WL', '20_NBI', '17_NBI', '16_NBI', '27_NBI', '05_NBI', 
                     '13_NBI', '38_NBI', '28_NBI', '40_NBI', '07_NBI', '26_NBI', '10_NBI'],
        'label': 0
    },
    'hyperplastic': {
        'suffixes': ['08_WL', '10_WL', '11_WL', '21_WL', '14_WL', '09_WL', '15_WL', '01_NBI', 
                     '18_NBI', '16_NBI', '02_NBI', '09_NBI', '06_NBI', '12_NBI'],
        'label': 1
    },
    'serrated': {
        'suffixes': ['13_WL', '11_WL', '09_WL', '06_WL', '08_WL', '10_NBI', '12_NBI', 
                     '01_NBI', '14_NBI', '06_NBI'],
        'label': 2
    }
}

# Directory containing the prediction files
predictions_dir = '/cs/student/projects3/aibh/2023/jingqzhu/output/labels'

# Initialize lists to store ground truth and predictions
y_true = []
y_pred = []

# Loop through all files in the predictions directory
for filename in os.listdir(predictions_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(predictions_dir, filename)
        
        # Determine the correct label based on the filename
        correct_label = None
        for category, data in correct_labels.items():
            if category in filename:
                for suffix in data['suffixes']:
                    if filename.endswith(suffix + '_preds.txt'):
                    
                        correct_label = data['label']
                        break
                if correct_label is not None:
                    break
        
        # If a correct label was found, read the predictions
        if correct_label is not None:
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    predicted_label = int(parts[-1])  # The predicted class is the last item in the line
                    
                    y_true.append(correct_label)
                    y_pred.append(predicted_label)

# Calculate classification metrics
conf_matrix = confusion_matrix(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# Generate the classification report
class_report = classification_report(y_true, y_pred, target_names=['adenoma', 'hyperplastic', 'serrated'])
# Output the metrics
#print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
print('\nClassification Report:')
print(class_report)
print('Confusion Matrix:')
print(conf_matrix)
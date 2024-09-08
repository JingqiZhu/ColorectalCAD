import os

def update_labels(labels_dir):
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as file:
                lines = file.readlines()
            
            with open(label_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    # Map all polyp subtypes to class 0
                    new_class_id = 0 if class_id >= 0 else class_id
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    file.write(new_line)

# Update training labels
update_labels('/cs/student/projects3/aibh/2023/jingqzhu/data/SUN/train/labels')

# Update validation labels
update_labels('/cs/student/projects3/aibh/2023/jingqzhu/data/SUN/val/labels')
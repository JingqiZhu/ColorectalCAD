import os
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def split_videos(input_dir, train_size=0.5, val_size=0.2, test_size=0.3):
    classes = ['Adenoma', 'Hyperplastic', 'Serrated']
    split_data = {'train': [], 'val': [], 'test': []}
    
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        videos = [os.path.join(cls_dir, video) for video in os.listdir(cls_dir) if video.endswith('.mp4')]
        
        # Separate WL and NBI videos, stratified split for each set
        wl_videos = [video for video in videos if 'WL' in video]
        nbi_videos = [video for video in videos if 'NBI' in video]
        
        wl_train_and_val, wl_test = train_test_split(wl_videos, test_size=test_size, random_state=12)
        wl_train, wl_val = train_test_split(wl_train_and_val, test_size=val_size/(train_size + val_size), random_state=12)
        
        nbi_train_and_val, nbi_test = train_test_split(nbi_videos, test_size=test_size, random_state=42)
        nbi_train, nbi_val = train_test_split(nbi_train_and_val, test_size=val_size/(train_size + val_size), random_state=42)
        
        split_data['train'].extend(wl_train + nbi_train)
        split_data['val'].extend(wl_val + nbi_val)
        split_data['test'].extend(wl_test + nbi_test)

        # Print the video names in each set
        print(f"Class: {cls}")
        print(f"  Train: {[os.path.basename(video).replace(f'{cls.lower()}_', '').split('.')[0] for video in wl_train + nbi_train]}")
        print(f"  Validation: {[os.path.basename(video).replace(f'{cls.lower()}_', '').split('.')[0] for video in wl_val + nbi_val]}")
        print(f"  Test: {[os.path.basename(video).replace(f'{cls.lower()}_', '').split('.')[0] for video in wl_test + nbi_test]}")

    return split_data

def extract_evenly_distributed_frames(video_path, output_dir, num_frames):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)  # Ensure interval is at least 1

    frame_indices = [i * interval for i in range(num_frames)]
    saved_frame_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
        else:
            print(f"Error reading frame at index {idx} from {video_path}")

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir}")

def count_videos_and_frames(split_data, frame_dir):
    classes = ['Adenoma', 'Hyperplastic', 'Serrated']
    phases = ['train', 'val', 'test']
    
    video_counts = {phase: {cls: {'WL': 0, 'NBI': 0} for cls in classes} for phase in phases}
    frame_counts = {phase: {cls: {'WL': 0, 'NBI': 0} for cls in classes} for phase in phases}
    
    for phase in phases:
        for video_path in split_data[phase]:
            cls = os.path.basename(os.path.dirname(video_path))
            video_name = os.path.basename(video_path)
            
            # Count videos
            if 'WL' in video_name:
                video_counts[phase][cls]['WL'] += 1
            elif 'NBI' in video_name:
                video_counts[phase][cls]['NBI'] += 1
        
        # Count frames
        for cls in classes:
            phase_frame_dir = os.path.join(frame_dir, phase, cls)
            for frame in os.listdir(phase_frame_dir):
                if frame.endswith('.jpg'):
                    if 'WL' in frame:
                        frame_counts[phase][cls]['WL'] += 1
                    elif 'NBI' in frame:
                        frame_counts[phase][cls]['NBI'] += 1
    
    return video_counts, frame_counts

def plot_counts(video_counts, frame_counts, save_path):
    classes = ['Adenoma', 'Hyperplastic', 'Serrated']
    phases = ['train', 'val', 'test']
    plt.rcParams.update({'font.size': 16})  # adjust this value to make the font larger or smaller
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, phase in enumerate(phases):
        wl_video_values = [video_counts[phase][cls]['WL'] for cls in classes]
        nbi_video_values = [video_counts[phase][cls]['NBI'] for cls in classes]
        wl_frame_values = [frame_counts[phase][cls]['WL'] for cls in classes]
        nbi_frame_values = [frame_counts[phase][cls]['NBI'] for cls in classes]
        
        width = 0.35
        x = range(len(classes))
        
        # Plot number of videos against classes
        axs[0, i].bar(x, wl_video_values, width, label='WL', color='skyblue')
        axs[0, i].bar([p + width for p in x], nbi_video_values, width, label='NBI', color='darkblue')
        axs[0, i].set_title(f'{phase.capitalize()}')
        axs[0, i].set_ylabel('Number of Videos')
        axs[0, i].set_xticks([p + width / 2 for p in x])
        axs[0, i].set_xticklabels(classes)
        axs[0, i].legend()
        axs[0, i].set_ylim(0, 20)
        axs[0, i].yaxis.set_major_locator(MultipleLocator(2))
        
        # Plot number of frames against classes
        axs[1, i].bar(x, wl_frame_values, width, label='WL', color='lightgreen')
        axs[1, i].bar([p + width for p in x], nbi_frame_values, width, label='NBI', color='darkgreen')
        axs[1, i].set_title(f'{phase.capitalize()}')
        axs[1, i].set_ylabel('Number of Frames')
        axs[1, i].set_xticks([p + width / 2 for p in x])
        axs[1, i].set_xticklabels(classes)
        axs[1, i].legend()
        axs[1, i].set_ylim(0, 1000)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

input_dir = '../data/colonoscopic'  # Directory of videos
split_data = split_videos(input_dir)
output_frame_dir = '../data/colonoscopic/frames'  # Directory of extracted frames

# num_frames_to_extract = 50  # Number of frames to extract from each video
frame_extraction_requirements = {
    'Adenoma': 50,
    'Hyperplastic': 50,
    'Serrated': 50
}

for phase in ['train', 'val', 'test']:
    for video_path in split_data[phase]:
        cls = os.path.basename(os.path.dirname(video_path))
        frame_output_dir = os.path.join(output_frame_dir, phase, cls)
        num_frames = frame_extraction_requirements.get(cls, 50)  # Default to 50 frames if class not found
        extract_evenly_distributed_frames(video_path, frame_output_dir, num_frames=num_frames)

# Get video and frame counts
video_counts, frame_counts = count_videos_and_frames(split_data, output_frame_dir)
print("Video counts:", video_counts)
print("Frame counts:", frame_counts)

# Plot the number of videos and frames
save_path = '../plots/video_frame_counts.png'
plot_counts(video_counts, frame_counts, save_path)
import os
import cv2
import re
import glob
from sys import argv


DATASET_ROOT = argv[1]
VIDEOS_DIR = os.path.join("videos", "used")

# Regex for filenames: xy_frame_counter
FILENAME_REGEX = re.compile(r"^(\d+)_(frame_\d+)\.txt$")

def get_video_map(video_dir):
    """
    Create a dict that for each video has: {video_name: video_path}
    """
    video_map = {}
    if not os.path.exists(video_dir):
        print(f"[ERROR] video dir not exists: {video_dir}")
        return video_map

    for video in os.listdir(video_dir):
        name, ext = os.path.splitext(video)
        if ext == ".mp4":
            video_map[name] = os.path.join(video_dir, video)
    
    print(f"Found {len(video_map)} videos")
    return video_map

def build_tasks(dataset_root):
    """
    Check each label in train and val and create a task dict that contains all the frame to extract for each video
    """
    tasks = {}
    
    for split in ['train', 'val']:
        labels_dir = os.path.join(dataset_root, split, "labels")
        images_dir = os.path.join(dataset_root, split, "images")
        
        os.makedirs(images_dir, exist_ok=True)
        
        if not os.path.exists(labels_dir):
            continue
        
        txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        for txt_path in txt_files:
            filename = os.path.basename(txt_path)
            
            match = FILENAME_REGEX.match(filename)
            if match:
                frame_num_str = match.group(2).split('_')[1]
                video_name = f"Zona_2_{match.group(1)}"
                frame_id = int(frame_num_str)
                
                img_name = filename.replace(".txt", ".jpg")
                image_path = os.path.join(images_dir, img_name)
                
                if video_name not in tasks:
                    tasks[video_name] = []
                
                tasks[video_name].append({
                    'frame_id': frame_id,
                    'image_path': image_path
                })
            else:
                pass

    return tasks

def extract_frames(tasks, video_map):
    print("\n--- Extracting frames ---")
    
    frame_extracted = 0
    frame_skipped = 0
    
    for video_name, requests in tasks.items():
        if video_name not in video_map:
            print(f"[ERROR] Video not found: {video_name}. Skipped {len(requests)} frames.")
            continue
            
        video_path = video_map[video_name]
        
        # Skips images that are already in the folder
        pending_requests = [r for r in requests if not os.path.exists(r['image_path'])]
        skipped = len(requests) - len(pending_requests)
        frame_skipped += skipped
        
        if not pending_requests:
            print(f"[SKIP] {video_name}: All frames already extracted\n")
            continue

        # Dict { frame_id: image_path }
        targets = { req['frame_id']: req['image_path'] for req in pending_requests }
        
        max_frame = max(targets.keys())
        
        print(f"[PROCESS] {video_name}: Extracting {len(targets)} frame (stop on frame {max_frame})")
        

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Couldn't open {video_path}")
            continue

        current_frame_id = 0
        extracted_count = 0
        
        while True:
            if current_frame_id > max_frame:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame_id in targets:
                image_path = targets[current_frame_id]
                cv2.imwrite(image_path, frame)
                frame_extracted += 1
                
                if frame_extracted % 10 == 0:
                    print(f"Extracting frame {current_frame_id}", end='\r')

            current_frame_id += 1
        
        cap.release()
        print(f"[FINISH] {video_name} ended\n")

    print("\n--- SUMMARY ---")
    print(f"Frame extracted: {frame_extracted}")
    print(f"Frame already extracted (skip): {frame_skipped}")


if __name__ == "__main__":
    video_mapping = get_video_map(VIDEOS_DIR)
    
    if not video_mapping:
        print(f"No video found in {VIDEOS_DIR}")
        exit()

    tasks_dict = build_tasks(DATASET_ROOT)
    
    if not tasks_dict:
        print("No label found")
        exit()
        
    extract_frames(tasks_dict, video_mapping)
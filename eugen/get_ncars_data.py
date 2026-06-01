#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# Add g1-resnet to path to import prophesee_utils
sys.path.append(os.path.abspath('g1-resnet'))
try:
    from prophesee_utils.io.psee_loader import PSEELoader
except ImportError:
    # Fallback in case of running from g1-resnet subdirectory
    sys.path.append(os.path.abspath('.'))
    from prophesee_utils.io.psee_loader import PSEELoader

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess N-Cars dataset for EMS-YOLO")
    parser.add_argument("--path", type=str, default="/scratch/datasets/ncars_raw", help="Path to raw N-Cars dataset")
    parser.add_argument("--outpath", type=str, default="/scratch/datasets/ncars_processed", help="Path to save processed dataset")
    parser.add_argument("-T", type=int, default=5, help="Number of time-steps")
    return parser.parse_args()

def create_frame_tensor(video, T, width=304, height=240, duration_us=100000):
    video.seek_time(0)
    delta_t = duration_us // T
    events = []
    for _ in range(T):
        events.append(video.load_delta_t(delta_t))
        
    img = 127 * np.ones((T, height, width, 3), dtype=np.uint8)
    for i in range(T):
        if len(events[i]):
            # Clip coordinates to be safe
            x = np.clip(events[i]['x'], 0, width - 1)
            y = np.clip(events[i]['y'], 0, height - 1)
            p = events[i]['p']
            img[i, y, x, :] = 255 * p[:, None]
            
    # Resize to 320x320 for EMS-YOLO compatibility
    resized_img = np.zeros((T, 320, 320, 3), dtype=np.uint8)
    for i in range(T):
        resized_img[i] = cv2.resize(img[i], (320, 320))
        
    return resized_img

def main():
    args = parse_args()
    raw_path = args.path
    out_path = args.outpath
    T = args.T
    
    print(f"Preprocessing N-Cars from: {raw_path}")
    print(f"Output directory: {out_path}")
    
    # Process both train and test splits
    # Map raw 'test' directory to YOLO 'val' split
    splits = [('train', 'train'), ('test', 'val')]
    classes = ['cars', 'background']
    
    os.makedirs(out_path, exist_ok=True)
    
    for raw_split, out_split in splits:
        split_dir = os.path.join(out_path, out_split)
        os.makedirs(split_dir, exist_ok=True)
        
        file_list_path = os.path.join(out_path, f"{out_split}.txt")
        file_paths_for_txt = []
        
        print(f"\nProcessing split: {raw_split} -> {out_split}")
        
        for cls in classes:
            class_raw_dir = os.path.join(raw_path, raw_split, cls)
            if not os.path.exists(class_raw_dir):
                print(f"Warning: Directory {class_raw_dir} does not exist. Skipping.")
                continue
                
            dat_files = [f for f in os.listdir(class_raw_dir) if f.endswith('.dat')]
            print(f"Found {len(dat_files)} files for class '{cls}'")
            
            pbar = tqdm(dat_files, desc=f"{cls}", unit="file")
            for filename in pbar:
                filepath = os.path.join(class_raw_dir, filename)
                base_name = os.path.splitext(filename)[0]
                
                # Output files
                out_npy_path = os.path.join(split_dir, f"{cls}_{base_name}.npy")
                out_txt_path = os.path.join(split_dir, f"{cls}_{base_name}.txt")
                
                try:
                    video = PSEELoader(filepath)
                    frames = create_frame_tensor(video, T)
                    np.save(out_npy_path, frames)
                    
                    # Create labels: class_id is 0 for car, empty for background
                    if cls == 'cars':
                        with open(out_txt_path, 'w') as lf:
                            lf.write("0 0.5 0.5 1.0 1.0\n")
                    else:
                        # Create empty label file for background
                        with open(out_txt_path, 'w') as lf:
                            pass
                            
                    file_paths_for_txt.append(out_npy_path)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        # Write file paths for this split
        with open(file_list_path, 'w') as f:
            for p in file_paths_for_txt:
                f.write(f"{p}\n")
                
    print("\nN-Cars Preprocessing Complete!")

if __name__ == "__main__":
    main()

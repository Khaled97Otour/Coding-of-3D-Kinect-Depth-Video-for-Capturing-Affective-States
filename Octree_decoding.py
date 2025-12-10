import os
import math
import time
import pickle
import glob
import numpy as np
import cv2
from pathlib import Path
from pyntcloud import PyntCloud
from octree_partitioning import partition_octree,departition_octree
import re

def crop_body_frame(image):
    """Crop image to the non-zero bounding box."""
    rows = np.any(image != 0, axis=1)
    cols = np.any(image != 0, axis=0)
    
    if not rows.any() or not cols.any():
        # image is empty
        return image, (0, 0, 0, 0)
    
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    
    y1, y2 = y_indices[0], y_indices[-1]
    x1, x2 = x_indices[0], x_indices[-1]
    
    cropped_image = image[y1:y2+1, x1:x2+1]
    return cropped_image, (y1, y2, x1, x2)

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(f'{f.__name__} function took {(time2 - time1) * 1000.0:.3f} ms')
        return ret
    return wrap
def extract_number(filename):
    match = re.search(r'\d',filename)
    if not match:
        return float('inf')
    return int(match.group())
def find_expo2_bbox(box_max):
    return math.ceil(math.log2(box_max))




def image_decoding(input_dir, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = glob.glob(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pc_path in filenames:
        # --- Load point cloud
        filename = os.path.basename(pc_path)[:-4]
        pc = PyntCloud.from_file(pc_path)
        points = pc.points.values.astype(np.float32)

        # --- Normalize (shift to origin)
        try:
            points = points - np.min(points, axis=0)
            points = np.round(points)
        except:
            continue
        # --- Find bounding box exponent
        level = find_expo2_bbox(np.max(points))
        box = math.pow(2, level)

        # --- Partition with octree
        blocks, binstr = timing(partition_octree)(
            points, [0, 0, 0], [box, box, box], level
        )


        # --- Save metadata (so decoder knows how to reconstruct)
        meta_file = output_dir / f"{filename}.pkl"
        data = {
            "blocks": blocks,
            "binstr": binstr,
            "level": level,
            "box": box,
        }
        with open(meta_file, "wb") as f:
            pickle.dump(data, f)
   
      																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
    # Decoding
    input_dir = Path(output_dir)
    matching_files = list(input_dir.glob("*.pkl"))
    point_cloud_tree = []
    counter = 0 

    for pc_path in range(len(matching_files)):
        pkl_file = output_dir / f'pc_{pc_path+14800}.pkl'
        if not pkl_file.exists():
            print(f"Skipping {pkl_file} (not found)")
            counter +=1
            continue  # skip to next iteration
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        blocks2 = data['blocks']
        binstr2 = data['binstr']
        level = data['level']
        box = data['box']
        print(f'done with the frame {pc_path}')
        decoded_blocks = departition_octree(blocks2, binstr2, [0,0,0], [box,box,box], level)

    
        output_img_dir = "/home/ga20lydi/frames/Octree"

        body_frame = f'/home/ga20lydi/frames/body_frame/bf_{counter+14800}.png'
        body_frame_image =cv2.imread(body_frame, cv2.IMREAD_UNCHANGED)
        try:
            coords = np.vstack(decoded_blocks)
        except:
            print('empty image')
            counter +=1
            continue

        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        X, Y = X.astype(int), Y.astype(int)
        width, height = X.max() + 1, Y.max() + 1

        image = np.zeros((height, width), dtype=np.uint16)
        try:
            Z_norm = (Z).astype(np.uint16)
        except:
            print('empty image')
            counter +=1
            continue
        image[Y, X] = Z_norm
        image = image.astype(np.float32)
        body_frame_image = body_frame_image.astype(np.float32)
        
        
        min1, max1 = image.min() , image.max()
        min2, max2 =  body_frame_image.min() , body_frame_image.max()
        
        image = (image-min1)/(max1 - min1) *(max2-min2)+ min2
        image = np.clip(image , min2, max2).astype(np.uint16)
        
        cropped_body, (y1, y2, x1, x2) = crop_body_frame(body_frame_image)
        top =y1 
        bottom = body_frame_image.shape[0] - (y2+1)
        left = x1 
        right = body_frame_image.shape[1] - (x2+1)
        image = cv2.resize(image, (cropped_body.shape[1], cropped_body.shape[0]))
        octree_image = np.pad(image, pad_width=((top, bottom), (left, right)), mode='constant', constant_values=0)
        octree_image = cv2.resize(octree_image, (body_frame_image.shape[1], body_frame_image.shape[0]))

        cv2.imwrite(f"{output_img_dir}/Octree{counter+14800}.png", octree_image)
        counter +=1
        
image_decoding(
    input_dir="/home/ga20lydi/frames/point_cloud/pc_*.ply",
    output_dir="/home/ga20lydi/frames/PC_Bin"
)

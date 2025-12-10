import os
import cv2
import numpy as np
import pandas as pd

# Paths
full_depth_with_background_dir = '/home/ga20lydi/frames/full_depth'
full_depth_body_dir = '/home/ga20lydi/frames/body_frame'
octree_dir = '/home/ga20lydi/frames/Octree'
raw_image = '/home/ga20lydi/frames/depth_raw'
# -------------------------------
# Helper function to read image
def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# -------------------------------
def crop_nonzero (image):
    
    cropped_image =image[np.ix_(rows,cols)]
    return cropped_image
# Read Full Depth images
method_1 = []
method_2 = []
method_3 = []
# uncompressed size in byte 
uncompressed_size = (640*480*16)//8
full_depth_compression_rate = []
full_depth_bitrate = []
body_frame_compression_rate = []
body_frame_bitrate = []
octree_compression_rate = []
octree_bitrate = []
for fname in range(len(os.listdir(raw_image))):
    number = fname + 14800
    full_depth = f'/home/ga20lydi/frames/full_depth/d_{number}.png'
    body_frame = f'/home/ga20lydi/frames/body_frame/bf_{number}.png'
    raw_image_dir =f'/home/ga20lydi/frames/depth_raw/dr_{number}.tiff'
    octree_image_dir = f'/home/ga20lydi/frames/Octree/Octree{number}.png'
    raw_frame = read_image(raw_image_dir)
    full_depth_frame = read_image(full_depth)
    body_frame_image = read_image(body_frame)
    octree_image = read_image(octree_image_dir)
    if os.path.exists(raw_image_dir):
        print(f"processing dr_{number}")
    else:
        continue
    if os.path.exists(full_depth):
        rmse = np.sqrt(np.mean((raw_frame - full_depth_frame) ** 2))
        psnr = 20 * np.log10(65535.0 / rmse)
        method_1.append(round(psnr, 2))
        uncompressed_size   =  os.path.getsize(raw_image_dir)
        size_bytes = os.path.getsize(full_depth)
        compression_rate = round((size_bytes/uncompressed_size)*100,2)
        bitrate = round((size_bytes*8*30)/1048576,2)
        full_depth_bitrate.append(bitrate)
        full_depth_compression_rate.append(compression_rate)
    else:
        method_1.append(None)        
        full_depth_bitrate.append(None)
        full_depth_compression_rate.append(None)
        print(f'done witht the frame number{fname}')
    # PSNR for Method 2 
    if os.path.exists(body_frame):
        rmse = np.sqrt(np.mean((raw_frame- body_frame_image) ** 2))
        psnr = 20 * np.log10(65535.0 / rmse)
        method_2.append(round(psnr, 2))
        uncompressed_size   =  os.path.getsize(raw_image_dir)
        size_bytes = os.path.getsize(body_frame)
        compression_rate = round((size_bytes/uncompressed_size)*100,2)
        bitrate = round((size_bytes*8*30)/1048576,2)
        body_frame_bitrate.append(bitrate)
        body_frame_compression_rate.append(compression_rate)
    else:
        method_2.append(None)        
        body_frame_bitrate.append(None)
        body_frame_compression_rate.append(None)
        print(f'done witht the frame number{number}')
    # PSNR for Method 3
    if os.path.exists(octree_image_dir):
        octree_image = read_image(octree_image_dir)
        rmse = np.sqrt(np.mean((raw_frame- octree_image) ** 2))
        psnr = 20 * np.log10(65535.0 / rmse)
        method_3.append(round(psnr, 2))
        size_bytes = os.path.getsize(octree_image_dir)
        uncompressed_size   =  os.path.getsize(raw_image_dir)
        compression_rate = round((size_bytes / uncompressed_size) * 100, 2)
        bitrate = round((size_bytes * 8 * 30) / 1048576, 2)
        octree_bitrate.append(bitrate)
        octree_compression_rate.append(compression_rate)
    else:
        method_3.append(None)        
        octree_bitrate.append(None)
        octree_compression_rate.append(None)
    print(f'done witht the frame number{number}')
    
print("Done with PSNR table")

# Save results
results = {
    'Method 1 PSNR': method_1,
    'method 1 compression rate %':full_depth_compression_rate,
    'method 1 bitrate': full_depth_bitrate,
    'Method 2 PSNR': method_2,
    'method 2 compression rate %':body_frame_compression_rate,
    'method 2 bitrate':body_frame_bitrate,
    'Method 3 PSNR': method_3,
    'method 3 compression rate %':octree_compression_rate,
    'method 3 bitrate':octree_bitrate

}

df = pd.DataFrame(results)
df.to_excel("psnr.xlsx", index=True)
print("Results saved to psnr.xlsx")

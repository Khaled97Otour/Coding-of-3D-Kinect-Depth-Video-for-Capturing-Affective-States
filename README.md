# Coding-of-3D-Kinect-Depth-Video-for-Capturing-Affective-States

# Azure Kinect MKV Processing Pipeline  
This project processes **Azure Kinect `.mkv` recordings** and extracts:

- Full depth images (aligned to the color camera)
- Body-only depth masks
- Body index maps
- Body-filtered depth frames
- Raw aligned depth (uncompressed TIFF)
- 3D body point clouds (binary PLY)

It uses the **Azure Kinect Sensor SDK**, **Azure Kinect Body Tracking SDK**, and **OpenCV** to process each frame inside the MKV file.

---

## ✨ Features

### ✔ Extracts from each frame:
- **Aligned depth frame** (`d_<index>.png`)
- **Body-only depth** (`bf_<index>.png`)
- **Raw depth aligned to color** (`dr_<index>.tiff`)
- **Binary PLY point cloud** (`pc_<index>.ply`)

### ✔ Performs:
- Depth → Color alignment
- Body index map processing
- 3D back-projection using calibration
- Body-only point cloud extraction

---

## 📂 Output Directory Structure

The code writes files to the following directories:



# Octree-Based Point Cloud Compression & Depth Reconstruction

This script performs **octree compression and decompression** on point clouds extracted from Azure Kinect MKV recordings.  
It encodes each point cloud (`.ply` from Azure Kinect) into an octree-based binary representation, stores the metadata in `.pkl` files, and then **reconstructs a depth image** from the decoded points.

It reconstructs a depth image that matches the original body depth frame geometry.

---

## ✨ Features

### ✔ Encoding Stage
For each input `.ply` point cloud:
- Loads the point cloud using **PyntCloud**
- Normalizes point positions (translates to origin)
- Finds the bounding box and computes the smallest `2^n` cube
- Encodes the cloud with **octree partitioning**
- Saves metadata in `.pkl`:
  - `blocks`
  - `binary string`
  - `octree level`
  - `bounding cube size`

### ✔ Decoding Stage
For each `.pkl`:
- Runs octree **departitioning** to reconstruct points  
- Rebuilds a depth image using `(X, Y, Z)`  
- Normalizes reconstructed depth to match original body-frame depth range  
- Crops and pads output so it aligns with body-frame image  
- Saves reconstructed PNG as:

# Octree-Based Point Cloud Encoding & Depth Reconstruction

This script performs **octree-based partitioning and reconstruction** for point clouds.  
It encodes each `.ply` point cloud into an octree representation, stores the metadata in `.pkl` files, and later **reconstructs a depth image** from the decoded points.

The final reconstructed depth image is resized, normalized, and aligned to match the original body-frame depth geometry.

---

## ✨ Features

### ✔ Octree Encoding  
For each input `.ply` point cloud:

- Loads the point cloud using **PyntCloud**  
- Normalizes and rounds point coordinates  
- Computes the smallest power-of-two bounding cube (`2^level`)  
- Applies **vectorized octree partitioning**  
- Stores metadata in `.pkl` files:
  - `blocks`  
  - `binstr` (8-bit child-occupancy mask)  
  - `octree level`  
  - `bounding cube size`  

---

### ✔ Octree Decoding  
For each `.pkl` file:

- Restores octree structure and decodes all blocks  
- Reconstructs the full set of `(X, Y, Z)` points  
- Converts points into a depth image  
- Normalizes reconstructed depth to the body-frame depth range  
- Crops and pads the image to match the original frame layout  
- Saves reconstructed PNGs with consistent naming  

---

## 🔧 Octree Functions Included

### compute_all_bboxes  
Computes the 8 child bounding boxes for a parent cube.

### compute_new_bbox  
Returns the bounding box for a specific octree child (0–7).

### split_octree  
Vectorized point-cloud splitting based on midpoint thresholding.

### partition_octree  
Recursive octree construction producing `blocks` + `binstr`.

### departition_octree  
Rebuilds the octree traversal and returns decoded blocks.

---

## 📥 Input Format

- Point clouds:  
  `pc_*.ply`

- Original depth frames (used for alignment):  
  `bf_XXXXX.png`

---

## 📤 Output Format

- Octree metadata:  
  `PC_Bin/pc_XXXXX.pkl`

- Reconstructed depth images:  
  `Octree/OctreeXXXXX.png`

---

## ▶ Running the Pipeline

```python
image_decoding(
    input_dir="/home/.../frames/point_cloud/pc_*.ply",
    output_dir="/home/.../frames/PC_Bin"
)

```
# Pose Extraction System for Azure Kinect Recordings

This system processes **Azure Kinect MKV recordings** to extract 3D human pose data.  
It combines depth information and AI models to generate accurate spatial positioning of key joints.  
The extracted poses are saved in a structured **JSON format**, aligned with the original depth frames.

---

## ✨ Features

### ✔ Virtual Display Support
- Uses **Xvfb** to enable headless operation on servers without GUI  
- Automatically starts/stops a virtual display  
- Avoids conflicts with existing displays

### ✔ Pose Estimation
For each frame in the MKV recording:

- Extracts **color and depth images** from Azure Kinect  
- Applies **RT-DETR** for person detection  
- Uses **ViTPose** for high-accuracy 2D keypoint estimation  
- Converts 2D keypoints to **3D coordinates** using depth information and camera intrinsics  
- Enforces **anatomical constraints** on depth for realistic joint positioning  
- Stores:
  - 2D coordinates (`x_2d`, `y_2d`)  
  - 3D coordinates (`x_3d`, `y_3d`, `z_3d`)  
  - Confidence scores  
  - Joint labels  

### ✔ Output
- Saves results to a **JSON file** containing:
  - Metadata (source file, camera parameters, processing date)  
  - Frame-wise pose data for each detected person  
- Automatically handles frame skipping if data is missing or invalid  

---

## 📥 Input Format

- Azure Kinect MKV recordings:  
  `recording.mkv`

---

## 📤 Output Format

- JSON file containing extracted poses:  
  `poses.json`

- Example structure:
```json
{
  "metadata": {
    "source_file": "recording.mkv",
    "processing_date": "2025-12-10 15:30:00",
    "camera_params": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
    "smoothing_params": "none"
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": "",
      "poses": [
        {"x_2d": ..., "y_2d": ..., "x_3d": ..., "y_3d": ..., "z_3d": ..., "joint": 0, "confidence": ...},
        ...
      ]
    },
    ...
  ]
}


# Pose Extraction System for Kinect Azure Recordings
# This system processes Azure Kinect MKV recordings to extract 3D human pose data
# using state-of-the-art AI models and depth information for accurate spatial positioning.

import torch
import numpy as np
import cv2
import json
import os
import time
import pykinect_azure as pykinect
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from PIL import Image
import subprocess
import atexit
import random

class VirtualDisplay:
    """
    Manages a virtual X11 display using Xvfb for headless operation.
    This is essential for running GUI-dependent applications on servers without displays.
    """
    def __init__(self, xvfb_path='/home/hpc/iwso/iwso162h/t2_sandbox/usr/bin/Xvfb', width=1280, height=1024, color_depth=24):
        """
        Initialize virtual display configuration.
        
        Args:
            xvfb_path: Path to the Xvfb executable
            width: Display width in pixels
            height: Display height in pixels 
            color_depth: Color depth in bits
        """
        self.xvfb_path = xvfb_path
        self.width = width
        self.height = height
        self.color_depth = color_depth
        # Generate random display number to avoid conflicts with existing displays
        self.display_num = random.randint(1000, 9999)
        self.display = f":{self.display_num}"
        self.process = None

    def start(self):
        """
        Start the virtual display and set the DISPLAY environment variable.
        The display will automatically shut down when the program exits.
        """
        cmd = [
            self.xvfb_path,
            self.display,
            "-screen", "0",
            f"{self.width}x{self.height}x{self.color_depth}"
        ]
        self.process = subprocess.Popen(cmd)
        time.sleep(0.5)  # Give Xvfb time to start up
        os.environ["DISPLAY"] = self.display
        print(f"Started virtual display at DISPLAY={self.display}")
        # Ensure cleanup happens even if program crashes
        atexit.register(self.stop)
        return self.display

    def stop(self):
        """Clean shutdown of the virtual display."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print(f"Stopped virtual display at DISPLAY={self.display}")

class KinectDepthPoseEstimator:
    """
    Advanced pose estimation system that combines Azure Kinect depth data with AI models
    to extract precise 3D human pose information from recorded sessions.
    """
    
    def __init__(self, mkv_path, output_json):
        """
        Initialize the pose estimation pipeline with camera calibration and AI models.
        
        Args:
            mkv_path: Path to the Azure Kinect MKV recording file
            output_json: Path where the extracted pose data will be saved
        """
        self.mkv_path = mkv_path

        # Initialize Azure Kinect SDK with custom library paths
        # This setup is required for the specific HPC environment
        pykinect.initialize_libraries(
            module_k4a_path='/home/ga20lydi/k4a/sdk/usr/lib/x86_64-linux-gnu/libk4a.so',
            module_k4abt_path='/home/ga20lydi/k4a/sdk/usr/lib/libk4abt.so',
            track_body=True
        )

        self.output_json = output_json
        # Use GPU acceleration if available for faster processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using" + self.device)

        # Set up Azure Kinect playback and extract camera parameters
        self.playback = pykinect.start_playback(self.mkv_path)
        self.playback_config = self.playback.get_record_configuration()
        self.playback_calibration = self.playback.get_calibration()
        
        # Extract camera intrinsic parameters for 3D reconstruction
        camera_intrinsics = self.playback_calibration.color_params
        self.fx = camera_intrinsics.fx  # Focal length in x direction
        self.fy = camera_intrinsics.fy  # Focal length in y direction
        self.cx = camera_intrinsics.cx  # Principal point x coordinate
        self.cy = camera_intrinsics.cy  # Principal point y coordinate

        # Load pre-trained AI models for person detection and pose estimation
        # RT-DETR: Real-time object detection transformer for finding people
        self.person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(self.device)
        
        # ViTPose: Vision transformer for high-accuracy human pose estimation
        self.pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge").to(self.device)

        # Pose validation parameters
        self.nose_depth = 0  # Reference depth for anatomical constraints
        self.depth_threshold = 0.1
        
        # Initialize output data structure with metadata
        self.pose_data = {
            "metadata": {
                "source_file": mkv_path,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "camera_params": {
                    "fx": float(self.fx),
                    "fy": float(self.fy),
                    "cx": float(self.cx),
                    "cy": float(self.cy)
                },
                "smoothing_params": "none"
            },
            "frames": []
        }

    def depth_to_3d(self, x, y, depth_value):
        """
        Convert 2D pixel coordinates and depth to 3D world coordinates.
        
        Uses the pinhole camera model with intrinsic parameters to perform
        the perspective projection inverse transformation.
        
        Args:
            x, y: Pixel coordinates in the image
            depth_value: Depth value in millimeters from the depth sensor
            
        Returns:
            tuple: (x_3d, y_3d, z_3d) coordinates in meters, or (None, None, None) if invalid
        """
        if depth_value <= 0:
            return None, None, None

        # Convert depth from millimeters to meters
        z = depth_value / 1000.0

        # Apply inverse perspective projection using camera intrinsics
        x_3d = (x - self.cx) * z / self.fx
        y_3d = (y - self.cy) * z / self.fy
        return x_3d, y_3d, z

    def process_frame(self, color_image, depth_image):
        """
        Extract 3D pose data from a single frame using AI models and depth information.
        
        This method combines computer vision and depth sensing to create accurate
        3D pose estimates with anatomical constraints for realistic results.
        
        Args:
            color_image: RGB image from the Kinect color camera
            depth_image: Corresponding depth map from the Kinect depth sensor
            
        Returns:
            list: Array of pose keypoints with 2D, 3D coordinates and confidence scores
        """
        # Convert BGR to RGB for the AI model (OpenCV uses BGR by default)
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        # Apply bilateral filtering to reduce depth noise while preserving edges
        depth_image_filtered = cv2.bilateralFilter(depth_image.astype(np.float32), d=5, sigmaColor=75, sigmaSpace=75)

        # Prepare inputs for the pose estimation model
        # Using full image bounding box for whole-body pose detection
        dataset_index = torch.zeros((1, 1), dtype=torch.int64).to(self.device)
        pose_inputs = self.pose_processor(pil_image, boxes=[[[0, 0, 1080, 1920]]], return_tensors="pt").to(self.device)
        pose_inputs["dataset_index"] = dataset_index

        # Run pose estimation inference
        with torch.no_grad():
            pose_outputs = self.pose_model(pixel_values=pose_inputs["pixel_values"], dataset_index=pose_inputs["dataset_index"])

        # Post-process the model outputs to get keypoint coordinates
        pose_results = self.pose_processor.post_process_pose_estimation(pose_outputs, boxes=[[[0, 0, 1080, 1920]]], threshold=0.5)
        image_pose_result = pose_results[0][0]
        
        # Extract keypoints, confidence scores, and joint labels
        keypoints, scores, labels = list(image_pose_result['keypoints']), list(image_pose_result['scores']), list(image_pose_result['labels'])

        # Define confidence thresholds for different joints (unused but kept for future use)
        joint_thresholds = {0: 0.10, 1: 0.15, 2: 0.15, 3: 0.20, 4: 0.20}
        person_pose = []

        # Process each detected keypoint (focusing on upper body - first 11 joints)
        for kp, conf, lb in zip(keypoints[:11], scores[:11], labels[:11]):
            try:
                # Get integer pixel coordinates
                x, y = map(int, kp[:2])
                h, w = depth_image_filtered.shape
                
                # Sample depth from a small neighborhood to reduce noise
                grid_size = 6
                half_grid = grid_size // 2

                # Ensure sampling window stays within image bounds
                x_min, x_max = max(x - half_grid, 0), min(x + half_grid + 1, w)
                y_min, y_max = max(y - half_grid, 0), min(y + half_grid + 1, h)

                # Extract depth patch and use median for robustness
                depth_patch = depth_image_filtered[y_min:y_max, x_min:x_max]
                valid_depths = depth_patch[depth_patch > 0]
                depth = np.median(valid_depths) if valid_depths.size > 0 else 0
                lb = int(lb)

                # Convert to 3D coordinates
                x_3d, y_3d, z = self.depth_to_3d(x, y, depth)
                if z is None:
                    continue
                # Apply anatomical constraints based on joint type
                # This ensures realistic pose geometry relative to the nose position
                if lb == 0:  # Nose - reference point
                    self.nose_depth = z
                    max_z = z
                    min_z = z
                elif lb in (1,2):  # Eyes - close to nose
                    max_z = self.nose_depth + 0.07
                    min_z = self.nose_depth + 0.05
                elif lb in (3,4):  # Ears - slightly behind eyes
                    max_z = self.nose_depth + 0.1
                    min_z = self.nose_depth + 0.07
                elif lb in (5,6):  # Shoulders - wider depth range
                    max_z = self.nose_depth + 0.1
                    min_z = self.nose_depth + 0.05
                elif lb in (7,8):  # Elbows - more movement freedom
                    max_z = self.nose_depth + 0.15
                    min_z = self.nose_depth - 0.15
                elif lb in (9,10):  # Wrists - maximum movement range
                    max_z = self.nose_depth + 0.2
                    min_z = self.nose_depth - 0.2
                    
                # Enforce depth constraints for anatomical realism
                if z > max_z and lb != 0:
                    z = max_z
                elif z < min_z and lb != 0:
                    z = min_z 

                # Store the complete pose keypoint data
                person_pose.append({
                    "x_2d": x,
                    "y_2d": y,
                    "x_3d": x_3d,
                    "y_3d": y_3d,
                    "z_3d": z,
                    "joint": lb,
                    "confidence": float(conf)
                })
            except Exception as e:
                print('process_frame')
                print(e)
                continue

        return person_pose

    def process_recording(self):
        """
        Process the entire MKV recording frame by frame and save results to JSON.
        
        This is the main processing loop that handles the full video sequence,
        extracting pose data from each frame and compiling it into a structured output.
        """
        try:
            frame_count = 0
            print("Starting pose extraction from recording...")
            
            while True:
                try:
                    # Get the next frame from the recording
                    ret, capture = self.playback.update()
                    if not ret:
                        print("Reached end of recording")
                        break
                        
                    # Extract color and depth images
                    try:
                        ret_color, color_image = capture.get_color_image()
                        ret_depth, depth_image = capture.get_transformed_depth_image()
                    except:
                        print(f"Skipping frame {frame_count} due to error: {e}")
                        continue

                    
                    # Skip frames with missing data
                    if not ret_color or not ret_depth or color_image is None:
                        continue
                    
                    # Process the current frame to extract pose data
                    frame_poses = self.process_frame(color_image, depth_image)
                    
                    # Store frame data with metadata
                    self.pose_data["frames"].append({
                        "frame_number": frame_count,
                        "timestamp": '',  # Could be populated with actual timestamps if needed
                        "poses": frame_poses
                    })
                    
                    # Optional: Limit processing for testing (uncomment if needed)
                    #if frame_count == 30 * 60 * 5:  # Process 5 minutes at 30fps
                        #break
                        
                    if frame_poses:  # only increment if some pose was found
                        frame_count += 1
                    
                    # Progress indicator for long recordings
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames...")
                        
                except Exception as e:
                    print('Error processing frame:', frame_count)
                    print(e)
                    continue
                    
        finally:
            # Ensure proper cleanup of resources
            self.playback.close()
            
            # Create output directory if it doesn't exist
            directory = os.path.dirname(self.output_json)
            os.makedirs(directory, exist_ok=True)
            
            # Save the complete pose data to JSON file
            with open(self.output_json, 'w') as f:
                json.dump(self.pose_data, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
            
            print(f"Processing complete! Extracted poses from {frame_count} frames.")
            print(f"Results saved to: {self.output_json}")

if __name__ == "__main__":
    import sys
    
    # Command line argument validation
    if len(sys.argv) != 3:
        print("Usage: python extractor.py <input.mkv> <output.json>")
        print("Example: python extractor.py recording.mkv poses.json")
        sys.exit(1)

    # Set up virtual display for headless operation
    print("Initializing virtual display for headless processing...")
    #display = VirtualDisplay(xvfb_path='/home/hpc/iwso/iwso162h/t2_sandbox/usr/bin/Xvfb')
    #display.start()

    # Create and run the pose estimation system
    print("Initializing pose estimation system...")
    estimator = KinectDepthPoseEstimator(sys.argv[1], sys.argv[2])
    estimator.process_recording()

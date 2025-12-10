#include <k4a/k4a.h>
#include <k4abt.h>
#include <k4arecord/playback.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// ------------------ Output directories ------------------
const std::string BODY_DIR   = "/home/ga20lydi/frames/body_frame/";
const std::string DEPTH_DIR  = "/home/ga20lydi/frames/full_depth/";
const std::string RAW_DIR    = "/home/ga20lydi/frames/depth_raw/";
const std::string PCLOUD_DIR = "/home/ga20lydi/frames/point_cloud/";

// ------------------ PLY binary writer (XYZ only) ------------------
void write_point_cloud_binary(const std::string& filename,
                              const std::vector<cv::Vec3f>& points)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to write PLY: " << filename << std::endl;
        return;
    }

    size_t count = points.size();
    ofs << "ply\nformat binary_little_endian 1.0\n";
    ofs << "element vertex " << count << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < count; ++i) {
        ofs.write(reinterpret_cast<const char*>(&points[i][0]), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&points[i][1]), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&points[i][2]), sizeof(float));
    }
}

// ---------------- Save body point cloud (PLY) ----------------
void SaveBodyPointCloud(const cv::Mat& bodyDepth,
                        const k4a_calibration_t& calibration,
                        int frameIndex)
{
    std::vector<cv::Vec3f> points;
    int width  = bodyDepth.cols;
    int height = bodyDepth.rows;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint16_t depthVal = bodyDepth.at<uint16_t>(y, x);
            if (depthVal > 0) {
                k4a_float2_t pixel = { (float)x, (float)y };
                k4a_float3_t ray;
                int valid = 0;

                if (k4a_calibration_2d_to_3d(&calibration,
                                             &pixel,
                                             depthVal,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             K4A_CALIBRATION_TYPE_COLOR,
                                             &ray, &valid) == K4A_RESULT_SUCCEEDED && valid)
                {
                    points.emplace_back(ray.xyz.x, ray.xyz.y, ray.xyz.z);
                }
            }
        }
    }

    std::string filename = PCLOUD_DIR + "pc_" + std::to_string(frameIndex) + ".ply";
    write_point_cloud_binary(filename, points);
}

// ------------------ Frame processing ------------------
void ProcessFrame(k4abt_frame_t bodyFrame,
                  int frameIndex,
                  const k4a_calibration_t& calibration,
                  k4a_transformation_t transformation)
{
    k4a_capture_t capture = k4abt_frame_get_capture(bodyFrame);
    k4a_image_t depthImg = k4a_capture_get_depth_image(capture);
    k4a_image_t indexMap = k4abt_frame_get_body_index_map(bodyFrame);

    if (!depthImg || !indexMap) {
        if (depthImg) k4a_image_release(depthImg);
        if (indexMap) k4a_image_release(indexMap);
        k4a_capture_release(capture);
        return;
    }

    int colorWidth  = calibration.color_camera_calibration.resolution_width;
    int colorHeight = calibration.color_camera_calibration.resolution_height;

    // Aligned full depth
    k4a_image_t alignedDepth = nullptr;
    k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, colorWidth, colorHeight,
                     colorWidth * sizeof(uint16_t), &alignedDepth);

    // Aligned index (body mask)
    k4a_image_t alignedIndex = nullptr;
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM8, colorWidth, colorHeight,
                     colorWidth * sizeof(uint8_t), &alignedIndex);

    if (k4a_transformation_depth_image_to_color_camera_custom(
            transformation, depthImg, indexMap,
            alignedDepth, alignedIndex,
            K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST, 255) != K4A_RESULT_SUCCEEDED)
    {
        std::cerr << "Transformation failed on frame " << frameIndex << std::endl;
        k4a_image_release(alignedDepth);
        k4a_image_release(alignedIndex);
        k4a_image_release(depthImg);
        k4a_image_release(indexMap);
        k4a_capture_release(capture);
        return;
    }

    // Convert to OpenCV Mats
    cv::Mat depthMat(colorHeight, colorWidth, CV_16UC1, k4a_image_get_buffer(alignedDepth));
    cv::Mat indexMat(colorHeight, colorWidth, CV_8UC1, k4a_image_get_buffer(alignedIndex));

    // ---------------- Save full aligned depth ----------------
    {
        std::string depthPath = DEPTH_DIR + "d_" + std::to_string(frameIndex) + ".png";
        cv::imwrite(depthPath, depthMat);
    }

    // ---------------- Save body-only aligned depth ----------------
    cv::Mat bodyDepth(colorHeight, colorWidth, CV_16UC1, cv::Scalar(0));
    {
        for (int y = 0; y < colorHeight; ++y) {
            for (int x = 0; x < colorWidth; ++x) {
                if (indexMat.at<uint8_t>(y, x) != 255) { // body pixel
                    bodyDepth.at<uint16_t>(y, x) = depthMat.at<uint16_t>(y, x);
                }
            }
        }
        std::string bodyPath = BODY_DIR + "bf_" + std::to_string(frameIndex) + ".png";
        cv::imwrite(bodyPath, bodyDepth);

        // Save body point cloud
       SaveBodyPointCloud(bodyDepth, calibration, frameIndex);
    }

    // ---------------- Save raw depth aligned to color (uncompressed) ----------------
    {
        k4a_image_t alignedRaw = nullptr;
        k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, colorWidth, colorHeight,
                         colorWidth * sizeof(uint16_t), &alignedRaw);

        if (k4a_transformation_depth_image_to_color_camera(
                transformation, depthImg, alignedRaw) == K4A_RESULT_SUCCEEDED)
        {
            cv::Mat rawAligned(colorHeight, colorWidth, CV_16UC1,
                               k4a_image_get_buffer(alignedRaw));
            std::string rawPath = RAW_DIR + "dr_" + std::to_string(frameIndex) + ".tiff";
            cv::imwrite(rawPath, rawAligned, {cv::IMWRITE_TIFF_COMPRESSION, 1});
        }
        k4a_image_release(alignedRaw);
    }

    // Cleanup
    k4a_image_release(alignedDepth);
    k4a_image_release(alignedIndex);
    k4a_image_release(depthImg);
    k4a_image_release(indexMap);
    k4a_capture_release(capture);
}

// ------------------ Main ------------------
int main()
{
    const char* file_path = "/home/ga20lydi/007_t2_20230309.mkv";
    k4a_playback_t playback = nullptr;

    if (k4a_playback_open(file_path, &playback) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to open MKV file." << std::endl;
        return -1;
    }

    k4a_calibration_t calibration;
    k4a_playback_get_calibration(playback, &calibration);

    k4abt_tracker_t tracker = nullptr;
    k4abt_tracker_configuration_t config = K4ABT_TRACKER_CONFIG_DEFAULT;
    if (k4abt_tracker_create(&calibration, config, &tracker) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to create body tracker." << std::endl;
        return -1;
    }

    k4a_transformation_t transformation = k4a_transformation_create(&calibration);

    int frameIndex = 0;
    k4a_capture_t capture = nullptr;

    while (k4a_playback_get_next_capture(playback, &capture) == K4A_STREAM_RESULT_SUCCEEDED) {
        if (frameIndex < 14800)
        {
            frameIndex++;
            k4a_capture_release(capture);
            continue;
        }

        // Stop after 21794
        if (frameIndex > 18700)
        {
            k4a_capture_release(capture);
            break;
        }
        if (k4abt_tracker_enqueue_capture(tracker, capture, K4A_WAIT_INFINITE) == K4A_WAIT_RESULT_SUCCEEDED) {
            k4abt_frame_t bodyFrame = nullptr;
            if (k4abt_tracker_pop_result(tracker, &bodyFrame, K4A_WAIT_INFINITE) == K4A_WAIT_RESULT_SUCCEEDED) {
                ProcessFrame(bodyFrame, frameIndex++, calibration, transformation);
                k4abt_frame_release(bodyFrame);
            }
        }
        k4a_capture_release(capture);
    }

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    k4a_transformation_destroy(transformation);
    k4a_playback_close(playback);

    std::cout << "Processing complete." << std::endl;
    return 0;
}


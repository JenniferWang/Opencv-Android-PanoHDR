#include <opencv2/opencv.hpp>
#include "NativeLogging.h"

using namespace std;
using namespace cv;

// help function
static void get_poly_extremes(const vector<Point2f>& corners, vector<float>& outputs) {
    outputs.clear();
    if (corners.size() == 0) {
        cout << "input size is 0" << endl;
        return;
    }
    outputs.push_back(corners[0].y); // output[0] = min y
    outputs.push_back(corners[0].y); // output[1] = max y
    outputs.push_back(corners[0].x); // output[2] = min x
    outputs.push_back(corners[0].x); // output[3] = max x
    
    for (size_t i = 1; i < corners.size(); i++) {
        outputs[0] = corners[i].y < outputs[0] ? corners[i].y : outputs[0];
        outputs[1] = corners[i].y > outputs[1] ? corners[i].y : outputs[1];
        outputs[2] = corners[i].x < outputs[2] ? corners[i].x : outputs[2];
        outputs[3] = corners[i].x > outputs[3] ? corners[i].x : outputs[3];
    }
}

static const char* TAG = "Panorama";

static void read_input_images(const vector<string>& inputImagePaths, vector<Mat>& images) {
    for (int i = 0; i < inputImagePaths.size(); i ++) {
        Mat img = imread(inputImagePaths[i]);
        if (img.data != NULL) {
            images.push_back(img);
        }
    }
}

static void find_correspondences_between_two_images(const Mat& left, const Mat& right,vector<vector<Point2f>>& matchesBetween2Imgs) {
    if ((!left.data) || (!right.data)) {
        cout << "Empty image" << endl;
        return;
    }
    std::vector<KeyPoint> keyPoints1, keyPoints2;
    Mat descriptor1, descriptor2;
    
    OrbFeatureDetector detector(5000);
    detector.detect(left, keyPoints1);
    detector.detect(right, keyPoints2);
    
    OrbDescriptorExtractor extractor;
    extractor.compute(left, keyPoints1, descriptor1);
    extractor.compute(right, keyPoints2, descriptor2);
    
    // Match descriptor vectors
    BFMatcher matcher;
    std::vector<vector< DMatch >> allmatches;
    matcher.knnMatch(descriptor2, descriptor1, allmatches, 2);
    
    // Feature distance ratio test
    std::vector< DMatch > good_matches;
    for (int i = 0; i < allmatches.size(); i ++) {
        float rejectRatio = 0.8;
        if (allmatches[i][0].distance / allmatches[i][1].distance > rejectRatio)
            continue;
        good_matches.push_back(allmatches[i][0]);
    }
    std::vector<Point2f> good_keyPoints1, good_keyPoints2;
    for (int i = 0; i < good_matches.size(); i ++) {
        good_keyPoints1.push_back(keyPoints1[good_matches[i].trainIdx].pt);
        good_keyPoints2.push_back(keyPoints2[good_matches[i].queryIdx].pt);
    }
    matchesBetween2Imgs.clear();
    matchesBetween2Imgs.push_back(good_keyPoints1);
    matchesBetween2Imgs.push_back(good_keyPoints2);
}

static void find_correspondences(const vector<Mat>& images, vector<vector<Point2f>>& matches) {
    size_t num_imgs = images.size();
    if (num_imgs < 2) return;
    find_correspondences_between_two_images(images[0], images[1], matches);
}

static void compute_homography_fm_right_to_left(const vector<vector<Point2f>>& matches, Matx33f& H) {
    if (matches.size()!= 2) {
        return;
    }
    H = findHomography( matches[1], matches[0], CV_RANSAC );
}

static void compute_homography_fm_left_to_right(const vector<vector<Point2f>>& matches, Matx33f& H) {
    if (matches.size()!= 2) {
        return;
    }
    H = findHomography( matches[0], matches[1], CV_RANSAC );
}

static void compute_homography(const vector<vector<Point2f>>& matches, Matx33f& H, bool mapLeftToRight) {
    if (mapLeftToRight)
        compute_homography_fm_left_to_right( matches, H );
    else
        compute_homography_fm_right_to_left( matches, H );
}

static void compute_mask_C1(int maskWidth, Size totalSize, Mat& mask, bool leftOn) {
    if (totalSize.width < maskWidth) {
        return;
    }
    mask = Mat::ones(totalSize.height, maskWidth, CV_32F); 
    if (leftOn) {
        copyMakeBorder(mask, mask, 0, 0, 0, totalSize.width - maskWidth, BORDER_CONSTANT, 0);
    }
    else {
        copyMakeBorder(mask, mask, 0, 0, totalSize.width - maskWidth, 0, BORDER_CONSTANT, 0);
        mask = 1 - mask;
    }
}

static Rect warp_right_to_left(const Mat& leftImg, const Mat& rightImg,
                               const Matx33f& H, Mat& paddedLeftImage, Mat& warppedRightImage) {
    if ((!leftImg.data) || (!rightImg.data)) {
        return Rect();
    }
    // Get the corners from the right image, Figure out the dimensions of the warped right image.
    std::vector<Point2f> rImg_corners(4);
    rImg_corners[0] = cvPoint( 0, 0 );
    rImg_corners[1] = cvPoint( 0, rightImg.rows );
    rImg_corners[2] = cvPoint( rightImg.cols, rightImg.rows );
    rImg_corners[3] = cvPoint( rightImg.cols, 0);
    std::vector<Point2f> warpped_corners(4);
    perspectiveTransform( rImg_corners, warpped_corners, H);
    
    // Pad/Extend the left image so that it matches the dimensions of the right image.
    vector<float> corner_extremes;
    get_poly_extremes(warpped_corners, corner_extremes);
    int top = corner_extremes[0] < 0 ? std::ceil(-corner_extremes[0]) : 0;
    int bottom = (corner_extremes[1] - leftImg.rows) > 0 ? std::ceil(corner_extremes[1] - leftImg.rows) : 0;
    int left = corner_extremes[2] < 0 ? std::ceil(-corner_extremes[2]) : 0;
    int right = (corner_extremes[3] - leftImg.cols) > 0 ? std::ceil(corner_extremes[3] - leftImg.cols) : 0;
    copyMakeBorder(leftImg, paddedLeftImage, top, bottom, left, right, BORDER_CONSTANT, 0);
    
    // Warp the right image.
    Rect original_box = Rect(0, 0, leftImg.cols, leftImg.rows);
    Rect warpped_box = boundingRect(warpped_corners);
    Rect merged_box = warpped_box | original_box;
    Size warpped_size = Size(merged_box.size());
    
    Mat translationMat = (Mat_<float>(3,3) << 1, 0, left,
                                              0, 1, top,
                                              0, 0, 1);
    Mat H_(H);
    warpPerspective(rightImg, warppedRightImage, translationMat * H_, Size(warpped_size));
    return warpped_box;
}

static Rect warp_left_to_right(const Mat& leftImg, const Mat& rightImg,
                               const Matx33f& H, Mat& warppedLeftImage, Mat& paddedRightImage) {
    return warp_right_to_left(rightImg, leftImg, H, paddedRightImage, warppedLeftImage);
}

static Rect warp(const Mat& leftImg, const Mat& rightImg, const Matx33f& H,
                 Mat& outputLeft, Mat& outputRight, bool leftToRight) {
    if (leftToRight)
        return warp_left_to_right(leftImg, rightImg, H, outputLeft, outputRight);
    else
        return warp_right_to_left(leftImg, rightImg, H, outputLeft, outputRight);
}

static void upSample(bool isHeightOdd, bool isWidthOdd, Mat& lowResolution, Mat& highResolution) {
    int borderWidth = 1;
    Mat paddedMat;
    copyMakeBorder(lowResolution, paddedMat, borderWidth, borderWidth, borderWidth, borderWidth, BORDER_REPLICATE);
    pyrUp(paddedMat, paddedMat);
    Size highResSize = paddedMat.size();
    highResolution = paddedMat(Range(2, highResSize.height - 2 - isHeightOdd),
                               Range(2, highResSize.width - 2 - isWidthOdd));
}

static void buildLaplacianPyramid(const vector<Mat>& gaussianPyr, vector<Mat>& laplacianPyr ){
    laplacianPyr.clear();
    for (int i = 0; i < gaussianPyr.size() - 1; i ++) {
        Size imgSize = gaussianPyr[i].size();
        Mat highRes;
        upSample(imgSize.height % 2, imgSize.width % 2, (Mat &)gaussianPyr[i + 1], highRes);
        laplacianPyr.push_back(gaussianPyr[i] - highRes);
    }
    laplacianPyr.push_back(gaussianPyr.back().clone());
}

static void reconstructLaplacianPyramid(vector<Mat>& laplacianPyr, Mat& output) {
    if (laplacianPyr.size() < 1) return;
    Mat higherRes;
    for (int i = (int)laplacianPyr.size() - 1; i > 0; i --) {
        Size higherResSize = laplacianPyr[i - 1].size();
        upSample(higherResSize.height % 2, higherResSize.width % 2, laplacianPyr[i], higherRes);
        laplacianPyr[i - 1] += higherRes;
    }
    convertScaleAbs(laplacianPyr[0], output);
}

static void blend_two_images(const Mat& leftImg, const Mat& rightImg, const Mat& mask, Mat& outputImage) {
    if ((!leftImg.data) || (!rightImg.data)) {
        return;
    }
    
    int maxPyrIndex = 6;
    // build pyramids for two images
    Mat img1, img2;
    vector<Mat> gaussianPyr1, laplacianPyr1, gaussianPyr2, laplacianPyr2;
    
    leftImg.convertTo(img1, CV_32F);
    rightImg.convertTo(img2, CV_32F);
    
    // build masks and Gaussian pyramids for masks
    vector<Mat> gaussianPyr_mask;
    buildPyramid(mask, gaussianPyr_mask, maxPyrIndex);
    buildPyramid(img1, gaussianPyr1, maxPyrIndex);
    buildPyramid(img2, gaussianPyr2, maxPyrIndex);
    buildLaplacianPyramid(gaussianPyr1, laplacianPyr1);
    buildLaplacianPyramid(gaussianPyr2, laplacianPyr2);

    // blend pyramids
    vector<Mat> cumulatedPyr;
    for (int k = 0; k < laplacianPyr1.size(); k ++) {
        vector<Mat> splitted1, splitted2, cumulated;
        Mat mergedOneLayer;
        split(laplacianPyr1[k], splitted1);
        split(laplacianPyr2[k], splitted2);
        for (int c = 0; c < splitted1.size(); c++) {
            cumulated.push_back(splitted2[c].mul(1- gaussianPyr_mask[k]) +
                                splitted1[c].mul(gaussianPyr_mask[k]));
        }
        merge(cumulated, mergedOneLayer);
        cumulatedPyr.push_back(mergedOneLayer);
    }
    reconstructLaplacianPyramid(cumulatedPyr, outputImage);
    convertScaleAbs( outputImage, outputImage );
}

// This is the main entry point for the panorama stitching method.
// It accepts two arguments:
// 1. inputImagePaths: A vector of absolute paths of the images to stitch together.
// 2. outputImagePath: The path where the output image should be saved.
bool create_panorama(const vector<string>& inputImagePaths, const string& outputImagePath) {
    //Load the images.
    vector<Mat> images;
    read_input_images(inputImagePaths, images);
    if(images.size()!=inputImagePaths.size()) {
       LOG_ERROR(TAG, "Images were not read in correctly!");
       return false;
    }
    if(images.front().empty()) {
       LOG_ERROR(TAG, "Invalid input image detected!");
       return false;
    }
    LOG_DEBUG(TAG, "%d images successfully read.", images.size());
   
    // Start creating panorama
    int mid = images.size() / 2;
    Mat preImg = images[mid].clone();
    Mat panorama;
    for (int i = mid + 1; i < images.size(); i++) {
        vector<vector<Point2f>> matches;
        find_correspondences_between_two_images(preImg, images[i], matches);
        if(matches.front().empty()) {
            LOG_ERROR(TAG, "No correspondences found!");
            return false;
        }

        const float kDetHomographyThresh = 0.01;
        Matx33f H = Matx33f::zeros();
        compute_homography(matches, H, false);
        if(determinant(H)<kDetHomographyThresh) {
            LOG_ERROR(TAG, "Homography not computed correctly!");
            return false;
        }
        LOG_DEBUG(TAG, "Homography computed.");

        vector<Mat> warpedImages;
        Mat outputLeft, outputRight;
        Rect warppedRightPart = warp(preImg, images[i], H, outputLeft, outputRight, false);
        if (outputLeft.size() != outputRight.size()) {
            LOG_ERROR(TAG, "Warped images not set correctly!");
            return false;
        }
        LOG_DEBUG(TAG, "Warping complete.");

        Mat mask;
        compute_mask_C1(warppedRightPart.width - (preImg.cols + warppedRightPart.width - outputLeft.cols) / 2, outputLeft.size(), mask, false);
        if (!mask.data) {
            LOG_ERROR(TAG, "Empty Mask!");

        }
        LOG_DEBUG(TAG, "Mask computed");


        blend_two_images(outputLeft, outputRight, mask, panorama);
        LOG_DEBUG(TAG, "Blending complete!");
        preImg = panorama;
    }

    for (int i = mid - 1; i >= 0; i--) {
        vector<vector<Point2f>> matches;
        find_correspondences_between_two_images(images[i], preImg, matches);
        if(matches.front().empty()) {
            LOG_ERROR(TAG, "No correspondences found!");
            return false;
        }

        const float kDetHomographyThresh = 0.01;
        Matx33f H = Matx33f::zeros();
        compute_homography(matches, H, true);
        if(determinant(H)<kDetHomographyThresh) {
            LOG_ERROR(TAG, "Homography not computed correctly!");
            return false;
        }
        LOG_DEBUG(TAG, "Homography computed.");

        vector<Mat> warpedImages;
        Mat outputLeft, outputRight;
        Rect warppedLeftPart = warp(images[i], preImg, H, outputLeft, outputRight, true);
        if (outputLeft.size() != outputRight.size()) {
            LOG_ERROR(TAG, "Warped images not set correctly!");
            return false;
        }
        LOG_DEBUG(TAG, "Warping complete.");

        Mat mask;
        compute_mask_C1(warppedLeftPart.width - (preImg.cols + warppedLeftPart.width - outputLeft.cols) / 2, outputLeft.size(), mask, true);
        if (!mask.data) {
            LOG_ERROR(TAG, "Empty Mask!");

        }
        LOG_DEBUG(TAG, "Mask computed");

        blend_two_images(outputLeft, outputRight, mask, panorama);
        LOG_DEBUG(TAG, "Blending complete!");
        preImg = panorama;
    }

   // vector<vector<Point2f>> matches;
   // find_correspondences(images, matches);
   // if(matches.size()!=images.size())
   // {
   //     LOG_ERROR(TAG, "Correspondences not set correctly!");
   //     return false;
   // }
   // if(matches.front().empty())
   // {
   //     LOG_ERROR(TAG, "No correspondences found!");
   //     return false;
   // }
   // LOG_DEBUG(TAG, "Found %d correspondences.", matches.front().size());
   
   // //Compute homography.
   // const float kDetHomographyThresh = 0.01;
   // Matx33f H = Matx33f::zeros();
   // compute_homography(matches, H);
   // if(determinant(H)<kDetHomographyThresh)
   // {
   //     LOG_ERROR(TAG, "Homography not computed correctly!");
   //     return false;
   // }
   // LOG_DEBUG(TAG, "Homography computed.");
   
   // //Warp the image.
   // vector<Mat> warpedImages;
   // warp(images, H, warpedImages);
   // if(warpedImages.size()!=images.size())
   // {
   //     LOG_ERROR(TAG, "Warped images not set correctly!");
   //     return false;
   // }
   // if(warpedImages.front().empty())
   // {
   //     LOG_ERROR(TAG, "Invalid warped image detected!");
   //     return false;
   // }
   // for(auto& img: warpedImages)
   // {
   //     if(img.size()!=warpedImages.front().size())
   //     {
   //         LOG_ERROR(TAG, "Warped images must have the same dimension!");
   //         return false;
   //     }
   // }
   // LOG_DEBUG(TAG, "Warping complete.");
   
   // //Blend.

   // blend(warpedImages, panorama);
   // LOG_DEBUG(TAG, "Blending complete!");
   
   //Save the image.
   imwrite(outputImagePath, panorama);
   
   return true;
}

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "NativeLogging.h"

using namespace std;
using namespace cv;

// helper function
static inline void normalizeSingleImage(const Mat& src, Mat& des){
    src.convertTo(des, CV_32F);
    des /= 255;
}

// begin
static const char* TAG = "HDR";

static void read_input_images(const vector<string>& inputImagePaths, vector<Mat>& images) {
    images.clear();
    for (int i = 0; i < inputImagePaths.size(); i ++) {
        Mat img = imread(inputImagePaths[i]);
        if (img.data != NULL) {
            images.push_back(img);
        }
    }
}

static void align_2nd_to_1st_img(Mat& img1, Mat& img2) {
    // Calculate descriptors (feature vectors)
    std::vector<KeyPoint> keyPoints1, keyPoints2;
    Mat descriptor1, descriptor2;
    
    OrbFeatureDetector detector(5000);
    detector.detect(img1, keyPoints1);
    detector.detect(img2, keyPoints2);

    OrbDescriptorExtractor extractor;
    extractor.compute(img1, keyPoints1, descriptor1);
    extractor.compute(img2, keyPoints2, descriptor2);
    
    // Match descriptor vectors
    BFMatcher matcher;
    std::vector<vector< DMatch >> matches;
    matcher.knnMatch(descriptor2, descriptor1, matches, 2);
    
    std::vector< DMatch > good_matches;
    for (int i = 0; i < matches.size(); i ++) {
        float rejectRatio = 0.8;
        if (matches[i][0].distance / matches[i][1].distance > rejectRatio)
            continue;
        good_matches.push_back(matches[i][0]);
    }
    
    std::vector<Point2f> good_keyPoints1, good_keyPoints2;
    for (int i = 0; i < good_matches.size(); i ++) {
        good_keyPoints1.push_back(keyPoints1[good_matches[i].trainIdx].pt);
        good_keyPoints2.push_back(keyPoints2[good_matches[i].queryIdx].pt);
    }
    
    Mat H = findHomography( good_keyPoints2, good_keyPoints1, CV_RANSAC );
    warpPerspective(img2, img2, H, img1.size(), INTER_NEAREST);
}

static void align_images(vector<Mat>& images) {
    size_t numImages = images.size();
    if (numImages < 2) return;
    for (int i = 0; i < numImages - 1; i++) {
        // do alignment to each pair of images
        align_2nd_to_1st_img(images[i], images[i + 1]);
    }
}

static void getContrast(const Mat& image, Mat& c) {
    // Laplacian filter
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    Mat img_gray, img_filtered;
    cvtColor(image, img_gray, CV_RGB2GRAY );
    Laplacian(img_gray, img_filtered, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    img_filtered = cv::abs(img_filtered);
    normalizeSingleImage(img_filtered, c);
}

static void getSaturation(const Mat& image, Mat& s) {
    Mat img;
    Size imgSize = image.size();
    normalizeSingleImage(image, img);
    vector<Mat> splitted;
    split(img, splitted);
    Mat saturation = Mat(imgSize.height, imgSize.width, CV_32F, 0.0);
    Mat mu = (splitted[0] + splitted[1] + splitted[2]) / 3.0;
    saturation = ((splitted[0] - mu).mul(splitted[0] - mu) +
                  (splitted[1] - mu).mul(splitted[1] - mu) +
                  (splitted[2] - mu).mul(splitted[2] - mu)) / 3.0;
    cv::sqrt(saturation, s);
}

static void getWellExposedness(const Mat& image, Mat& e) {
    // Gaussian
    const float sig_gs = 0.2;
    const float mu_gs = 0.5;
    Mat img;
    normalizeSingleImage(image, img);
    Size imgSize = image.size();
    vector<Mat> splitted;
    split(img, splitted);
    
    Mat exposedness = Mat(imgSize.height, imgSize.width, CV_32F, 1);
    for (int j = 0; j < 3; j ++) {
        cv::exp((splitted[j] - mu_gs).mul(splitted[j] - mu_gs) / ( - 0.5 * sig_gs * sig_gs),
                splitted[j]);
        multiply(exposedness, splitted[j], exposedness);
    }
    e = exposedness;
}

static void normalizeMatrix(vector<Mat>& mats) {
    if (mats.size() < 1) return;
    assert(mats[0].depth() == CV_32F);
    Mat sum = Mat(mats[0].rows, mats[0].cols, CV_32F, 0.0); // should be 0.0, 0 will be interpreted as void
    for ( int i = 0; i < mats.size(); i++ ) {
        Mat thresholdMat; // numerical
        threshold(mats[i], thresholdMat, 1e-30, 1e-30, THRESH_BINARY_INV);
        mats[i] += thresholdMat;
        sum += mats[i];
    }
    for ( int i = 0; i < mats.size(); i++ ) {
        mats[i] /= sum;
    }
}

static void compute_weights(const vector<Mat>& images, vector<Mat>& weights) {
    if (images.size() < 1) return;
    weights.clear();
    Size imgSize = images[0].size();
    
    // Weight parameter
    const float w_c = 1;
    const float w_s = 1;
    const float w_e = 1;

    for (int i = 0; i < images.size(); i ++) {
        assert (images[i].channels() == 3 && images[i].size() == imgSize);
        Mat currentWeight = Mat(imgSize.height, imgSize.width, CV_32F, 1);
        Mat c, s, e;
    
        getContrast(images[i], c);
        multiply(currentWeight, c * w_c, currentWeight);
        
        getSaturation(images[i], s);
        multiply(currentWeight, s * w_s, currentWeight);
        
        getWellExposedness(images[i], e);
        multiply(currentWeight, e * w_e, currentWeight);
    
        weights.push_back(currentWeight);
    }
    normalizeMatrix(weights);
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

static void blend_pyramids(const vector<Mat>& images, const vector<Mat>& weights, int maxPyrIndex, Mat& output) {
    if (images.size() != weights.size()) return;
    Mat currImg, currLevel;
    vector<Mat> blendedPyr, splitted;
    
    for (int i = 0; i < images.size(); i ++){
        images[i].convertTo(currImg, CV_32F);
        vector<Mat> weightGaussianPyr, imgGaussianPyr, imgLaplacianPyr;
        buildPyramid(weights[i], weightGaussianPyr, maxPyrIndex);
        buildPyramid(currImg, imgGaussianPyr, maxPyrIndex);
        buildLaplacianPyramid(imgGaussianPyr, imgLaplacianPyr);
        
        for (int j = 0; j < maxPyrIndex + 1; j ++) {
            split(imgLaplacianPyr[j], splitted);
            vector<Mat> blendedForOneChannel;
            for (int c = 0; c < splitted.size(); c ++){
                blendedForOneChannel.push_back(weightGaussianPyr[j].mul(splitted[c]));
            }
            cv::merge(blendedForOneChannel, currLevel);
            if ( blendedPyr.size() < maxPyrIndex + 1) {
                blendedPyr.push_back(currLevel);
            }
            else {
                blendedPyr[j] += currLevel;
            }
        }
    }
    reconstructLaplacianPyramid(blendedPyr, output);
}

// This is the main entry point for the exposure-fusion module.
// It accepts two arguments:
// 1. inputImagePaths: A vector of absolute paths of the images to fuse.
// 2. outputImagePath: The path where the output image should be saved.
bool exposure_fusion(const vector<string>& inputImagePaths, const string& outputImagePath)
{
   //Read in the given images.
   vector<Mat> images;
   read_input_images(inputImagePaths, images);

   //Verify that the images were correctly read.
   if(images.size()!=inputImagePaths.size()) {
       LOG_ERROR(TAG, "Images were not read in correctly!");
       return false;
   }

   //Verify that the images are RGB images of the same size.
   const int numChannels = 3;
   Size imgSize = images[0].size();
   for(const Mat& img: images) {
       if(img.channels()!=numChannels) {
           LOG_ERROR(TAG, "CreateHDR expects 3 channel RGB images!");
           return false;
       }
       if(img.size()!=imgSize) {
           LOG_ERROR(TAG, "HDR images must be of equal sizes!");
           return false;
       }
   }
   LOG_DEBUG(TAG, "%d images successfully read.", images.size());

   //Make sure that the images line up correctly.
   align_images(images);
   LOG_DEBUG(TAG, "Image alignment complete.");

   //Compute the weights for each image.
   vector<Mat> weights;
   compute_weights(images, weights);
   if(weights.size()!=images.size()) {
       LOG_ERROR(TAG, "Image weights were not generated!");
       return false;
   }
   LOG_DEBUG(TAG, "Weight computation complete.");

   //Fusion using Laplacian pyramid blending.
   int maxPyrIdx = 6;
   Mat outputImage;
   blend_pyramids(images, weights, maxPyrIdx, outputImage);
   if(outputImage.empty()) {
       LOG_ERROR(TAG, "Blending did not produce an output!");
       return false;
   }
   LOG_DEBUG(TAG, "Blending complete!");

   //Save output.
   bool result = imwrite(outputImagePath, outputImage);
   if(!result) {
       LOG_ERROR(TAG, "Failed to save output image to file!");
   }
   return result;
}

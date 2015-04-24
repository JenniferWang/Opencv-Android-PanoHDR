#include "PanoHDR.h"
#include <vector>
#include <string>
#include <iostream>

using namespace std;

//In HDR.cpp:
bool exposure_fusion(const vector<string>& inputImagePaths, const string& outputImagePath);

//In Panorama.cpp:
bool create_panorama(const vector<string>& inputImagePaths, const string& outputImagePath);

enum ImageOperation
{
    IMAGE_OP_PANORAMA = 0,
    IMAGE_OP_HDR = 1
};

static string convert_to_string(JNIEnv* env, jstring js)
{
    const char* stringChars = env->GetStringUTFChars(js, 0);
    string s = string(stringChars);
    env->ReleaseStringUTFChars(js, stringChars);
    return s;
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_panohdr_ImageProcessingTask_processImages(JNIEnv* env,
        jobject, jobjectArray inputImages, jstring outputPath, jint opCode)
{
    string outputImagePath = convert_to_string(env, outputPath);
    vector<string> inputImagePaths;
    int imageCount = env->GetArrayLength(inputImages);
    for(int i = 0; i < imageCount; ++i)
    {
        jstring js = (jstring) (env->GetObjectArrayElement(inputImages, i));
        inputImagePaths.push_back(convert_to_string(env, js));
    }

    switch(opCode)
    {
        case IMAGE_OP_PANORAMA:
            create_panorama(inputImagePaths, outputImagePath);
            break;
        case IMAGE_OP_HDR:
            exposure_fusion(inputImagePaths, outputImagePath);
            break;
        default:
            cerr << "Invalid operation code provided: " << opCode << endl;
            break;
    }
}

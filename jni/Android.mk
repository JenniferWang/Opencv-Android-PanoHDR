LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# Tegra optimized OpenCV.mk
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
# include $(OPENCV_PATH)/sdk/native/jni/OpenCV-tegra3.mk
include /Users/jennifer/CS231m/android/OpenCV-2.4.8.2-Tegra-sdk/sdk/native/jni/OpenCV-tegra3.mk

# Linker
LOCAL_LDLIBS += -llog

# Our module sources
LOCAL_MODULE    := PanoHDR
LOCAL_SRC_FILES := PanoHDR.cpp Panorama.cpp HDR.cpp NativeLogging.cpp

include $(BUILD_SHARED_LIBRARY)

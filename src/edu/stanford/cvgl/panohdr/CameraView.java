package edu.stanford.cvgl.panohdr;

import java.io.FileOutputStream;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.util.AttributeSet;
import android.util.Log;

public class CameraView extends JavaCameraView
{
    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }
    //private static final String TAG = "CameraView";
    private static final String TAG = "test";
    private int picCount = 0;
    
    public void takePicture(final String outputPath) {
        PictureCallback pictureCallback = new PictureCallback() {
            @Override
            public void onPictureTaken(byte[] data, Camera cam) {
                Bitmap picture = BitmapFactory.decodeByteArray(data, 0, data.length);
                try {
                	float aspectRatio = picture.getHeight()/(float)picture.getWidth();
                	int scaledWidth = 640;
                	int scaledHeight = (int)(aspectRatio*scaledWidth);
                	Bitmap scaled = Bitmap.createScaledBitmap(picture, scaledWidth, scaledHeight, true);                	
                    FileOutputStream out = new FileOutputStream(outputPath);
                    scaled.compress(Bitmap.CompressFormat.JPEG, 95, out);
                    scaled.recycle();
                    picture.recycle();
                    //mCamera.startPreview();
                    Log.i(TAG, "preview started");
                } 
                catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        Log.i(TAG, "Taking picture.");
        mCamera.takePicture(null, null, pictureCallback);
    }
    
    public void takeHDRPicture(final String outputPath) {
        Camera.Parameters parameters = mCamera.getParameters();
   
        if (picCount == 0) {
            int minExpComp = parameters.getMinExposureCompensation();
            parameters.setExposureCompensation((2 * minExpComp) / 3);
            Log.i(TAG, "Set expsure compensation to " + parameters.getExposureCompensation());
            mCamera.setParameters(parameters);
            try{
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            mCamera.startPreview();
            takePicture(outputPath);
            picCount++;
        }
        else if (picCount == 1) {
            int maxExpComp = parameters.getMaxExposureCompensation();
            parameters.setExposureCompensation(maxExpComp / 3);
            Log.i(TAG, "Set expsure compensation to " + parameters.getExposureCompensation());
            mCamera.setParameters(parameters);
            try{
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            mCamera.startPreview();
            takePicture(outputPath);
            picCount++;
        }
    }
   
}

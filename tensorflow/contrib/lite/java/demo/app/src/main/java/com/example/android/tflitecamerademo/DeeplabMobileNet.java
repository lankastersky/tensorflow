/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo;

import android.app.Activity;

import java.io.IOException;

/**
 * This classifier works with the quantized MobileNet model.
 */
public class DeeplabMobileNet extends ImageClassifier {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  private float[][] outputs;

  /**
   * Initializes an {@code ImageClassifier}.
   *
   * @param activity
   */
  DeeplabMobileNet(Activity activity) throws IOException {
    super(activity);
    int w = getImageSizeX();
    int h = getImageSizeY();
    outputs = new float[w][h];
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip
    //return "mobilenet_quant_v1_224.tflite";
    return "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "labels_mobilenet_quant_v1_224.txt";
  }

  @Override
  protected int getImageSizeX() {
    return 513;
  }

  @Override
  protected int getImageSizeY() {
    return 513;
  }

  @Override
  protected int getNumBytesPerChannel() {
    // the quantized model uses a single byte only
    return 1;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
    imgData.put((byte) (pixelValue & 0xFF));
  }

  @Override
  protected float getProbability(int labelIndex) {
    return 0;
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    return 0;
  }

  private static final int num_classes = 21;

  @Override
  protected void runInference() {
    int w = getImageSizeX();
    int h = getImageSizeY();
    float[][][][] internal_outputs = new float[1][w][h][num_classes];
    tflite.run(imgData, internal_outputs);
    arg_max(internal_outputs[0], w, h);
  }

  private void arg_max(float[][][] internal_outputs, int w, int h) {
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        float max = Integer.MIN_VALUE;
        int max_index = 0;
        for (int k = 0; k < num_classes; k++) {
          if (max < internal_outputs[i][j][k]) {
            max = internal_outputs[i][j][k];
            max_index = k;
          }
        }
        outputs[i][j] = internal_outputs[i][j][max_index];
      }
    }
  }
}

# TF Lite Android App

## Building in Android Studio with TensorFlow Lite AAR from JCenter.
The build.gradle is configured to use TensorFlow Lite's nightly build.

If you see a build error related to compatibility with Tensorflow Lite's Java API (example: method X is
undefined for type Interpreter), there has likely been a backwards compatible
change to the API. You will need to pull new app code that's compatible with the
nightly build and may need to first wait a few days for our external and internal
code to merge.

## Building from Source with Bazel

1. Follow the [Bazel steps for the TF Demo App](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#bazel):

  1. [Install Bazel and Android Prerequisites](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install-bazel-and-android-prerequisites).
     It's easiest with Android Studio.

      - You'll need at least SDK version 23.
      - Make sure to install the latest version of Bazel. Some distributions
        ship with Bazel 0.5.4, which is too old.
      - Bazel requires Android Build Tools `26.0.1` or higher.
      - **Bazel is incompatible with NDK revisions 15 and above,** with revision
        16 being a compile-breaking change. [Download an older version manually
        instead of using the SDK Manager.](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install-bazel-and-android-prerequisites)
      - You also need to install the Android Support Repository, available
        through Android Studio under `Android SDK Manager -> SDK Tools ->
        Android Support Repository`.

  2. [Edit your `WORKSPACE`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#edit-workspace)
     to add SDK and NDK targets.

     NOTE: As long as you have the SDK and NDK installed, the `./configure`
     script will create these rules for you. Answer "Yes" when the script asks
     to automatically configure the `./WORKSPACE`.

      - Make sure the `api_level` in `WORKSPACE` is set to an SDK version that
        you have installed.
      - By default, Android Studio will install the SDK to `~/Android/Sdk` and
        the NDK to `~/Android/Sdk/ndk-bundle` (but the NDK should be a manual
        download until Bazel supports NDK 16. See bullet points under (1)).

2. Build the app with Bazel. The demo needs C++11:

  ```shell
  bazel build -c opt --cxxopt='--std=c++11' \
    //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo
  ```

3. Install the demo on a
   [debug-enabled device](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install):

  ```shell
  adb install bazel-bin/tensorflow/contrib/lite/java/demo/app/src/main/TfLiteCameraDemo.apk
  ```
## Deeplab background segmentation model
The model is converted to Tensorflow lite from the checkpoint [mobilenetv2_coco_voc_trainaug](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) using the next command where _toco_ was built from the latest github version:

./toco \
  --input_file=/tmp/tf/frozen_inference_graph.pb \
  --output_file=/tmp/tf/frozen_inference_graph.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shapes=1,513,513,3 \
  --inference_type=FLOAT \
  --inference_input_type=QUANTIZED_UINT8 \
  --output_arrays=ResizeBilinear_3 \
  --input_arrays=sub_7
  
\#  --input_arrays=ImageTensor \
\#  --output_arrays=SemanticPredictions  
  
  Converting the full model (with ImageTensor as input and SemanticPredictions as output) is currently not supported and gets the error:

_Some of the operators in the model are not supported by the standard TensorFlow Lite runtime. If you have a custom implementation for them you can disable this error with --allow_custom_ops, or by setting allow_custom_ops=True when calling tf.contrib.lite.toco_convert(). Here is a list of operators for which you will need custom implementations: Pack_
  

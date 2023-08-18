# MLPerf Inference - Object Detection - KILT
This implementation runs object detection models with the KILT backend using TensorRT API on an Nvidia GPU.
The inference is run on serialized engines generated with the NVidia toolset (details below).

Currently it supports the following models:
- retinanet

## Setting up your environment

| Step | Description | Command |
| --- | --- | --- |
| 1 | Start with a clean work_collection | <pre><code>axs byname work_collection , remove</code></pre> |
| 2a | Import this repo and config into your work_collection using SSH `axs2tensorrt-dev` | <pre><code>axs byquery git_repo,collection,repo_name=axs2tensorrt-dev,url=git@github.com:krai/axs2tensorrt-dev.git</code></pre> |
|   | `axs2kilt-dev`    | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev,url=git@github.com:krai/axs2kilt-dev.git</code></pre> |
| 2b | Import this repo and config into your work_collection using HTTPS | <pre><code>axs byquery git_repo,collection,repo_name=axs2tensorrt-dev</code></pre> |
|   | | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev</code></pre> |
| 3 | Import other necessary repos (`axs2mlperf`) into your work_collection | <pre><code>axs byquery git_repo,collection,repo_name=axs2mlperf</code></pre> |
|   | `kilt-mlperf-dev` | <pre><code>axs byquery git_repo,repo_name=kilt-mlperf-dev,url=git@github.com:krai/kilt-mlperf-dev.git</code></pre> |
| 4 | Set Python version for compatibility | <pre><code>ln -s /usr/bin/python3.9 $HOME/bin/python3</code></pre> |
| 5 | Set Python version in axs | <pre><code>axs byquery shell_tool,can_python</code></pre> |

## Downloading Retinanet dependencies

| Step | Description | Command |
| --- | --- | --- |
| 1 | Compile the program binary | <pre><code>axs byquery compiled,kilt_executable,retinanet,device=tensorrt</code></pre> |
| 2 | Download `openimages-mlperf.json` | <pre><code>axs byquery inference_ready,openimages_annotations,v2_1</code></pre> |
| 3 | Download Openimages Datasets Validation | <pre><code>axs byquery downloaded,openimages_mlperf,validation+</code></pre> |
| 4 | Download Calibration Openimages Datasets | <pre><code>axs byquery openimages_mlperf,calibration</code></pre> |
| 5 | Preprocess Calibration Datasets | <pre><code>axs byquery preprocessed,dataset_name=openimages,preprocess_method=pillow_torch,index_file=openimages_cal_images_list.txt,calibration+</code></pre> |
| 6 | Preprocess Full Openimages Datasets | <pre><code>axs byquery preprocessed,dataset_name=openimages,preprocess_method=pillow_torch,first_n=24781,quantized+</code></pre> |
| 7 | Download Original Model | <pre><code>axs byquery downloaded,onnx_model,model_name=retinanet,no_nms</code></pre> |
| 8 | Set up a docker container for running NVidia submissions (https://github.com/mlcommons/inference_results_v3.0/tree/main/closed/NVIDIA). And use it to generate engines with provided custom configs, that are available at axs2kilt/retinanet_kilt_loadgen_tensorrt/config/. Before generating engines ensure that the TensorRT version inside the container is the same as shown in the requrements (8.6.1).| |
| 9 | Copy the resulting engines (one for the Offline scenatio, anothe one - for the Server scenario) into your work_collection. | <pre><code>scp ... ~/work_collection/tensorrt_bert_model/</code></pre> |
| 10 | Copy the plugin libnmsoptplugin.so from inside the docker container at a path <pre><code>build/plugins/RNNTOptPlugin/</code></pre> into <pre><code>../kilt-mlperf-dev/plugins/</code></pre>. | |

## Benchmarking Retinanet

| Step | Description | Command |
| --- | --- | --- |
| 1 | Measure Accuracy  | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy_report</code></pre> |
| 2 | Run Performance (Quick Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1</code></pre> |
| 4 | Run Performance (Full Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=</measured value/></code></pre> |

To see the results of a performance run, look at the last line in the console output, go to the output directory and open a file named "mlperf_log_summary.txt".
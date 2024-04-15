# LVChat

This is the official implementation of our paper **LVChat: Facilitating Long Video Comprehension**. Our code base is built on the repo [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2). 

## Environment Preparation
```shell
conda create --name lvchat python=3.11
pip install -r requirements.txt
```
## Datasets
We used the [instruction data](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md) for training. Specifically, we used the following subsets (Please refer to the link [here](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT/tree/main/video) which includes all the json file needed for training): 
```
conversation_videochat1
conversation_videochat2
conversation_videochatgpt
caption_videochat
reasoning_clevrer_qa
reasoning_clevrer_mc
reasoning_next_qa
```
To replicate our training for Frame Scalable Encoding (FSE), please download the datasets [Clevrer](http://clevrer.csail.mit.edu/), [NExT-QA](https://github.com/doc-doc/NExT-QA), [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data), [WebVid-10M](https://github.com/m-bain/webvid)(However, this dataset is no longer available) as well as the json files from [VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT/tree/main/video). 
Then we put all the datasets as the following structure: 
```
- data
    - ANet
        - activitynet_train_videos_video_chatgpt
    - anno
        - video
            - caption
            - conversation
            - reasoning
    - clevrer
    - internvid-10s (This is the instruction dataset collected by VideoChat2. These videos are from InternVid (https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid). Considering the data is too large, you may can download the video by yourself.  For example, “LLU5X98aozs_648.258.mp4”, “LLU5X98aozs”is YouTube ID, “648.258”is the start time，and the video clip duration is 10s. Thanks to the author Kunchang Li of VideoChat2 for offering the link and instructions.)
    - nextqa
    - WebVid10M (All the videos of VideoChat v1 data are from here)
```

## Base model preparation

1. Download the VideoBLIP model.
```shell
wget -P video_models https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/umt_l16_qformer.pth
```

2. Follow [here](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat#running-usage) to prepare vicuna-7b-v0 and place it under video_models


## Training with Frame Scalable Encoding (FSE)
Download the model `videochat2_7b_stage3.pth` from [here](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/videochat2/videochat2_7b_stage3.pth) then put it under the folder `video_models`. Now the folder `video_models` should have the following structure: 
```
- video_models
    - vicuna-7b-v0
    l16_25m.pth
    umt_l16_qformer.pth
    videochat2_7b_stage3.pth
```

For Validation, please refer to the following section to download MVBench and put the dataset under the folder `./MVBench`. 

Then simply run the following code (remember to set the number of gpus in the file `NUM_GPUS`). 
```
sh run_7b_stage4.sh
```

## Evaluation

### Download MVBench
Download from [Hugging Face](https://huggingface.co/datasets/OpenGVLab/MVBench) and place it under `./MVBench`. The file structure under `MVBench` is: 
```
- assert
- json
- video
.gitattributes
README.md
```

### Prepare street-scene data(required if want to use the extended MVBench data) 
```shell
bash download_street_scnene.sh 
```

### Prepare LV-Chat Model
Please download the model from [LV-Chat](https://huggingface.co/YuWangX/LVChat). Put the pth file `7b_stage4.pth` under the folder `video_models`. 

### Evaluate LV-Chat on MVBench
Run the script to test our model and the result will be written to `logs`: 
```shell
bash run_mvbench.sh
```
You can also run the baseline (VideoChat2) using:
```shell
bash run_mvbench.sh --config ./configs/config_videochat2.json
```


### Evaluate LV-Chat on Real-world datasets
#### TACoS
1. Download TACoS dataset from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus) and place the `videos` folder under `./TACoS`.
2. Download GPT-4 generated summary:
```shell
wget -P ./TACoS https://huggingface.co/datasets/Kevin99z/tacos_summary/resolve/main/summary.json
```
3. Evaluate TACoS
```shell
bash run_tacos.sh # add --config ./configs/config_videochat2.json to test the baseline
```

#### EgoSchema
1. Download EgoSchema [here](https://github.com/egoschema/EgoSchema) and place it under `./EgoSchema`.
2. Evaluate EgoSchema
```shell
bash run_egoschema.sh # add --config ./configs/config_videochat2.json to test the baseline
```

If you find our paper or code useful, please consider citing our paper. 

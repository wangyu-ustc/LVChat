import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "data/anno"
data_path = 'data'

# ============== pretraining datasets=================
available_corpus = dict(
    # image
    caption_coco=[
        f"{anno_root_it}/image/caption/coco/train.json", 
        f"{data_path}/coco_caption",
    ],
    caption_llava=[
        f"{anno_root_it}/image/caption/llava/train.json", 
        f"{data_path}/coco_caption",
    ],
    caption_minigpt4=[
        f"{anno_root_it}/image/caption/minigpt4/train.json", 
        f"{data_path}/minigpt4/image",
    ],
    caption_paragraph_captioning=[
        f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
        f"{data_path}/m3it/image-paragraph-captioning",
    ],
    caption_textcaps=[
        f"{anno_root_it}/image/caption/textcaps/train.json", 
        f"{data_path}/m3it/textcap",
    ],
    classification_imagenet=[
        f"{anno_root_it}/image/classification/imagenet/train.json", 
        f"{data_path}/m3it/imagenet",
    ],
    classification_coco_itm=[
        f"{anno_root_it}/image/classification/coco_itm/train.json", 
        f"{data_path}/m3it/coco-itm",
    ],
    conversation_llava=[
        f"{anno_root_it}/image/conversation/llava/train.json", 
        f"{data_path}/coco_caption",
    ],
    reasoning_clevr=[
        f"{anno_root_it}/image/reasoning/clevr/train.json", 
        f"{data_path}/m3it/clevr",
    ],
    reasoning_visual_mrc=[
        f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
        f"{data_path}/m3it/visual-mrc",
    ],
    reasoning_llava=[
        f"{anno_root_it}/image/reasoning/llava/train.json", 
        f"{data_path}/coco_caption",
    ],
    vqa_vqav2=[
        f"{anno_root_it}/image/vqa/vqav2/train.json", 
        f"{data_path}/m3it/vqa-v2",
    ],
    vqa_gqa=[
        f"{anno_root_it}/image/vqa/gqa/train.json", 
        f"{data_path}/m3it/gqa",
    ],
    vqa_okvqa=[
        f"{anno_root_it}/image/vqa/okvqa/train.json", 
        f"{data_path}/m3it/okvqa",
    ],
    vqa_a_okvqa=[
        f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
        f"{data_path}/m3it/a-okvqa",
    ],
    vqa_viquae=[
        f"{anno_root_it}/image/vqa/viquae/train.json", 
        f"{data_path}/m3it/viquae",
    ],
    vqa_ocr_vqa=[
        f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
        f"{data_path}/m3it/ocr-vqa",
    ],
    vqa_text_vqa=[
        f"{anno_root_it}/image/vqa/text_vqa/train.json", 
        f"{data_path}/m3it/text-vqa",
    ],
    vqa_st_vqa=[
        f"{anno_root_it}/image/vqa/st_vqa/train.json", 
        f"{data_path}/m3it/st-vqa",
    ],
    vqa_docvqa=[
        f"{anno_root_it}/image/vqa/docvqa/train.json", 
        f"{data_path}/m3it/docvqa",
    ],
    # video
    caption_textvr=[
        f"{anno_root_it}/video/caption/textvr/train.json", 
        f"{data_path}/TextVR/Video",
        "video"
    ],
    caption_videochat=[
        f"{anno_root_it}/video/caption/videochat/train.json", 
        f"{data_path}/WebVid10M",
        "video"
    ],
    caption_videochatgpt=[
        f"{anno_root_it}/video/caption/videochatgpt/train.json", 
        # f"{data_path}/ANet/ANet_320p_fps30",
        f"{data_path}/ANet/activitynet_train_videos_video_chatgpt",
        "video"
    ],
    caption_webvid=[
        f"{anno_root_it}/video/caption/webvid/train.json", 
        f"{data_path}/WebVid2M",
        "video"
    ],
    caption_youcook2=[
        f"{anno_root_it}/video/caption/youcook2/train.json", 
        f"{data_path}/youcook2/split_videos",
        "video"
    ],
    classification_k710=[
        f"{anno_root_it}/video/classification/k710/train.json", 
        "",
        "video"
    ],
    classification_ssv2=[
        f"{anno_root_it}/video/classification/ssv2/train.json", 
        f"{data_path}/video_pub/ssv2_video",
        "video"
    ],
    conversation_videochat1=[
        f"{anno_root_it}/video/conversation/videochat1/train.json", 
        f"{data_path}/WebVid10M",
        "video"
    ],
    conversation_videochat2=[
        f"{anno_root_it}/video/conversation/videochat2/train.json", 
        f"{data_path}/internvid-10s",
        "video"
    ],
    conversation_videochatgpt=[
        f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        f"{data_path}/ANet/activitynet_train_videos_video_chatgpt",
        "video"
    ],
    reasoning_next_qa=[
        f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        f"{data_path}/nextqa",
        "video"
    ],
    reasoning_clevrer_qa=[
        f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        f"{data_path}/clevrer/video_train",
        "video"
    ],
    reasoning_clevrer_mc=[
        f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",  
        f"{data_path}/clevrer/video_train",
        "video"
    ],
    vqa_ego_qa=[
        f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        f"{data_path}/EgoQA/split_videos",
        "video"
    ],
    vqa_tgif_frame_qa=[
        f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        f"{data_path}/tgif",
        "video"
    ],
    vqa_tgif_transition_qa=[
        f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        f"{data_path}/tgif",
        "video"
    ],
    vqa_webvid_qa=[
        f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        f"{data_path}/WebVid2M",
        "video"
    ],
)


# add mc for clevrer_qa
available_corpus["videochat2_instruction"] = [
    available_corpus["caption_coco"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
]


available_corpus["videochat2_instruction_stage4"] = [
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["caption_videochat"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["reasoning_next_qa"],
]


# available_corpus["videochat2_instruction_stage4"] = [
#     available_corpus["conversation_videochat1"],
#     # available_corpus["conversation_videochat2"],
#     available_corpus["conversation_videochatgpt"],
# ]
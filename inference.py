# %%
import os
import argparse
import json
import torch
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import math
from tqdm import tqdm


from models.videochat2_it_long import VideoChat2_it_Long
from common import answer, get_sinusoid_encoding_table
from eval_tacos import evaluate_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--dataset", type=str, default="TACoS", choices=["TACoS", "nextQA", "ES"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", default=200, type=int, help="Max number of tokens to generate")
    parser.add_argument("--debug", "-d", action="store_true")

    parser.add_argument("--output_file", "-o", type=str, default="outputs.json")
    parser.add_argument("--no_video", action="store_true", help="whether to use video features")
    parser.add_argument("--config_path", type=str, default="./configs/config_lvchat.json")

    parser.add_argument("--num_frm", type=int, default=-1, help="the number of frames to use, will use ceil(len(video)/8)*8 frames when num_frm=-1")
    parser.add_argument("--max_num_frm", type=int, default=160, help="the max number of frames when num_frm is set to -1; equals the number of max number of frames used in training")
    parser.add_argument("--n_frame_per_clip", type=int, default=16, help="number of frames per clip; -1 means all frames")
    parser.add_argument('--interleave', action='store_true')
    parser.add_argument('--inter_token_num', type=int, default=-1, help="the number of tokens to use when interleaving. when set to -1, will adopt the ceil(video_length/max_num_frm)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda:0")   

    args = parser.parse_args()

    return args

args = parse_args()
print(f"args: {args}")
print(args)


def initialize_model(args, resolution = 224):
    from utils.config import Config
    if args.config_path is not None:
        cfg = Config.from_file(args.config_path)
    else:
        cfg = Config.from_file(os.path.join(os.path.dirname(args.model_path), "config.json"))
    # load stage2 model
    # cfg.model.vision_encoder.num_frames = 4
    cfg.model.vision_encoder.num_frames = args.n_frame_per_clip
    cfg.model.base_frame_num = args.n_frame_per_clip
    model = VideoChat2_it_Long(config=cfg.model)
    model = model.to(args.device)
    model = model.eval()

    #  position embedding
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*args.n_frame_per_clip, cur_frame=args.n_frame_per_clip)
    model.vision_encoder.encoder.pos_embed = new_pos_emb
    return model



import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

input_mean = [0.48145466, 0.4578275, 0.40821073]
input_std = [0.26862954, 0.26130258, 0.27577711]
transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])

def get_num_segments(args, video_length):
    """
    Return: the total number of frames to use
    """
    if args.num_frm < 0:
        if args.interleave:
            if args.inter_token_num > -1:
                return args.inter_token_num * args.max_num_frm
            elif video_length > args.max_num_frm:
                return int(math.ceil(video_length / args.max_num_frm) * args.max_num_frm)
        # use 8 frames when len(video) in 1-8s, 16 frames when len(video) in 9-16s; capped at self.max_frame_num
        return min(int(math.ceil(video_length / args.n_frame_per_clip)) * args.n_frame_per_clip, args.max_num_frm) 
    else:
        return args.num_frm

def get_index(bound, fps, max_frame, first_idx=0):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    curr_video_length = max(1, (end_idx - start_idx) // fps)
    num_segments = get_num_segments(args, curr_video_length)

    curr_video_segments = num_segments
    seg_size = float(end_idx - start_idx) / curr_video_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(curr_video_segments)
    ])
    return frame_indices

vid_len_map = {}

def read_video(video_path, bound=None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    vid_len_map[video_path] = len(vr) / fps
    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0) 
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs


# %%
if args.seed > -100:
    from transformers import set_seed
    set_seed(args.seed)

if args.dataset == "TACoS":
    data = os.listdir("TACoS/videos")
    outputs_dict = {"file": [], "prompt": [], "output": []}
elif args.dataset == "nextQA":
    # enumerate all video paths
    video_paths = {}
    video_dir = "../nextQA/NExTVideo"
    for series in os.listdir("../nextQA/NExTVideo"):
        for video in os.listdir(os.path.join(video_dir, series)):
            video_paths[video[:-4]] = (os.path.join(video_dir, series, video))
    import csv
    split_file = open(f"../nextQA/nextqa/val.csv")
    data = [datum for datum in csv.DictReader(split_file)]
    data = sorted(data, key=lambda x: int(x['frame_count']), reverse=True)[:200]
    outputs_dict = {"file": [], "answer": [], "prompt": [], "output": []}
elif args.dataset == "ES": # Ego Schema
    base_dir = "./EgoSchema"
    subset_answers = json.load(open(os.path.join(base_dir, "subset_answers.json")))
    # Load necessary JSON files
    with open(os.path.join(base_dir,"questions.json")) as questions_f:
        questions = json.load(questions_f)
    data = [q for q in questions if q["q_uid"] in subset_answers]
    for q in data:
        q['answer'] = subset_answers[q['q_uid']]
    outputs_dict = {"file": [], "answer": [], "prompt": [], "output": []}

model = initialize_model(args)


def embed_video(video, system="", interleave=False):
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(args.device)
    video_token_length = args.max_num_frm # the maximum number of frames used in training, 160 by default
    num_vid_token =  TC//3//video_token_length

    if interleave and num_vid_token > 1:
        # from TC//3 values, extract 80 out of it
        indices = np.linspace(0, num_vid_token * (video_token_length-1), video_token_length)

        # video_for_context = video[:, indices]

        with torch.no_grad():
            video_list = []
            for i in range(0, num_vid_token):
                video_part = video[:, indices + i]
                video_emb, _ = model.embed_image(video_part, system)
                video_list.append(video_emb)
    else:
        video_list = []
        with torch.no_grad():
            video_emb, _ = model.embed_image(video, system) # embed the video into multiple video tokens
        video_list.append(video_emb)
        interleave=False
    return video_list, interleave
# %%

correct, total = 0, 0

for item in tqdm(data):
    
    img_list = []
    if args.dataset == "TACoS":
        video_path = os.path.join("TACoS/videos", item)
        system = ("Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. "
        "Based on your observations, describe what is happening in the video as detailed as possible.")
        prompt = (
        "###Human: <Video><VideoHere></Video>\n"
        f"###Human: {system}\n"
        "###Assistant: "
    )
    else: # QA data
        system = ("Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. "
                  "Based on your observations, select the best option that accurately addresses the question."
                  )
        if args.dataset == "nextQA":
            video_path = video_paths[item['video_id']]
            options = [item[f'a{i}'] for i in range(0, 5)]
            question = item['question']

        elif args.dataset == "ES":
            video_path = os.path.join(base_dir, 'videos/videos', item['q_uid']+'.mp4')
            options = [item[f'option {i}'] for i in range(0, 5)]
            question = item['question']
            words = question.split()
            if 'c' in words or "c's" in words:
                question += "c stands for the camera wearer."
            elif 'C' in words or "C's" in words:
                question += "C stands for the camera wearer."
        
        text_options = "\n".join([f"({chr(ord('A')+i)}) {option}" for i, option in enumerate(options)])
        prompt = (
            "###Human: <Video><VideoHere></Video>\n"
            f"###Human: {system}\n"
            f"Question: {question}\n"
            f"Options:\n {text_options}\n"
            "###Assistant: Best option:("
        )
    torch_imgs = read_video(video_path)

    video_list, interleave = embed_video(torch_imgs, system, interleave=args.interleave)

    output, _ = answer(
        args=args,
        prompt=prompt, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        print_res=True,
        interleave=interleave,
    )
    if args.dataset == 'TACoS':
        outputs_dict['file'].append(video_path)
        print("file:", video_path)
    else: # QA data
        label = chr(ord('A')+int(item['answer']))
        pred = output.strip()[0]
        if pred == label:
            correct += 1
        total += 1
        outputs_dict['file'].append(video_path)
        outputs_dict['answer'].append(label)

    outputs_dict['prompt'].append(prompt)
    outputs_dict['output'].append(output)

    print("prompt:", prompt)
    print("output:", output, flush=True)


    # add_text(state, "Please give me the description of the video", None, first_run=True)
    # for x in chat.answer(state, img_list, temperature=0.68, max_new_tokens=512, first_run=True):
    #     print(x)

    if args.debug:
        break
if args.dataset in ['nextQA', 'ES']:
    print(f"Accuracy: {correct / total}")

video_lengths = list(vid_len_map.values())
print("avg video length:", sum(video_lengths)/len(video_lengths) if video_lengths else 0)
outputs_dict["video_lengths"] = video_lengths
if args.dataset == "TACoS":
    outputs_dict["results"] = evaluate_answer(outputs_dict)

with open(args.output_file, "w") as f:
    json.dump(outputs_dict, f, indent=4)


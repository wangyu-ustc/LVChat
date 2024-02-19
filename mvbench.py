# %%
import os
import io
import json

from models.videochat2_it_long import VideoChat2_it_Long
from utils.easydict import EasyDict
import torch

from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from tqdm import tqdm

import imageio
import cv2
import math

from streetscene import StreetScene
from common import hashstr, answer, get_sinusoid_encoding_table
from utils.config import Config


# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, default="all", help="the datasets to use. Can be a list of dataset names (e.g. ['Action Sequence','Object Interaction']) or 'all' to use all datasets")
parser.add_argument("--num_frm", type=int, default=-1, help="the number of frames to use. When set to -1, will use min(max_num_frm, ceil(len(video)/n_frame_per_clip)*n_frame_per_clip) frames. Note len(video) is the length in seconds.")
parser.add_argument("--max_num_frm", type=int, default=160, help="the max number of frames when num_frm is set to -1; equals the number of max number of frames used in training")
parser.add_argument("--n_frame_per_clip", type=int, default=16, help="number of frames per clip; when set to -1, use all frames in one clip")
parser.add_argument("--output_file", "-o", type=str, default="outputs.json", help="the output file to save the result")
parser.add_argument("--base_dir", type=str, default="./MVBench", help="the base directory of the MVBench dataset")
parser.add_argument("--config_path", type=str, default="./configs/config_lvchat.json", help="the path to the config file")
parser.add_argument("--target_video_length", type=int, default=-1, help="target duration of video in seconds. When set to -1, will not extend the video length")
parser.add_argument('--interleave', action='store_true', help="whether to use the interleaved frame encoding")
parser.add_argument('--inter_token_num', type=int, default=-1, help="the number of tokens to use when interleaving. when set to -1, will adopt the ceil(len(video)/max_num_frm). Note len(video) is the length in seconds.")

parser.add_argument("--device", default="cuda:0")
parser.add_argument('--seed', type=int, default=-100, help='random seed')
parser.add_argument("--debug", '-d', action="store_true")
parser.add_argument("--f", help="a dummy argument to fool the jupyter notebook")

args = parser.parse_args()
args.max_num_frm = (args.max_num_frm // args.n_frame_per_clip) * args.n_frame_per_clip
args.join_video = args.target_video_length > -1
if args.datasets != "all":
    args.datasets = eval(args.datasets)

print(args, flush=True)
if args.seed > -100:
    from transformers import set_seed
    set_seed(args.seed)
# %%
def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        interleave=False,
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(args.device)
    video_token_length = args.max_num_frm # the maximum number of frames used in training, 160 by default
    num_vid_token =  TC//3//video_token_length
    if interleave and num_vid_token > 1:
        # from TC//3 values, extract 80 out of it
        indices = np.linspace(0, num_vid_token * (video_token_length-1), video_token_length)

        with torch.no_grad():
            video_list = []
            for i in range(0, num_vid_token):
                video_part = video[:, indices + i]
                if system_q:
                    video_emb, _ = model.embed_image(video_part, system + data_sample['question'])
                else:
                    video_emb, _ = model.embed_image(video_part, system)
                video_list.append(video_emb)
    else:
        video_list = []
        with torch.no_grad():
            if system_q:
                video_emb, _ = model.embed_image(video, system + data_sample['question'])
            else:
                video_emb, _ = model.embed_image(video, system) # embed the video into multiple video tokens
        video_list.append(video_emb)
        interleave=False

    chat = EasyDict({
        "system": system,
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    ask(prompt, chat)
    
    chat.messages.append([chat.roles[1], answer_prompt])
    if answer_prompt:
        prompt = get_prompt2(chat)
    else:
        prompt = get_prompt(chat)

    llm_message = answer(
        args=args,
        prompt=prompt, 
        model=model, 
        do_sample=False, 
        img_list=video_list, 
        max_new_tokens=100, 
        print_res=print_res,
        interleave=interleave
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['answer']}")
    return llm_message

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag
# %%

data_list_all = {
    "Action Sequence": ("action_sequence.json", f"{args.base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", f"{args.base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", f"{args.base_dir}/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", f"{args.base_dir}/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", f"{args.base_dir}/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", f"{args.base_dir}/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", f"{args.base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", f"{args.base_dir}/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", f"{args.base_dir}/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", f"{args.base_dir}/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", f"{args.base_dir}/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", f"{args.base_dir}/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", f"{args.base_dir}/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", f"{args.base_dir}/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", f"{args.base_dir}/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", f"{args.base_dir}/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", f"{args.base_dir}/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", f"{args.base_dir}/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", f"{args.base_dir}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", f"{args.base_dir}/video/clevrer/video_validation/", "video", False),
}

if args.datasets == "all":
    data_list = data_list_all
else:
    data_list = {k: data_list_all[k] for k in args.datasets}
assert len(data_list) > 0
data_dir = f"{args.base_dir}/json"

frame_counter = {}
vr_counter = {}
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, target_video_length=-1, resolution=224, max_num_frm=80, interleave=False):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame 
        }
        self.use_join_video = args.join_video
        self.num_segments = num_segments
        self.target_video_length = target_video_length
        self.max_num_frm = max_num_frm
        self.interleave = interleave

        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
        self.transform2 = T.Compose([
            Stack(),
            ToTorchFormatTensor(),
            # GroupNormalize(input_mean, input_std) 
        ])
        self.second_video = StreetScene(base_dir=f"{args.base_dir}/../street-scene/raw/")
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_num_segments(self, video_length):
        """
        Return: the total number of frames to use
        """
        if self.num_segments < 0:
            if self.interleave:
                if args.inter_token_num > -1:
                    return args.inter_token_num * self.max_num_frm
                elif video_length > self.max_num_frm:
                    return int(math.ceil(video_length / self.max_num_frm) * self.max_num_frm)
            # use 8 frames when len(video) in 1-8s, 16 frames when len(video) in 9-16s; capped at self.max_frame_num
            return min(int(math.ceil(video_length / args.n_frame_per_clip)) * args.n_frame_per_clip, self.max_num_frm) 
        else:
            return self.num_segments
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        curr_video_length = max(1, (end_idx - start_idx) // fps)
        self.curr_video_length = curr_video_length
        num_segments = self.get_num_segments(max(self.target_video_length, curr_video_length))
        if self.target_video_length > curr_video_length:
            # only use a part of the whole num_segments for the current video
            curr_video_segments = int(math.ceil(curr_video_length / self.target_video_length * num_segments))
        else:
            curr_video_segments = num_segments
        seg_size = float(end_idx - start_idx) / curr_video_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(curr_video_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        vr_counter[video_path] = len(vr)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        self.fps = fps
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        self.fps = fps
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        self.fps = fps
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def join_video(self, video_path, torch_imgs, target_len):
        """
        extend the length of the given torch_imgs with a second video
        Args:
            torch_imgs: the read video as images processed by self.transform
            video_path: the path where the torch_imgs are read
            target_len: the length (by second) of the video to return 
        """
        if args.debug:
            print("========join video========")
            print("current length:", self.curr_video_length)
            print("target_length:", target_len)
        second_length = target_len - self.curr_video_length
        if second_length < 0:
            return torch_imgs
        
        torch_imgs = torch_imgs.view(-1, 3, 224, 224) # TODO: use variable
        if second_length >= len(self.second_video):
            print("WARNING: Target length cannot be satisfied:", target_len)

        max_start_time = len(self.second_video) - second_length
        start_time = hashstr(video_path) % max_start_time # select start_time from 0 -> max_end_time
        num_frm = self.get_num_segments(target_len)-len(torch_imgs)
        if num_frm <= 0:
            return torch_imgs.view(-1,224,224)
        if args.debug:
            print(f"video_path: {video_path}")
            print(f"second video to join: {start_time}s-{start_time+second_length}s, {num_frm} frames")
        second_video = self.second_video.read_by_time(start_time=start_time, duration=second_length, num_frm=num_frm)
        torch_imgs2 = self.transform2(second_video).view(-1, 3, 224, 224)
        pos = hashstr(video_path+":insert") % len(torch_imgs2)
        return torch.concat([torch_imgs2[:pos], torch_imgs, torch_imgs2[pos:]], dim=0).view(-1, 224, 224)



    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        if args.debug:
            print("used frames", len(torch_imgs)//3)    
        if self.use_join_video:
            torch_imgs = self.join_video(video_path, torch_imgs, target_len=self.target_video_length)
        frame_counter[video_path] = len(torch_imgs)//3
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

# num_frame = 16
# %%
resolution = 224
dataset = MVBench_dataset(data_dir, data_list, num_segments=args.num_frm, resolution=resolution, target_video_length=args.target_video_length, max_num_frm=args.max_num_frm, interleave=args.interleave)

# %%
if args.config_path is not None:
    cfg = Config.from_file(args.config_path)
else:
    cfg = Config.from_file(os.path.join(os.path.dirname(args.model_path), "config.json"))

cfg.model.vision_encoder.num_frames = args.n_frame_per_clip
cfg.model.base_frame_num = args.n_frame_per_clip
model = VideoChat2_it_Long(config=cfg.model)
model = model.to(args.device)
model = model.eval()

#  position embedding
new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*args.n_frame_per_clip, cur_frame=args.n_frame_per_clip)
model.vision_encoder.encoder.pos_embed = new_pos_emb


# %%
save_path = "./test"

correct = 0
total = 0
res_list = []
acc_dict = {}

for example in tqdm(dataset):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    pred = infer_mvbench(
        example, 
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
        question_prompt="\nOnly give the best option.",
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=True,
        system_llm=True,
        interleave=args.interleave
    )
    gt = example['answer']
    res_list.append({
        'pred': pred,
        'gt': gt
    })
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 30, task_type, '-' * 30)

with open(f"{save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

# %%
final_res = dict()
final_res['avg_frame_number'] = sum(frame_counter.values()) / len(frame_counter) if len(frame_counter) > 0 else 0
final_res['avg_video_length'] = sum(vr_counter.values()) / len(vr_counter) if len(vr_counter) > 0 else 0
correct = 0
total = 0
for k, v in acc_dict.items():
    final_res[k] = v[0] / v[1] * 100
    correct += v[0]
    total += v[1]    
final_res['Avg'] = correct / total * 100

print(final_res)

with open(args.output_file, "w") as f:
    json.dump(final_res, f)


    
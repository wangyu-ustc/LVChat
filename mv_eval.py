# %%
import os
import io
import json

from models.videochat2_it_long import VideoChat2_it_Long
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

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

from torchvision import transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy
import imageio
import cv2
import math

from pathlib import Path
from streetscene import StreetScene

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


def get_context_emb(prompt, model, img_list, print_res=False, use_mem=False, device='cuda'):
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False, use_mem=False, device='cuda'):
    stop_words_ids = [
        torch.tensor([835]).to(device),
        torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if use_mem:
        prompt1, prompt2 = prompt.split('</Video>')
        prompt1 += '</Video>'
        video_embeds = get_context_emb(prompt1, model, img_list, print_res=print_res, device=device) # embeds before the end of video
        model.llama_model.fill_in_memory(video_embeds)
        # input_ids = model.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        prompt_ids = model.llama_tokenizer(prompt2, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        # mem = model.llama_model.search_memory(prompt_ids, max_length=args.num_frm)
        # T = mem[0][0].shape[-2]
        # attention_mask = torch.ones((prompt_ids.shape[0], prompt_ids.shape[1]+T), dtype=torch.long).to(model.device)
        # position_ids = torch.arange(T, T + prompt_ids.shape[1], dtype=torch.long).unsqueeze(0).to(model.device)

        outputs = model.llama_model.generate(
                input_ids=prompt_ids,
                query_ids=prompt_ids,
                # inputs_embeds=,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
    else:
        embs = get_context_emb(prompt, model, img_list, print_res=print_res, device=device)
        with torch.no_grad():
            outputs = model.llama_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()



def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

# %%
def infer_mvbench(
        model,
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        device='cuda'
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(device)
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.embed_image(video, system + data_sample['question'])
        else:
            video_emb, _ = model.embed_image(video, system) # embed the video into multiple video tokens
    video_list.append(video_emb)
#     video_list.append(torch.zeros_like(video_emb))

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

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res, device=device
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    # print(llm_message)
    # print(f"GT: {data_sample['answer']}")
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





# %%
from common import CachedMethod, hashstr
frame_counter = {}
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, base_frame_num=8, num_segments=8, target_video_length=-1, resolution=224, max_num_frm=80, join_video=False):
        self.base_frame_num = base_frame_num
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
        
        self.read_frame_joined = CachedMethod("tvqa.cache", instance_method=True)(self.read_frame_joined)
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame 
            # 'frame': self.read_frame_joined
        }
        self.use_join_video = join_video
        self.num_segments = num_segments
        self.target_video_length = target_video_length
        self.max_num_frm = max_num_frm

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
        # self.second_video = StreetScene(base_dir="/home/zeyuan/data/street-scene/raw/")
    
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
            # use 8 frames when len(video) in 1-8s, 16 frames when len(video) in 9-16s; capped at self.max_frame_num
            return min(int(math.ceil(video_length / self.base_frame_num)) * self.base_frame_num, self.max_num_frm) 
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
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        self.fps = fps
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
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

    def read_frame_joined(self, video_path, bound=None, fps=3):
        path = Path(video_path)
        base_dir = path.parent
        vid_id = path.stem.split('_clip_')[0]
        image_files = []
        for clip_name in os.listdir(f"{base_dir}"):
            if clip_name.startswith(vid_id):
                clip_dir = os.path.join(os.path.join(base_dir, clip_name))
                image_files.extend([os.path.join(clip_dir, f) for f in os.listdir(clip_dir)])
        frame_indices = np.linspace(0, len(image_files), self.num_segments, dtype=int, endpoint=False)
        images_group = []
        for frame_index in frame_indices:
            img = Image.open(image_files[frame_index])
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

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


def evaluate(model, 
             datasets,
             base_frame_num,
             num_frm, 
             resolution, 
             target_video_length, 
             max_num_frm, 
             output_file,
             join_video=False,
             device='cuda'
             ):

    base_dir = "./MVBench"

    data_list_all = {
        "Action Sequence": ("action_sequence.json", f"{base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Action Antonym": ("action_antonym.json", f"{base_dir}/video/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{base_dir}/video/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{base_dir}/video/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{base_dir}/video/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{base_dir}/video/star/Charades_v1_480/", "video", True), # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{base_dir}/video/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{base_dir}/video/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{base_dir}/video/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{base_dir}/video/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{base_dir}/video/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{base_dir}/video/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{base_dir}/video/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{base_dir}/video/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{base_dir}/video/nturgbd/", "video", False),
        "Character Order": ("character_order.json", f"{base_dir}/video/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{base_dir}/video/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{base_dir}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        "Counterfactual Inference": ("counterfactual_inference.json", f"{base_dir}/video/clevrer/video_validation/", "video", False),
    }
    
    # datasets = ['Action Sequence', 'Object Interaction', 'State Change']
    data_list = {k: data_list_all[k] for k in datasets}
    assert len(data_list) > 0   
    data_dir = f"{base_dir}/json"

    # num_frame = 16
    num_frame = num_frm
    
    dataset = MVBench_dataset(data_dir, data_list, 
                              num_segments=num_frame, 
                              base_frame_num=base_frame_num,
                              resolution=resolution, 
                              target_video_length=target_video_length, 
                              max_num_frm=max_num_frm,
                              join_video=join_video)

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
            model,
            example, 
            system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
            question_prompt="\nOnly give the best option.",
            answer_prompt="Best option:(",
            return_prompt='(',
            system_q=False,
            print_res=True,
            system_llm=True,
            device=device,
        )
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
            
    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    # %%
    final_res = dict()
    final_res['avg_frame_number'] = sum(frame_counter.values()) / len(frame_counter) if len(frame_counter) > 0 else 0
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    return final_res['Avg']

import os
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from functools import wraps
import pickle
import json
from transformers import StoppingCriteria, StoppingCriteriaList

class CachedMethod:
    def __init__(self, cache_path, instance_method=False):
        self.cache_path = cache_path
        self.instance_method = instance_method
        self.use_cache=True
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as cache_file_object:
                self.cache = pickle.load(cache_file_object)
        else:
            self.cache = {}

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # Need to make the key hashable. Here we make it a tuple of the args and the items of the kwargs
            # dump the args and kwargs to json and then hash it
            args_idx = 1 if self.instance_method else 0 # Skip the first argument if it's an instance method
            cache_key = json.dumps([args[args_idx:], sorted(kwargs.items())], sort_keys=True)
            if self.use_cache and cache_key in self.cache:
                return self.cache[cache_key]
            else:
                result = func(*args, **kwargs)
                self.cache[cache_key] = result
                wrapped.calls += 1  # Increment the call counter
                if wrapped.calls >= 10:  # Check if it's time to save the cache
                    self.save()
                    wrapped.calls = 0  # Reset the counter after saving
                return result

        wrapped.calls = 0  # Initialize the call counter
        wrapped.save_cache = self.save  # Attach the save function to the wrapped function
        return wrapped

    def save(self):
        with open(self.cache_path, "wb") as cache_file_object:
            pickle.dump(self.cache, cache_file_object)

    def __get__(self, instance, owner):
        # This method is necessary to make the decorator work on instance methods
        # Create a bound method by using the functools.partial method
        from functools import partial
        return partial(self.__call__, instance)

def load_image(img_path):
    """Load image from a path and return as a numpy array."""
    img = cv2.imread(img_path)
    # Convert from BGR to RGB (OpenCV loads images in BGR format by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

crop_size = 224
scale_size = 224
input_mean = [0.48145466, 0.4578275, 0.40821073]
input_std = [0.26862954, 0.26130258, 0.27577711]

transform = T.Compose([
    GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
    GroupCenterCrop(crop_size),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std) 
])

def tvqa_load_video(show_name, vid_id, n_clips=1, max_seconds=-1, base_dir="./TVQA/frames_hq/"):

    clip_num = 0
    
    img_array = []

    while os.path.exists(f"{base_dir}/{show_name}/{vid_id}_clip_{clip_num:02}"):
        foldername = f"{base_dir}/{show_name}/{vid_id}_clip_{clip_num:02}"
        frame_num = 1
        img_path = os.path.join(foldername, f"{frame_num:05}.jpg")
        while os.path.exists(img_path):
            img_array.append(load_image(img_path))
            frame_num += 3
            img_path = os.path.join(foldername, f"{frame_num:05}.jpg")

        clip_num += 1

    img_array = np.stack(img_array)
    

    if max_seconds > 0:
        # get "max_seconds" of imgs from img_array
        if len(img_array) > max_seconds:
            print(f"Video has {len(img_array)} seconds. Extracting {max_seconds} seconds of video.")
            indices = np.linspace(0, len(img_array) - 1, max_seconds, dtype=int)
            img_array = img_array[indices]

    total_frames = len(img_array)
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_frames, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_frames)]

    vid = transform(clip_imgs)
    secs = [str(i) for i in range(len(clip_imgs))]
    TC, H, W = vid.shape
    video = vid.reshape(1, TC//3, 3, H, W)

    return video, secs
    

tvqa_load_video = CachedMethod("tvqa_cache.pkl")(tvqa_load_video)


def load_video(video_path, num_segments=8, return_secs=False):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    TC, H, W = torch_imgs.shape
    video = torch_imgs.reshape(1, TC//3, 3, H, W)
    
    if return_secs:
        fps = float(vr.get_avg_fps())
        secs = [str(round(f / fps, 1)) for f in frame_indices]
        # " " should be added in the start and end
        return video, secs
    else:
        return video
    

def get_context_emb(prompt, model, img_list, print_res=False):
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    # print(prompt_segs)
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.long().to(model.device)
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

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


def hashstr(s: str) -> int:
    return sum(ord(c) * 31**(i % 3) for i, c in enumerate(s)) 


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
def answer(args, prompt, model, img_list, do_sample=True, max_new_tokens=20, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, print_res=False, interleave=False):
    stop_words_ids = [
        torch.tensor([835]).to(args.device),
        torch.tensor([2277, 29937]).to(args.device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
    gen_kwargs = {
        'stopping_criteria': stopping_criteria,
        'num_beams': num_beams,
        'do_sample': do_sample,
        'min_length': min_length,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'length_penalty': length_penalty,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
    }
    
    prompt1, prompt2 = prompt.split('</Video>')
    prompt1 += '</Video>'

    if interleave:
        video_embeds = [
            get_context_emb(prompt1, model, [il], print_res=False)
            for il in img_list
        ]
        # TODO: should we use the first one instead of the last one ?
        past_key_values_list = []
        video_token_length = args.max_num_frm // args.n_frame_per_clip * 96
        with torch.no_grad():
            for idx in range(len(video_embeds)):
                # assert inputs_embeds[idx].shape[1] <= batch_size
                outputs = model.llama_model(inputs_embeds=video_embeds[idx])
                past_key_values = outputs.past_key_values
                # [57 + 960 + 3]
                if idx < len(video_embeds) - 1:
                    past_key_values = tuple(tuple(x[:, :, -(video_token_length+3):-3] for x in past_key_value) for past_key_value in past_key_values )
                past_key_values_list.append(past_key_values)
            past_key_values_all = tuple(tuple(torch.cat([past_key_values[i][j] for past_key_values in past_key_values_list], dim=-2)
                                                for j in range(len(past_key_values_list[0][0])))
                                                for i in range(len(past_key_values_list[0])))
            # embs = get_context_emb(prompt2, model, [], print_res=print_res)
            indices = torch.ones(video_token_length*(len(video_embeds) - 1) + video_embeds[-1].shape[1])
            indices[video_token_length*(len(video_embeds)-1):-(video_token_length+3)] = 0
            indices[-3:] = 0
            indices = torch.where(indices==1)[0]

        prompt_ids = model.llama_tokenizer(prompt2, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        
        gen_kwargs['dividing_indices'] = indices
        gen_kwargs['dividing_factor'] = len(video_embeds)
        gen_kwargs['past_key_values'] = past_key_values_all
        gen_kwargs['position_ids'] = torch.arange(video_embeds[-1].shape[1], video_embeds[-1].shape[1]+prompt_ids.shape[1], dtype=torch.long).unsqueeze(0).to(model.device)

        # # debug
        # new_pkv = []
        # for pkv in past_key_values_all:
        #     new_pkv.append(tuple([pkv[0][:,:,-1020:], pkv[1][:,:,-1020:]]))
        
        with torch.no_grad():
            # outputs = model.llama_model.generate(
            #     # inputs_embeds=embs,
            #     input_ids=prompt_ids,
            #     **gen_kwargs
            # )

            generated_token_ids = []
            while len(generated_token_ids) < gen_kwargs['max_new_tokens']:
                if len(generated_token_ids) == 0:
                
                    output = model.llama_model(
                        input_ids=prompt_ids,
                        past_key_values=gen_kwargs['past_key_values'],
                        position_ids=gen_kwargs['position_ids'],
                        dividing_indices=indices,
                        dividing_factor=len(video_embeds),
                    )

                    generated_token_ids.append(torch.argmax(output.logits[:, -1, :]).cpu().item())
                    past_key_values = output.past_key_values

                else:
                    
                    output = model.llama_model(
                        input_ids=torch.tensor([generated_token_ids[-1]], dtype=torch.long, device=prompt_ids.device).unsqueeze(0),
                        past_key_values=past_key_values,
                        position_ids=torch.tensor(prompt_ids.shape[1] + video_embeds[-1].shape[1] + len(generated_token_ids), dtype=torch.long, device=prompt_ids.device).unsqueeze(0),
                        dividing_indices=indices,
                        dividing_factor=len(video_embeds),
                    )
                    generated_token_ids.append(torch.argmax(output.logits[:, -1, :]).cpu().item())
                    past_key_values = output.past_key_values

            outputs = torch.tensor(generated_token_ids, dtype=torch.long, device=prompt_ids.device).unsqueeze(0)
            # embs = get_context_emb(prompt2, model, [], print_res=print_res)

            # # for debugging:
            # outputs2 = model.llama_model(
            #     inputs_embeds=torch.cat([video_embeds[-1], embs[:, 1:]], dim=1)
            # )

    else:
        embs = get_context_emb(prompt, model, img_list, print_res=print_res)

        with torch.no_grad():
            outputs = model.llama_model.generate(
                inputs_embeds=embs,
                **gen_kwargs
            )
    
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    if args.debug:
        print(output_text)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    # conv.messages[-1][1] = output_text
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

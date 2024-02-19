from configs.instruction_data import *

# ========================= data ==========================
train_corpus = "videochat2_instruction_stage4"
# train_corpus = "conversation_videochatgpt"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict()
test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
num_frames = 160
base_frame_num = 16
random_frames = False
num_frames_test = 160
batch_size = 1
max_txt_l = 512

pre_text = False

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model = dict(
    model_cls="VideoChat2_it_Long",
    vit_blip_model_path="video_models/umt_l16_qformer.pth",
    llama_model_path="video_models/vicuna-7b-v0",
    # videochat2_model_path="output/7b_stage4/ckpt_latest.pth",
    videochat2_model_path="video_models/videochat2_7b_stage3.pth",
    freeze_vit=True,
    freeze_qformer=True,
    max_txt_len="${max_txt_l}", # use large max_txt_len on stage3
    # vit
    low_resource=False,
    add_temp_embed=False,
    vision_encoder=dict(
        name="vit_l14",
        img_size=224, 
        patch_size=16, 
        d_model=1024,
        encoder_embed_dim=1024, 
        encoder_depth=24,
        encoder_num_heads=16, 
        drop_path_rate=0., 
        num_frames="${base_frame_num}",
        tubelet_size=1,
        use_checkpoint=False,
        checkpoint_num=0,
        pretrained="",
        return_index=-2,
        vit_add_ln=True,
        ckpt_num_frame=4,
    ),
    # qformer
    num_query_token=32,
    qformer_hidden_dropout_prob=0.1,
    qformer_attention_probs_dropout_prob=0.1,
    qformer_drop_path_rate=0.2,
    extra_num_query_token=64,
    qformer_text_input=False,
    # prompt
    system="",
    start_token="<Video>",
    end_token="</Video>",
    add_second_msg=False,
    img_start_token="<Image>", 
    img_end_token="</Image>",
    random_shuffle=True, 
    use_flash_attention=False,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    # debug=True,
    base_frame_num=base_frame_num,
)

optimizer = dict(
    opt="adamW",
    lr=2e-6,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.25, warmup_epochs=0.3)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="user",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="videochat2",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "it"

# ========================= others ==========================
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
eval_freq = 5000
seed = 42

save_latest = True
auto_resume = True
# pretrained_path = '/home/wangyu/work/VideoChat2/output/7b_stage5_ceil/ckpt_step_00.pth'  # path to pretrained model weights, for resume only?
pretrained_path = ''  # path to pretrained model weights, for resume only?
output_dir = 'output/7b_stage4'
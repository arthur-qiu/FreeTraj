import gradio as gr

import sys
import pandas as pd
import os
import argparse
import random
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download

sys.path.insert(0, "scripts/evaluation")
from funcs import (
    batch_ddim_sampling,
    batch_ddim_sampling_freetraj,
    load_model_checkpoint,
)
from utils.utils import instantiate_from_config

MAX_KEYS = 5

def check_move(trajectory, video_length=16):
    traj_len = len(trajectory)
    if traj_len < 2:
        return False
    prev_pos = trajectory[0]
    for i in range(1, traj_len):
        cur_pos = trajectory[i]
        if cur_pos[0] > video_length - 1:
            return False
        if (cur_pos[0] - prev_pos[0]) * ((cur_pos[1] - prev_pos[1]) ** 2 + (cur_pos[2] - prev_pos[2]) ** 2) ** 0.5 < 0.02:
            print("Too small movement, please use ori mode.")
            return False
        prev_pos = cur_pos

    return True

def infer(*user_args):
    prompt_in = user_args[0]
    target_indices = user_args[1]
    ddim_edit = user_args[2]
    seed = user_args[3]
    ddim_steps = user_args[4]
    unconditional_guidance_scale = user_args[5]
    video_fps = user_args[6]
    save_fps = user_args[7]
    height_ratio = user_args[8]
    width_ratio = user_args[9]
    radio_mode = user_args[10]
    dropdown_diy = user_args[11]
    frame_indices = user_args[-3 * MAX_KEYS: -2 * MAX_KEYS]
    h_positions = user_args[-2 * MAX_KEYS: -MAX_KEYS]
    w_positions = user_args[-MAX_KEYS:]
    print(user_args)

    video_length = 16
    width = 512 
    height = 320
    ckpt_dir_512 = "checkpoints/base_512_v2"
    ckpt_path_512 = "checkpoints/base_512_v2/model.ckpt"
    if radio_mode == 'ori':
        config_512 = "configs/inference_t2v_512_v2.0.yaml"
    else:
        config_512 = "configs/inference_t2v_freetraj_512_v2.0.yaml"

    trajectory = []
    for i in range(dropdown_diy):
        trajectory.append([int(frame_indices[i]), h_positions[i], w_positions[i]])
    trajectory.sort()
    print(trajectory)

    if not check_move(trajectory):
        print("Error trajectory.")

    input_traj = []
    h_remain = 1 - height_ratio
    w_remain = 1 - width_ratio
    for i in trajectory:
        h_relative = i[1] * h_remain
        w_relative = i[2] * w_remain
        input_traj.append([i[0], h_relative, h_relative+height_ratio, w_relative, w_relative+width_ratio])

    indices_list = target_indices.split(',')
    idx_list = []
    for i in indices_list:
        idx_list.append(int(i))

    config_512 = OmegaConf.load(config_512)
    model_config_512 = config_512.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config_512)
    model = model.cuda()
    if not os.path.exists(ckpt_path_512):
        os.makedirs(ckpt_dir_512, exist_ok=True)
        hf_hub_download(repo_id="VideoCrafter/VideoCrafter2", filename="model.ckpt", local_dir=ckpt_dir_512)
    try:
        model = load_model_checkpoint(model, ckpt_path_512)
    except:
        hf_hub_download(repo_id="VideoCrafter/VideoCrafter2", filename="model.ckpt", local_dir=ckpt_dir_512, force_download=True)
        model = load_model_checkpoint(model, ckpt_path_512)
    model.eval()

    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    seed_everything(seed)

    args = argparse.Namespace(
        mode="base",
        savefps=save_fps,
        n_samples=1,
        ddim_steps=ddim_steps,
        ddim_eta=0.0,
        bs=1,
        height=height,
        width=width,
        frames=video_length,
        fps=video_fps,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_guidance_scale_temporal=None,
        cond_input=None,
        ddim_edit = ddim_edit,
    )

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    batch_size = 1
    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps] * batch_size).to(model.device).long()
    prompts = [prompt_in]
    text_emb = model.get_learned_conditioning(prompts)

    cond = {"c_crossattn": [text_emb], "fps": fps}

    ## inference
    if radio_mode == 'ori':
        batch_samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
            args=args,
        )
    else:
        batch_samples = batch_ddim_sampling_freetraj(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
            idx_list = idx_list,
            input_traj = input_traj,
            args=args,
        )

    vid_tensor = batch_samples[0]
    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

    if radio_mode == 'ori':
        video_path = "output.mp4"
        video_bbox_path = "output.mp4"
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)

        torchvision.io.write_video(
            video_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )
    else:
        video_path = "output_freetraj.mp4"
        video_bbox_path = "output_freetraj_bbox.mp4"
        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)

        torchvision.io.write_video(
            video_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )

        BOX_SIZE_H = input_traj[0][2] - input_traj[0][1]
        BOX_SIZE_W = input_traj[0][4] - input_traj[0][3]
        PATHS = plan_path(input_traj)
        h_len = grid.shape[1]
        w_len = grid.shape[2]
        sub_h = int(BOX_SIZE_H * h_len) 
        sub_w = int(BOX_SIZE_W * w_len)
        for j in range(grid.shape[0]):
            h_start = int(PATHS[j][0] * h_len)
            h_end = h_start + sub_h
            w_start = int(PATHS[j][2] * w_len)
            w_end = w_start + sub_w

            h_start = max(1, h_start)
            h_end = min(h_len-1, h_end)
            w_start = max(1, w_start)
            w_end = min(w_len-1, w_end)

            grid[j, h_start-1:h_end+1, w_start-1:w_start+2, :] = torch.ones_like(grid[j, h_start-1:h_end+1, w_start-1:w_start+2, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
            grid[j, h_start-1:h_end+1, w_end-2:w_end+1, :] = torch.ones_like(grid[j, h_start-1:h_end+1, w_end-2:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
            grid[j, h_start-1:h_start+2, w_start-1:w_end+1, :] = torch.ones_like(grid[j, h_start-1:h_start+2, w_start-1:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)
            grid[j, h_end-2:h_end+1, w_start-1:w_end+1, :] = torch.ones_like(grid[j, h_end-2:h_end+1, w_start-1:w_end+1, :]) * torch.Tensor([127, 255, 127]).view(1, 1, 3)

        torchvision.io.write_video(
            video_bbox_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )

    return video_path, video_bbox_path


examples = [
    ["A squirrel jumping from one tree to another.",],
    ["A bear climbing down a tree after spotting a threat.",],
    ["A deer walking in a snowy field.",],
    ["A lion walking in the savanna grass.",],
    ["A kangaroo jumping in the Australian outback.",],
    ["A corgi running on the grassland on the grassland.",],
    ["A horse galloping on a street.",],
    ["A majestic eagle soaring high above the treetops, surveying its territory.",],
]

css = """
#col-container {max-width: 1024px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
  animation: spin 1s linear infinite;
}
#share-btn-container {
  display: flex; 
  padding-left: 0.5rem !important; 
  padding-right: 0.5rem !important; 
  background-color: #000000; 
  justify-content: center; 
  align-items: center; 
  border-radius: 9999px !important; 
  max-width: 15rem;
  height: 36px;
}
div#share-btn-container > div {
    flex-direction: row;
    background: black;
    align-items: center;
}
#share-btn-container:hover {
  background-color: #060606;
}
#share-btn {
  all: initial; 
  color: #ffffff;
  font-weight: 600; 
  cursor:pointer; 
  font-family: 'IBM Plex Sans', sans-serif; 
  margin-left: 0.5rem !important; 
  padding-top: 0.5rem !important; 
  padding-bottom: 0.5rem !important;
  right:0;
}
#share-btn * {
  all: unset;
}
#share-btn-container div:nth-child(-n+2){
  width: auto !important;
  min-height: 0px !important;
}
#share-btn-container .wrap {
  display: none !important;
}
#share-btn-container.hidden {
  display: none!important;
}
img[src*='#center'] { 
    display: inline-block;
    margin: unset;
}
.footer {
        margin-bottom: 45px;
        margin-top: 10px;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
    }
    .footer>p {
        font-size: .8rem;
        display: inline-block;
        padding: 0 10px;
        transform: translateY(10px);
        background: white;
    }
    .dark .footer {
        border-color: #303030;
    }
    .dark .footer>p {
        background: #0b0f19;
    }
"""

def mode_update(mode):
    if mode == 'demo':
        trajectories_mode = [gr.Row(visible=True), gr.Row(visible=False)]
    elif mode == 'diy':
        trajectories_mode = [gr.Row(visible=False), gr.Row(visible=True)]
    else:
        trajectories_mode = [gr.Row(visible=False), gr.Row(visible=False)]
    return trajectories_mode

def keyframe_update(num):
    keyframes = []
    if type(num) != int:
        num = 0

    for i in range(num):
        keyframes.append(gr.Row(visible=True))
    for i in range(MAX_KEYS - num):
        keyframes.append(gr.Row(visible=False))
    return keyframes

def demo_update(mode):
    if mode == 'topleft->bottomright':
        num = 2
    elif mode == 'bottomleft->topright':
        num = 2
    elif mode == 'topleft->bottomleft->bottomright':
        num = 3
    elif mode == 'bottomright->topright->topleft':
        num = 3
    elif mode == '"V"':
        num = 4
    elif mode == '"^"':
        num = 4
    elif mode == 'left->right->left->right':
        num = 4
    elif mode == 'triangle':
        num = 4
    else:
        num = 0

    return num

def demo_update_frame(mode):
    frame_indices = []
    if mode == 'topleft->bottomright':
        num = 2
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=15))
    elif mode == 'bottomleft->topright':
        num = 2
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=15))
    elif mode == 'topleft->bottomleft->bottomright':
        num = 3
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=9))
        frame_indices.append(gr.Text(value=15))
    elif mode == 'bottomright->topright->topleft':
        num = 3
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=6))
        frame_indices.append(gr.Text(value=15))
    elif mode == '"V"':
        num = 4
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=7))
        frame_indices.append(gr.Text(value=8))
        frame_indices.append(gr.Text(value=15))
    elif mode == '"^"':
        num = 4
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=7))
        frame_indices.append(gr.Text(value=8))
        frame_indices.append(gr.Text(value=15))
    elif mode == 'left->right->left->right':
        num = 4
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=5))
        frame_indices.append(gr.Text(value=10))
        frame_indices.append(gr.Text(value=15))
    elif mode == 'triangle':
        num = 4
        frame_indices.append(gr.Text(value=0))
        frame_indices.append(gr.Text(value=5))
        frame_indices.append(gr.Text(value=10))
        frame_indices.append(gr.Text(value=15))
    else:
        num = 0

    for i in range(MAX_KEYS - num):
        frame_indices.append(gr.Text())
    return frame_indices

def demo_update_h(mode):
    h_positions = []
    if mode == 'topleft->bottomright':
        num = 2
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
    elif mode == 'bottomleft->topright':
        num = 2
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
    elif mode == 'topleft->bottomleft->bottomright':
        num = 3
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.9))
    elif mode == 'bottomright->topright->topleft':
        num = 3
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.1))
    elif mode == '"V"':
        num = 4
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
    elif mode == '"^"':
        num = 4
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
    elif mode == 'left->right->left->right':
        num = 4
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
    elif mode == 'triangle':
        num = 4
        h_positions.append(gr.Slider(value=0.1))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.9))
        h_positions.append(gr.Slider(value=0.1))
    else:
        num = 0

    for i in range(MAX_KEYS - num):
        h_positions.append(gr.Slider())
    return h_positions

def demo_update_w(mode):
    w_positions = []
    if mode == 'topleft->bottomright':
        num = 2
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.9))
    elif mode == 'bottomleft->topright':
        num = 2
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.9))
    elif mode == 'topleft->bottomleft->bottomright':
        num = 3
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.9))
    elif mode == 'bottomright->topright->topleft':
        num = 3
        w_positions.append(gr.Slider(value=0.9))
        w_positions.append(gr.Slider(value=0.9))
        w_positions.append(gr.Slider(value=0.1))
    elif mode == '"V"':
        num = 4
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.8/15*7 + 0.1))
        w_positions.append(gr.Slider(value=0.8/15*8 + 0.1))
        w_positions.append(gr.Slider(value=0.9))
    elif mode == '"^"':
        num = 4
        w_positions.append(gr.Slider(value=0.9))
        w_positions.append(gr.Slider(value=0.8/15*8 + 0.1))
        w_positions.append(gr.Slider(value=0.8/15*7 + 0.1))
        w_positions.append(gr.Slider(value=0.1))
    elif mode == 'left->right->left->right':
        num = 4
        w_positions.append(gr.Slider(value=0.5))
        w_positions.append(gr.Slider(value=0.5))
        w_positions.append(gr.Slider(value=0.5))
        w_positions.append(gr.Slider(value=0.5))
    elif mode == 'triangle':
        num = 4
        w_positions.append(gr.Slider(value=0.5))
        w_positions.append(gr.Slider(value=0.9))
        w_positions.append(gr.Slider(value=0.1))
        w_positions.append(gr.Slider(value=0.5))
    else:
        num = 0

    for i in range(MAX_KEYS - num):
        w_positions.append(gr.Slider())
    return w_positions

def plot_update(*positions):
    key_length = int(positions[-1])
    frame_indices = positions[:key_length]
    h_positions = positions[MAX_KEYS:MAX_KEYS+key_length]
    h_positions_re = []
    for i in h_positions:
        h_positions_re.append(-i)
    w_positions = positions[2*MAX_KEYS:2*MAX_KEYS+key_length]
    traj_plot = gr.ScatterPlot(
        value=pd.DataFrame({"x": w_positions, "y": h_positions_re, "frame": frame_indices}),
        x="x",
        y="y",
        color='frame',
        x_lim= [-0.05, 1.05],
        y_lim= [-1.05, 0.05],
        label="Trajectory",
        width=512,
        height=320,
    )
    return traj_plot


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center;">FreeTraj</h1>
            <p style="text-align: center;">
            Tuning-Free Trajectory Control in Video Diffusion Models
            </p>
            <p style="text-align: center;">
            <a href="https://arxiv.org/abs/2406.16863" target="_blank"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
            <a href="http://haonanqiu.com/projects/FreeTraj.html" target="_blank"><b>[Project Page]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://github.com/arthur-qiu/FreeTraj" target="_blank"><b>[Code]</b></a>
            </p>         
            """
        )

        keyframes = []
        frame_indices = []
        h_positions = []
        w_positions = []

        with gr.Row():
            video_result = gr.Video(label="Video Output")
            video_result_bbox = gr.Video(label="Video Output with BBox")

        with gr.Row():
            prompt_in = gr.Textbox(label="Prompt", placeholder="A corgi running on the grassland on the grassland.", scale = 3)
            target_indices = gr.Textbox(label="Target Indices", placeholder="1,2", scale = 1)

        with gr.Row():
            radio_mode = gr.Radio(label='Trajectory Mode', choices = ['demo', 'diy', 'ori'], scale = 1)
            height_ratio = gr.Slider(label='Height Ratio of BBox',
                              minimum=0.2,
                              maximum=0.4,
                              step=0.01,
                              value=0.3,
                              scale = 1)
            width_ratio = gr.Slider(label='Width Ratio of BBox',
                              minimum=0.2,
                              maximum=0.4,
                              step=0.01,
                              value=0.3, 
                              scale = 1)
          
        with gr.Row(visible=False) as row_demo:
            dropdown_demo = gr.Dropdown(
                label="Demo Trajectory",
                choices= ['topleft->bottomright', 'bottomleft->topright', 'topleft->bottomleft->bottomright', 'bottomright->topright->topleft', '"V"', '"^"', 'left->right->left->right', 'triangle']
            )
            
        with gr.Row(visible=False) as row_diy:
            dropdown_diy = gr.Dropdown(
                label="Number of keyframes",
                choices=range(2, MAX_KEYS+1),
            )
            
        for i in range(MAX_KEYS):
            with gr.Row(visible=False) as row:
                text = f"Keyframe #{i}"
                text = gr.HTML(text, visible=True)
                frame_ids = gr.Textbox(
                    None,
                    label=f"Frame Indices #{i}",
                    visible=True,
                    interactive=True,
                    scale=1
                )
                h_position = gr.Slider(label='Position in Height',
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    scale=1)
                w_position = gr.Slider(label='Position in Width',
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    scale=1)
                
            frame_indices.append(frame_ids)
            h_positions.append(h_position)
            w_positions.append(w_position)
            keyframes.append(row)


        dropdown_demo.change(demo_update, dropdown_demo, dropdown_diy)
        dropdown_diy.change(keyframe_update, dropdown_diy, keyframes)
        dropdown_demo.change(demo_update_frame, dropdown_demo, frame_indices)
        dropdown_demo.change(demo_update_h, dropdown_demo, h_positions)
        dropdown_demo.change(demo_update_w, dropdown_demo, w_positions)
        radio_mode.change(mode_update, radio_mode, [row_demo, row_diy])

        traj_plot = gr.ScatterPlot(
            label="Trajectory",
            width=512,
            height=320,
        )

        h_positions[0].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        h_positions[1].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        h_positions[2].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        h_positions[3].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        h_positions[4].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        w_positions[0].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        w_positions[1].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        w_positions[2].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        w_positions[3].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)
        w_positions[4].change(plot_update, frame_indices + h_positions + w_positions + [dropdown_diy], traj_plot)

        with gr.Row():
            with gr.Accordion('Useful FreeTraj Parameters (feel free to adjust these parameters based on your prompt): ', open=True):
                with gr.Row():
                    ddim_edit = gr.Slider(label='Editing Steps (larger for better control while losing some quality)',
                             minimum=0,
                             maximum=12,
                             step=1,
                             value=6)
                    seed = gr.Slider(label='Random Seed',
                             minimum=0,
                             maximum=10000,
                             step=1,
                             value=123)
                    
        with gr.Row():
            with gr.Accordion('Useless FreeTraj Parameters (mostly no need to adjust): ', open=False):
                with gr.Row():
                    ddim_steps = gr.Slider(label='DDIM Steps',
                              minimum=5,
                              maximum=200,
                              step=1,
                              value=50)
                    unconditional_guidance_scale = gr.Slider(label='Unconditional Guidance Scale',
                              minimum=1.0,
                              maximum=20.0,
                              step=0.1,
                              value=12.0)
                with gr.Row():
                    video_fps = gr.Slider(label='Video FPS (larger for quicker motion)',
                              minimum=8,
                              maximum=36,
                              step=4,
                              value=16)
                    save_fps = gr.Slider(label='Save FPS',
                              minimum=1,
                              maximum=30,
                              step=1,
                              value=10)
                
        with gr.Row():
            submit_btn = gr.Button("Generate", variant='primary')

        with gr.Row():
            gr.Examples(label='Sample Prompts', examples=examples, inputs=[prompt_in, target_indices, ddim_edit, seed, ddim_steps, unconditional_guidance_scale, video_fps, save_fps, height_ratio, width_ratio, radio_mode, dropdown_diy, *frame_indices, *h_positions, *w_positions])

        with gr.Row():
            gr.Markdown(
                """
                <h2 style="text-align: center;">Hints</h2>
                <p style="text-align: center;">
                1. Choose trajectory mode <b>"ori"</b> to see whether the prompt works on the pre-trained model. 
                </p>    
                <p style="text-align: center;">
                2. Adjust the prompt or random seed to get a qualified video.
                </p>  
                <p style="text-align: center;">
                3. Choose trajectory mode <b>"demo"</b> to see whether <b>FreeTraj</b> works or not.
                </p>  
                <p style="text-align: center;">
                4. Choose trajectory mode <b>"diy"</b> to plan new trajectory. It may fail in some extreme cases.
                </p>       
                """
            )
            

    submit_btn.click(fn=infer,
            inputs=[prompt_in, target_indices, ddim_edit, seed, ddim_steps, unconditional_guidance_scale, video_fps, save_fps, height_ratio, width_ratio, radio_mode, dropdown_diy, *frame_indices, *h_positions, *w_positions],
            outputs=[video_result, video_result_bbox],
            api_name="zrscp")

demo.queue(max_size=12).launch(show_api=True)
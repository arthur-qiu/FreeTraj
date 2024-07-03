import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_freetraj import DDIMSampler as DDIMFreeTrajSampler

from utils.utils_freetraj import get_freq_filter, freq_mix_3d, get_path, plan_path

def batch_ddim_sampling_freetraj(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, idx_list=[], input_traj=[], x_T_total=None, args=None, **kwargs):
    ddim_sampler = DDIMFreeTrajSampler(model)
    uncond_type = model.uncond_type
    batch_size, channels, frames, h, w = noise_shape

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    total_shape = [args.n_samples, 1, channels, frames, h, w]
    print('total_shape', total_shape)

    if x_T_total is None:
        x_T_total = torch.randn(total_shape, device=model.device).repeat(1, batch_size, 1, 1, 1, 1)

        noise_flow = True
        if noise_flow:
            print('noise_flow')
            BOX_SIZE_H = input_traj[0][2] - input_traj[0][1]
            BOX_SIZE_W = input_traj[0][4] - input_traj[0][3]
            PATHS = plan_path(input_traj)
            sub_h = int(BOX_SIZE_H * h) 
            sub_w = int(BOX_SIZE_W * w)
            x_T_sub = torch.randn([args.n_samples, 1, channels, sub_h, sub_w], device=model.device)
            for i in range(frames):
                h_start = int(PATHS[i][0] * h)
                h_end = h_start + sub_h
                w_start = int(PATHS[i][2] * w)
                w_end = w_start + sub_w

                # no mix
                x_T_total[:, :, :, i, h_start:h_end, w_start:w_end] = x_T_sub

            filter_shape = [
                1, 
                channels, 
                frames, 
                h, 
                w
            ]

            freq_filter = get_freq_filter(
                filter_shape, 
                device = model.device, 
                filter_type='butterworth',
                n=4,
                d_s=0.25,
                d_t=0.1
            )

            x_T_rand = torch.randn([1, 1, channels, frames, h, w], device=model.device)
            x_T_total = freq_mix_3d(x_T_total.to(dtype=torch.float32), x_T_rand, LPF=freq_filter)

        
    # x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        x_T = x_T_total[_]
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            idx_list=idx_list,
                                            input_traj=input_traj,
                                            ddim_edit = args.ddim_edit,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants

def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_idx(prompt_file):
    f = open(prompt_file, 'r')
    idx_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            indices = l.split(',')
            indices_list = []
            for index in indices:
                indices_list.append(int(index))
            idx_list.append(indices_list)
        f.close()
    return idx_list

def load_traj(prompt_file):
    f = open(prompt_file, 'r')
    traj_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            numbers = l.split(',')
            numbers_list = []
            for number_index in range(len(numbers)):
                if number_index == 0:
                    numbers_list.append(int(numbers[number_index]))
                else:
                    numbers_list.append(float(numbers[number_index]))
            traj_list.append(numbers_list)
        f.close()
    return traj_list

def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def save_videos_with_bbox(batch_tensors, savedir, conddir, filenames, fps=10, input_traj=[]):
    # b,samples,c,t,h,w
    BOX_SIZE_H = input_traj[0][2] - input_traj[0][1]
    BOX_SIZE_W = input_traj[0][4] - input_traj[0][3]
    PATHS = plan_path(input_traj)
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        h_len = video.shape[3]
        w_len = video.shape[4]
        sub_h = int(BOX_SIZE_H * h_len) 
        sub_w = int(BOX_SIZE_W * w_len)
        for i in range(video.shape[1]):
            single_video = video[:, i]
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in single_video] #[3, 1*h, n*w]
            grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            savepath = os.path.join(savedir, f"{filenames[idx]}_{str(i)}.mp4")
            torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})
            for j in range(video.shape[0]): 
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
            bbox_savepath = os.path.join(conddir, f"{filenames[idx]}_{str(i)}.mp4")
            torchvision.io.write_video(bbox_savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


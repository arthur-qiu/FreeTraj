## ___***FreeTraj: Tuning-Free Trajectory Control in Video Diffusion Models***___

 <a href='https://arxiv.org/abs/2406.16863'><img src='https://img.shields.io/badge/arXiv-2406.16863-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='http://haonanqiu.com/projects/FreeTraj.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

_**[Haonan Qiu](http://haonanqiu.com/), [Zhaoxi Chen](https://frozenburning.github.io/), [Zhouxia Wang](http://luoping.me/author/zhouxia-wang/), [Yingqing He](https://github.com/YingqingHe), [Menghan Xia*](https://menghanxia.github.io), and [Ziwei Liu*](https://liuziwei7.github.io/)**_
<br>
(* corresponding author)


## 🔆 Introduction

🤗🤗🤗 FreeTraj is a tuning-free method for trajectory-controllable video generation based on pre-trained video diffusion models.

### Showcases (320x512)

<table class="center">
  <td><img src=assets/demo/0047_1_0.4_0.3.gif width="256"></td>
  <td><img src=assets/demo/0026_0_0.4_0.4.gif width="256"></td>
  <td><img src=assets/demo/0035_1_0.35_0.35.gif width="256"></td>
  <tr>
  <td style="text-align:center;" width="256">"<b>A chihuahua</b> in an astronaut suit floating in the universe, cinematic lighting, glow effect."</td>
  <td style="text-align:center;" width="256">"<b>A swan</b> floating gracefully on a lake."</td>
  <td style="text-align:center;" width="256">"<b>A corgi</b> running on the grassland on the grassland."</td>
  <tr>
</table >

<table class="center">
  <td><img src=assets/demo/0051_1_0.4_0.4.gif width="256"></td>
  <td><img src=assets/demo/0041_0_0.35_0.35.gif width="256"></td>
  <td><img src=assets/demo/0019_0_0.3_0.3.gif width="256"></td>
  <tr>
  <td style="text-align:center;" width="256">"<b>A barrel</b> floating in a river."</td>
  <td style="text-align:center;" width="256">"<b>A dog</b> running across the garden, photorealistic, 4k."</td>
  <td style="text-align:center;" width="256">"<b>A helicopter</b> hovering above a cityscape."</td>
  <tr>
</table >


## 📝 Changelog
- __[2024.07.04]__: 🔥🔥 Release the FreeTraj, trajectory controllable video generation!
- __[2024.07.09]__: 🔥🔥 Release a user-friendly interface.
<br>


## 🧰 Models

|Model|Resolution|Checkpoint|Description
|:---------|:---------|:--------|:--------|
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)|

<br>


## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n freetraj python=3.8.5
conda activate freetraj
pip install -r requirements.txt
```

<br>


## 🤗 Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

```bash
  gradio app/app.py
```

<br>

## 💫 Inference with Command
### 1. Demo

1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/base_512_v2/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_text2video_freetraj.sh
```

### 2. Plan new trajectory
1) Write new trajectory files, the format should be `frame index, h start, h end, w start, w end`. In the current version, the bbox size should be the same. Please refer to `prompts/freetraj/traj_l.txt`.
2) Modify `scripts/run_text2video_freetraj.sh` and set `$traj_file`.
3) Slightly increase `$ddim_edit` to enhance the control ability, but may reduce the video quality.

<br>

## 🚀 My Free Series
[FreeScale](https://github.com/ali-vilab/FreeScale): Tuning-free method for high-resolution image/video generation.

[FreeNoise](https://github.com/AILab-CVC/FreeNoise): Tuning-free method for longer video generation.

## 😉 Citation
```bib
@misc{qiu2024freetraj,
      title={FreeTraj: Tuning-Free Trajectory Control in Video Diffusion Models}, 
      author={Haonan Qiu and Zhaoxi Chen and Zhouxia Wang and Yingqing He and Menghan Xia and Ziwei Liu},
      year={2024},
      eprint={2406.16863},
      archivePrefix={arXiv}
}
```
<br>

## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes. The success rate is not guaranteed due to the variety of generative video prior. 
****

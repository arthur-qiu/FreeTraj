## ___***FreeTraj: Tuning-Free Trajectory Control in Video Diffusion Models***___

 <a href='https://arxiv.org/abs/2406.16863'><img src='https://img.shields.io/badge/arXiv-2406.16863-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='http://haonanqiu.com/projects/FreeTraj.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

_**[Haonan Qiu](http://haonanqiu.com/), [Zhaoxi Chen](https://frozenburning.github.io/), [Zhouxia Wang](http://luoping.me/author/zhouxia-wang/),
<br>
[Yingqing He](https://github.com/YingqingHe), [Menghan Xia*](https://menghanxia.github.io), and [Ziwei Liu*](https://liuziwei7.github.io/)**_
<br>
(* corresponding author)


## 🔆 Introduction

🤗🤗🤗 FreeTraj is a tuning-free method for trajectory-controllable video generation based on pre-trained video diffusion models.

### Showcases (512x320)

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
- __[2024.06.25]__: 🔥🔥 Code coming soon (under cleaning)
<br>


## 🧰 Models

|Model|Resolution|Checkpoint|Description
|:---------|:---------|:--------|:--------|
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)|

<br>

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
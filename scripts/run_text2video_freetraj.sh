name="base_512_v2_ddim6_l"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_freetraj_512_v2.0.yaml'

res_dir="results_freetraj"
prompt_file="prompts/freetraj/text.txt"
idx_file="prompts/freetraj/idx.txt"
traj_file="prompts/freetraj/traj_l.txt"


python3 scripts/evaluation/inference_freetraj.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 0.0 \
--prompt_file $prompt_file \
--idx_file $idx_file \
--traj_file $traj_file \
--ddim_edit 6 \
--fps 16

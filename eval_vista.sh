export ckpt_path="/mnt/src/VISTA_workspace/runs/vista-h-b1/06-02_23-40-25/model_best.pt"
export ckpt_path_1="/mnt/src/VISTA_workspace/runs/vista-h-b1/06-05_12-30-51/model_best.pt"
export ckpt_path_large="/mnt/src/VISTA_workspace/runs/vista-l-b1/06-12_14-42-37/model_best.pt"
export ckpt_4="/mnt/src/VISTA_workspace/runs/vista-h-b1/05-30_03-47-23/model_best.pt"


python /mnt/src/VISTA_workspace/VISTAForCT/inference/eval_vista.py --ckpt_path=$ckpt_path_1 --vit_type=vit_h --output_folder=/mnt/src/VISTA_workspace/pred_train --image_size=512 \
	--json_form=/mnt/src/data/vista_table.json --image_folder=/mnt/src/data \
	--patch_embed_3d --debug=10 --b_min=0 --class_prompts --point_prompts

#! /bin/bash


# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
# export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export WORLD_SIZE=$SLURM_NNODES

export HSU_HOME="/mnt/home/l159753807"
export SHARE_HOME="/mnt/share/NTU_Heart_CT"
export CONTAINER_NAME="$HSU_HOME/workspace/dimage/nemo_monai.sqsh"

export DATA_MNT="${SHARE_HOME}/data:/workspace/data"
# export MODEL_MNT="$HSU_HOME/workspace/model-weight:/workspace/model-weight"
export PROJ_MNT="$SHARE_HOME/VISTA_workspace/CTVISTA/training:/workspace"

export CONTAINER_MOUNT="$DATA_MNT,$PROJ_MNT"
export MAIN1="main_2pt5d.py"
export MAIN2="second_main.py"
export DIST="--distributed"
export vit_size='h'
export batch_size=4
export time=$(date +%m-%d_%H-%M-%S)
export reuse_ckpt="/mnt/VISTA_workspace/runs/vista-h-b1/05-30_03-47-23"
export torchrun="torchrun --nproc-per-node=1 --nnodes=1 --master-port=11234"

export PYTHON_CMD="
	python ./training/main_2pt5d.py --nc=11 --project=Vista \
--max_epochs 5000 --val_every 1 --optim_lr 0.00005 \
--num_patch 160 --num_prompt 32 \
--json_list /mnt/src/data/vista_table_small.json \
--data_dir /mnt/src/data --use_normal_dataset \
--roi_z_iter 27 --save_checkpoint \
--sam_base_model vit_${vit_size} \
--logdir /mnt/src/VISTA_workspace/ckpt/vista-${vit_size}-b$batch_size/$time --point_prompt --label_prompt --seed 114514 \
--iterative_training_warm_up_epoch 50 --batch_size=$batch_size \
--label_prompt_warm_up_epoch 25 --patch_embed_3d --fold=0 --name=vista-${vit_size}-b$batch_size-${time}_no_cache_full --sam_image_size=512 \
--workers=16 
"

#echo "MASTER_ADDR: $MASTER_ADDR"
#echo "MASTER_PORT: $MASTER_PORT"

$PYTHON_CMD


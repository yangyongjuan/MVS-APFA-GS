export CUDA_VISIBLE_DEVICES=1
data_dir="/home/zhanglj/yyj/tandt/tnt_data"
dir_ply="mvsgs_pointcloud/tnt/FMT_trans-source_size4"
scenes=(Train Truck)
iter=5000

python run.py --type evaluate --cfg_file configs/tnt_eval.yaml save_ply True dir_ply $dir_ply

for scene in ${scenes[@]}
do  
python lib/train.py  --eval --iterations $iter -s $data_dir/$scene -p $dir_ply
python lib/render.py -c -m output/$scene --iteration $iter -p $dir_ply
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]} --iter $iter
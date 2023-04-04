# Train PeerMLP
python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv    --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type mlp --gnn_model GraphSAGE --weight_decay 0
python -u src/main.py --batch_size 1000 --dataset ogbn-products --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type mlp --gnn_model GraphSAGE --weight_decay 0

#Train GNNs with RandomInit
python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv    --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type gnn --gnn_model GraphSAGE --weight_decay 0
python -u src/main.py --batch_size 1000 --dataset ogbn-products --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type gnn --gnn_model GraphSAGE --weight_decay 0

#Train GNNs with MLPInit
python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv    --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type gnn --gnn_model GraphSAGE --weight_decay 0 --pretrained_checkpoint ./ogbn-arxiv_GraphSAGE_mlp_512_4_31415.pt
python -u src/main.py --batch_size 1000 --dataset ogbn-products --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type gnn --gnn_model GraphSAGE --weight_decay 0 --pretrained_checkpoint ./ogbn-products_GraphSAGE_mlp_512_4_31415.pt
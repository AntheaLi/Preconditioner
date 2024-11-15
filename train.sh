
python train.py \
--use-data-num 5000 \
--batch-size 16 \
--mesh 'eight_mid_res' \
--param '100.0-100.0' \
--num-iterations 5 \
--hidden-layers-encoder 1 \
--hidden-layers-decoder 1 \
--hidden-layers-processor 1 \
--hidden-dim 16 \
--lr 1e-3 \
--epochs 10000 \
--model model_v1 \
--dataset heatmultisource \
--tensorboard \
--x-loss-weight 1.0 \
--rhs-loss-weight 1.0 \
--precond-loss-weight 0.0 \
--diag-loss-weight 1e2 \
--diag-loss2-weight 1.0 \
--simulate \
--val-freq 2 \
--loss l2 \
--diagonalize \
--use-pred-x \
--use-global \
--exp-name 'exp1'

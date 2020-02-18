# the training_way n=5 (default), just need 1 GPU with 12G memomry.
# for miniImageNet
python train.py --lambda_pre_fsl_loss 1 --lambda_da 1 --lambda_new_fsl_loss 1 --proto_attention 1 --gpu 0 \
--dataset 'MiniImageNet' --model_type 'ResNet' --init_weights './saves/mini_wideresnet28_best.pth.tar'

# for CUB
# python train.py --lambda_pre_fsl_loss 1 --lambda_da 1 --lambda_new_fsl_loss 1 --proto_attention 1 --gpu 0 \
# --lr 0.001 --dataset 'CUB' --model_type 'ResNet' --init_weights './saves/cub_wideresnet28_best.pth.tar'
# or
# python train.py --lambda_pre_fsl_loss 1 --lambda_da 1 --lambda_new_fsl_loss 1 --proto_attention 0 --gpu 0 \
# --lr 0.001 --dataset 'CUB' --model_type 'ResNet' --init_weights './saves/cub_wideresnet28_best.pth.tar'

# the training_way n=10, need 2 GPUs with 24G memory. 
# We set train_way=10 for tieredImageNet
# python train.py --lambda_pre_fsl_loss 1 --lambda_da 1 --lambda_new_fsl_loss 1 --proto_attention 1 --gpu 0,1 \
# --dataset 'TieredImageNet' --model_type 'ResNet' --init_weights './saves/tiered_wideresnet28_best.pth.tar'


cd VAE
echo "Training Autoencoder, this might take a long time"
CUDA_VISIBLE_DEVICES=0 python VAE_train.py --data_dir '/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad' --num_genes 18996 --save_dir '../checkpoint/AE/my_VAE' --max_steps 200000
echo "Training Autoencoder done"

cd ..
echo "Training diffusion backbone"
CUDA_VISIBLE_DEVICES=0 python cell_train.py --data_dir '/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad' --num_genes 18996 --ae_dir 'checkpoint/AE/my_VAE/model_seed=0_step=150000.pt' \
    --model_name 'my_diffusion' --lr_anneal_steps 800000
echo "Training diffusion backbone done"

echo "Training classifier"
CUDA_VISIBLE_DEVICES=0 python classifier_train.py --data_dir '/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad' --classifier_dir "checkpoint/classifier/my_classifier" \
    --iterations 400000 --num_genes 18996 --ae_dir 'checkpoint/AE/my_VAE/model_seed=0_step=150000.pt'
echo "Training classifier, done"
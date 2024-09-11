cd VAE
echo "Training Autoencoder, this might take a long time"
python VAE_train.py --data_dir '/stor/lep/diffusion/multiome/openproblems_RNA_new.h5ad' --num_genes 13431 --save_dir '../output/checkpoint/AE/open_problem' --max_steps 200000
echo "Training Autoencoder done"

cd ..
echo "Training diffusion backbone"
python cell_train.py --data_dir '/stor/lep/diffusion/multiome/openproblems_RNA_new.h5ad' --vae_path 'checkpoint/AE/open_problem/model_seed=0_step=150000.pt' \
    --model_name 'open_problem' --lr_anneal_steps 800000 --save_dir 'output/checkpoint/backbone'
echo "Training diffusion backbone done"

echo "Training classifier"
python classifier_train.py --data_dir '/stor/lep/diffusion/multiome/openproblems_RNA_new.h5ad' --model_path "output/checkpoint/classifier/open_problem_classifier" \
    --iterations 400000 --vae_path 'checkpoint/AE/open_problem/model_seed=0_step=150000.pt' --num_class 22
echo "Training classifier, done"
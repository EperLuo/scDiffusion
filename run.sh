cd VAE
python VAE_train.py --data_dir "/home/wma/Downloads/25335436/muris/data/tabula_muris/all.h5ad" --num_genes 18996 --save_dir '../output/checkpoint/AE/my_VAE' --max_steps 200000 --state_dict /home/wma/Downloads/annotation_model_v1 |& tee -a train.log
cd ..
mkdir output/checkpoint/backbone
python cell_train.py --data_dir '/home/wma/Downloads/25335436/muris/data/tabula_muris/all.h5ad' --vae_path 'output/checkpoint/AE/my_VAE/model_seed=0_step=150000.pt' --model_name 'my_diffusion' --save_dir 'output/checkpoint/backbone' --lr_anneal_steps 800000 |& tee -a train_backbone.log
mkdir output/checkpoint/classifier
python classifier_train.py --data_dir '/home/wma/Downloads/25335436/muris/data/tabula_muris/all.h5ad' --model_path "output/checkpoint/classifier/my_classifier" --iterations 200000 --vae_path 'output/checkpoint/AE/my_VAE/model_seed=0_step=150000.pt' --num_class=12 |& tee -a train_classifier.log
mkdir output/muris_condi
python classifier_sample.py --model_path 'output/checkpoint/backbone/my_diffusion/model800000.pt' --classifier_path 'output/checkpoint/classifier/my_classifier/model199999.pt' --sample_dir 'output/muris_condi/muris' --num_samples 3000 --batch_size 1000 |& tee -a sample.log

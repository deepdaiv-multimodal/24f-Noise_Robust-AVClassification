 ### 1. Fine-tuning
 ```
 python -W ignore src/run_cavmae_ft.py --model cav-mae-ft --dataset vggsound --data-train train_relative.json --data-val test_relative.json --exp-dir ./exp --label-csv class_labels_indices_urban.csv --n_class 10 --lr 1e-4 --n-epochs 10 --batch-size 128 --save_model True --freqm 48 --timem 192 --mixup 0.5 --label_smooth 0.1 --lrscheduler_start 2 --lrscheduler_decay 0.5 --lrscheduler_step 1 --dataset_mean 0 --dataset_std 1 --target_length 1024 --noise True --loss CE --metrics acc --warmup True --wa True --wa_start 3 --wa_end 10 --lr_adapt False --pretrain_path /home/jskim/project/24f-Noise_Robust-AVClassification/egs/vggsound/cav-mae-scale++.pth --ftmode multimodal --freeze_base False --head_lr 10 --num-workers 32             
 ```

 ### 2.Prompt Learning
 ```
 python -W ignore src/run_cavmae_pl.py --model cav-mae-ft --dataset vggsound --data-train train_relative.json --data-val test_relative.json --exp-dir ./exp_test --label-csv class_labels_indices_urban.csv --n_class 10 --lr 1e-3 --n-epochs 50 --batch-size 512 --finetuned_path {finetuned_model_path} --proportion 0.3 --dataset_mean 0 --dataset_std 1 --target_length 1024 --mode train
 ```

### 3.Eval
!python -W ignore src/run_cavmae_pl.py --model cav-mae-ft --dataset vggsound --data-train /content/drive/MyDrive/cav-mae/data/train.json --data-val /content/drive/MyDrive/cav-mae/data/test.json --exp-dir ./exp_test --label-csv /content/drive/MyDrive/cav-mae/data/class_labels_indices_urban.csv --n_class 10 --lr 1e-3 --n-epochs 5 --batch-size 128 --finetuned_path /content/drive/MyDrive/24f-Noise_Robust-AVClassification/exp_test/noise_to_both.pth --proportion 0.3 --dataset_mean 0 --dataset_std 1 --target_length 1024 --mode eval
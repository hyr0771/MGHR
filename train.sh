torchrun --nproc_per_node=2 --master_port=29501 main.py \
  --dataset VQA \
  --checkpoint 4m_base_finetune/vqa/model_state_epoch_9.th \
  --config ./xvlm/configs/VQA_480.yaml \
  --output saved_models/VQA/ddp_tune_xvlm_ce \
  --batch_size 16 --epochs 15 --xvlm_lr 5e-5 --distributed \
  --train_drop_last --mixed_precision --ddp_bucket_cap_mb 100 \
  --tune_xvlm --lambda_xvlm 0.1
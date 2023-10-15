CUDA_VISIBLE_DEVICES=0 python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_sum_top16/" --bs=32 --user_encoder=wo_multi_attn --num_trans_layers=4 --disease=None

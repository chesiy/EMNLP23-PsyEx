# run two-stream PsyEx model in binary setting
CUDA_VISIBLE_DEVICES=0 python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_sum_top16/" --bs=32 --user_encoder=psyex --num_trans_layers=4 --disease=None

# run two-stream PsyEx model in multi_label setting
CUDA_VISIBLE_DEVICES=0 python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_sum_top16/" --model_type=mental/mental-bert-base-uncased --bs=4 --user_encoder=psyex --num_trans_layers=4 --disease=None --setting="multi_label"

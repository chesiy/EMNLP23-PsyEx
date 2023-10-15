# EMNLP23-PsyEx
Code and data of the EMNLP2023 paper "Detection of Multiple Mental Disorders from Social Media with Two-Stream Psychiatric Experts"

First, ```select_posts.py``` select risky posts for further MDD with symptom-based methods. The selected posts are in ```processed/symptom_sum_top16```, which can be provided upon request, please contact chensiyuan925@sjtu.edu.cn or blmoistawinde@qq.com.
 
Next, ```bash runme_psyex.sh``` to run the experiments of PsyEx in both binary and multi-label setting.

Other files:
- data.py: defines the datasets and data module
- model.py: defines the models, including PsyEx and PsyEx without disease-specific attention layers.
- main_hier_clf.py: run experiments with PsyEx models
- runme_psyex_wo_multitask.sh: run the experiment of PsyEx without multi-task learning
- runme_psyex_wo_symp.sh: run the experiment of PsyEx without symptom stream
- runme_psyex_wo_multiattn: run the experiment of PsyEx without multiple attention layers for each disease.

Please don't hesitate to reach out if you encounter any issues with the code or dataset.
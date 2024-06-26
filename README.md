# FITCARL

This is the code for the paper **Improving Few-Shot Inductive Learning on Temporal Knowledge Graphs using Confidence-Augmented Reinforcement Learning** ([paper](https://arxiv.org/abs/2304.00613) (appendices also included)) accepted to ECML/PKDD 2023. 

Datasets are already preprocessed.
Due to file size limit, we put the ent2sec_matrix.npy of ICEWS18 and ICEWS0515 in this [link](https://www.dropbox.com/scl/fo/54e6u7ni2n1magx9r5xz2/AFiiYkSl4rWWEiETxYUyUXY?rlkey=urb5pxwv9m61lodk8pf5l6c14&st=24z930gc&dl=0).

To run 3-shot meta-training on ICEWS14-OOG:

```
python3 main_OOG.py --data_path data/ICEWS14/processed_data --pretrain --emb_nograd --history_encoder gru --score_module att --few 3 --valid_epoch 10 --max_action_num 50 --entity_learner nn --adaptive_sample --sector --save_path logs/icews14_few3 --beam_size 100 --conf --cuda
```

To run 1-shot meta-training on ICEWS14-OOG:

```
python3 main_OOG.py --data_path data/ICEWS14/processed_data --pretrain --emb_nograd --history_encoder gru --score_module att --few 1 --valid_epoch 10 --max_action_num 50 --entity_learner nn --adaptive_sample --sector --save_path logs/icews14_few1 --beam_size 100 --conf --cuda
```

ICEWS18-OOG and ICEWS0515-OOG are only trainable with more than 15GB GPU memory. 

You can also try as you wish. Just change the corresponding arguments from the above examples.

Code is developed based on [TITer](https://github.com/JHL-HUST/TITer/).

#! /bin/bash


### Build customised dictionary 
python3 main_build_dict.py \
	--doc_path 'eval_data/' \
	--dict_path 'dict/DENTAL'


python3 main_train_json.py \
	--doc_path 'eval_data/' \
	--save_prefix 'DENTAL' \
	--test_path '' \
	--embedding_file '' \
	--ckpt_path 'graph/' \
	--tokenize True \
	--update_dict True \
	--load_dict_from_path 'dict/DENTAL' \
	--rows_segment 72 \
	--cols_segment 72 \
	--augment_strategy 1 \
	--positional_mapping_strategy 1 \
	--rows_target 80 \
	--cols_target 80 \
	--rows_ulimit 80 \
	--cols_ulimit 80 \
	--fill_bbox False \
	--data_augmentation_extra True \
	--data_augmentation_dropout 1 \
	--data_augmentation_extra_rows 16 \
	--data_augmentation_extra_cols 16 \
	--batch_size 4 \
	--iterations 10000 \
	--lr_decay_step 13000 \
	--learning_rate 0.0001 \
	--lr_decay_factor 0.1 \
	--hard_negative_ratio 3 \
	--use_ghm 0 \
	--ghm_bins 30 \
	--ghm_momentum 0 \
	--log_path 'log/' \
	--log_disp_step 10 \
	--log_save_step 100 \
	--validation_step 100 \
	--test_step 100 \
	--ckpt_save_step 100 \
	--embedding_size 256 \
	--weight_decay 0.0005 \
	--eps 1e-6

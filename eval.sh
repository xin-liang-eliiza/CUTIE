#! /bin/bash


python3 main_evaluate_json.py \
	--doc_path 'eval_data/' \
	--save_prefix 'DENTAL' \
	--e_ckpt_path 'graph/' \
	--ckpt_file 'CUTIE_atrousSPP_d20000c6(r80c80)_iter_900.ckpt' \
	--load_dict_from_path 'dict/DENTAL' \
	--positional_mapping_strategy 1 \
	--rows_target 80 \
	--cols_target 80 \
	--rows_ulimit 80 \
	--embedding_size 256 \
	--batch_size 8 \
	--c_threshold 0.5 
	

#! /bin/bash


python3 main_evaluate_json.py \
	--doc_path 'invoice_data/' \
	--save_prefix 'INVOICE' \
	--e_ckpt_path 'graph/' \
	--ckpt_file 'CUTIE_atrousSPP_d20000c9(r80c80)_iter_51.ckpt'\
	--positional_mapping_strategy 1 \
	--rows_target 64 \
	--cols_target 64 \
	--rows_ulimit 80 \
	--embedding_size 128 \
	--batch_size 2 \
	

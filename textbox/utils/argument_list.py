# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


general_arguments = ['gpu_id', 'use_gpu',
                     'seed',
                     'reproducibility',
                     'state',
                     'data_path']

training_arguments = ['epochs', 'train_batch_size',
                      'learner', 'learning_rate',
                      'training_neg_sample_num',
                      'eval_step', 'stopping_step',
                      'checkpoint_dir']

evaluation_arguments = ['eval_setting',
                        'group_by_user',
                        'split_ratio', 'leave_one_num',
                        'real_time_process',
                        'metrics', 'topk', 'valid_metric',
                        'eval_batch_size']

dataset_arguments = ['field_separator', 'seq_separator',
                     'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD', 'TIME_FIELD'
                     'seq_len',
                     'LABEL_FIELD', 'threshold',
                     'NEG_PREFIX',
                     'ITEM_LIST_LENGTH_FIELD', 'LIST_SUFFIX', 'MAX_ITEM_LIST_LENGTH', 'POSITION_FIELD',
                     'HEAD_ENTITY_ID_FIELD', 'TAIL_ENTITY_ID_FIELD', 'RELATION_ID_FIELD', 'ENTITY_ID_FIELD',
                     'load_col', 'unload_col', 'additional_feat_suffix',
                     'max_user_inter_num', 'min_user_inter_num', 'max_item_inter_num', 'min_item_inter_num',
                     'lowest_val', 'highest_val', 'equal_val', 'not_equal_val', 'drop_filter_field',
                     'fields_in_same_space', 'fill_nan',
                     'preload_weight', 'drop_preload_weight',
                     'normalize_field', 'normalize_all']

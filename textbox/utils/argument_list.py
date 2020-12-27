# @Time   : 2020/11/14
# @Author : Junyi Li, Gaole He
# @Email  : lijunyi@ruc.edu.cn


general_arguments = ['gpu_id', 'use_gpu',
                     'seed',
                     'reproducibility',
                     'state',
                     'data_path', 'checkpoint_dir', 'generated_text_dir']

training_arguments = ['epochs', 'train_batch_size',
                      'learner', 'learning_rate',
                      'training_neg_sample_num',
                      'eval_step', 'stopping_step']

evaluation_arguments = ['metrics', 'n_grams',
                        'eval_batch_size', 'eval_generate_num']

dataset_arguments = ['max_vocab_size', 'source_max_vocab_size', 'target_max_vocab_size', 
                     'source_max_seq_length', 'target_max_seq_length',
                     'source_language', 'target_language',
                     'source_suffix', 'target_suffix',
                     'split_strategy', 'split_ratio', 
                     'share_vocab']

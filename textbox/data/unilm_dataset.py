from textbox.data.abstract_dataset import AbstractCollate, _pad_sequence
import random
import torch


class UnilmCollate(AbstractCollate):

    def __init__(self, config, tokenizer, set, max_pred_num=20, masked_prob=0.2):
        super(UnilmCollate, self).__init__(config, tokenizer, set)
        self.mask_word_id, self.pad_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[PAD]"])
        self.max_tgt_len = config['tgt_len']
        self.max_src_len = config['src_len']
        self.max_len = self.max_src_len + self.max_tgt_len
        if self.max_len > tokenizer.model_max_length:
            self.max_len = tokenizer.model_max_length
            self.max_tgt_len = self.max_len // 4
            self.max_src_len = self.max_len - self.max_tgt_len
        self._tril_matrix = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))
        self.vocab_size = tokenizer.vocab_size
        self.max_pred_num = max_pred_num
        self.masked_prob = masked_prob

    def __call__(self, samples):
        if self.set == "train":
            source_ids = []
            source_mask = []
            segment_ids = []
            source_text = []
            source_length = []
            target_text = []
            target_ids = []
            masked_ids_list = []
            masked_pos_list = []
            masked_weights_list = []

            for sample in samples:
                src_id = sample['source_ids'].tolist()
                tgt_id = sample['target_ids'].tolist()

                src_len = len(src_id)
                tgt_len = len(tgt_id)
                src_id = src_id + tgt_id
                input_len = len(src_id)

                n_pad = self.max_len - input_len

                src_id = src_id + [0] * n_pad

                second_st, second_end = src_len, input_len
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :src_len].fill_(1)
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end - second_st, :second_end - second_st]
                )

                segment_id = [4] * src_len + [5] * tgt_len + [0] * n_pad

                n_pred = min(self.max_pred_num, max(1, int(round(tgt_len * self.masked_prob))))
                cand_pos = []
                special_pos = set()
                for i, tk_id in enumerate(src_id):
                    if not tk_id:
                        break
                    # only mask tokens_b (target sequence)
                    # we will mask [SEP] as an ending symbol
                    if i >= src_len:
                        cand_pos.append(i)
                    else:
                        special_pos.add(i)
                random.shuffle(cand_pos)

                masked_pos = set()
                max_cand_pos = max(cand_pos)
                for pos in cand_pos:
                    if len(masked_pos) >= n_pred:
                        break
                    if pos in masked_pos:
                        continue

                    st_pos, end_pos = pos, pos + 1

                    for mp in range(st_pos, end_pos):
                        if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                            masked_pos.add(mp)
                        else:
                            break

                masked_pos = list(masked_pos)
                if len(masked_pos) > n_pred:
                    random.shuffle(masked_pos)
                    masked_pos = masked_pos[:n_pred]

                masked_ids = [src_id[pos] for pos in masked_pos]
                for pos in masked_pos:
                    if random.random() < 0.8:  # 80%
                        src_id[pos] = self.mask_word_id
                    elif random.random() < 0.5:  # 10%
                        src_id[pos] = random.randint(1, self.vocab_size - 1)
                # when n_pred < max_pred, we only calculate loss within n_pred
                masked_weights = [1] * len(masked_ids)

                # Zero Padding for masked target
                n_pad = 20 - len(masked_ids)
                if masked_ids is not None:
                    masked_ids.extend([0] * n_pad)
                if masked_pos is not None:
                    masked_pos.extend([0] * n_pad)
                if masked_weights is not None:
                    masked_weights.extend([0] * n_pad)

                source_mask.append(input_mask.tolist())
                segment_ids.append(segment_id)
                source_ids.append(src_id)
                masked_ids_list.append(masked_ids)
                masked_pos_list.append(masked_pos)
                masked_weights_list.append(masked_weights)
                source_text.append(sample['source_text'])
                source_length.append(src_len)
                target_text.append(sample['target_text'])
                target_ids.append(sample['target_ids'])

            model_inputs = {
                'source_ids': torch.tensor(source_ids, dtype=torch.long),
                'source_mask': torch.tensor(source_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
                'masked_lm_labels': torch.tensor(masked_ids_list, dtype=torch.long),
                'masked_pos': torch.tensor(masked_pos_list, dtype=torch.long),
                'masked_weights': torch.tensor(masked_weights_list, dtype=torch.long),
                'source_text': source_text,
                'target_text': target_text,
                'source_length': torch.tensor(source_length, dtype=torch.long),
                'target_ids': _pad_sequence(target_ids, -100, self.tokenizer.padding_side)
            }
            return model_inputs

        else:
            source_ids = []
            source_mask = []
            segment_ids = []
            position_ids = []
            source_text = []
            source_length = []
            target_text = []

            for sample in samples:
                src_id = sample['source_ids'].tolist()
                src_len = len(src_id)
                n_pad = self.max_len - src_len
                n_pad_src = self.max_src_len - src_len
                src_id = src_id + [0] * n_pad_src

                position_id = []
                for i in range(src_len):
                    position_id.append(i)
                for i in range(src_len, self.max_src_len):
                    position_id.append(0)
                for i in range(self.max_src_len, self.max_len):
                    position_id.append(i - self.max_src_len + src_len)

                segment_id = [4] * src_len + [5] * n_pad

                second_st, second_end = self.max_src_len, self.max_len
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                input_mask[:, :src_len].fill_(1)
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end - second_st, :second_end - second_st]
                )

                source_mask.append(input_mask.tolist())
                segment_ids.append(segment_id)
                source_ids.append(src_id)
                position_ids.append(position_id)

            source_mask = torch.tensor(source_mask, dtype=torch.long)
            source_ids = torch.tensor(source_ids, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            position_ids = torch.tensor(position_ids, dtype=torch.long)
            source_text.append(sample['source_text'])
            source_length.append(src_len)
            target_text.append(sample['target_text'])

            model_inputs = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'token_type_ids': segment_ids,
                'position_ids': position_ids,
                'source_text': source_text,
                'target_text': target_text,
                'source_length': torch.tensor(source_length, dtype=torch.long),
            }

            return model_inputs

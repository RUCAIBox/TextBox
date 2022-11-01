import torch
from transformers import EncoderDecoderModel, XLMTokenizer

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-17-1280")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#     "xlm-mlm-17-1280", "xlm-mlm-17-1280"
# ) 

# # training
# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size

input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=input_ids)
# loss, logits = outputs.loss, outputs.logits

# # save and load from pretrained
# model.save_pretrained("xlm2xlm")
model = EncoderDecoderModel.from_pretrained("xlm2xlm")

# generation
generated = model.generate(input_ids)
generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated_text)
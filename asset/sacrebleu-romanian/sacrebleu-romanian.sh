#!/bin/bash

set -eu

REPLACE_UNICODE_PUNCT=asset/sacrebleu-romanian/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=asset/sacrebleu-romanian/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=asset/sacrebleu-romanian/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=asset/sacrebleu-romanian/preprocess/remove-diacritics.py
NORMALIZE_ROMANIAN=asset/sacrebleu-romanian/preprocess/normalise-romanian.py
TOKENIZER=asset/sacrebleu-romanian/scripts/tokenizer/tokenizer.perl

sys=$1
ref=$2

lang=ro
for file in $sys $ref; do
  cat $file \
  | $REPLACE_UNICODE_PUNCT \
  | $NORM_PUNC -l $lang \
  | $REM_NON_PRINT_CHAR \
  | $NORMALIZE_ROMANIAN \
  | $REMOVE_DIACRITICS \
  | $TOKENIZER -no-escape -l $lang \
  > $(basename $file).tok
done

cat $(basename $sys).tok | sacrebleu -tok none -s none -b $(basename $ref).tok

rm $(basename $sys).tok
rm $(basename $ref).tok

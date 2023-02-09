# Dataset

### Dataset List

Now we support 13 generation tasks and their 83 corresponding datasets (the item in the bracket is the name used in `--dataset`):

- Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), WLE (wle), Newsroom (nr), WikiHow (wikihow), MicroSoft News (msn), MediaSum (mediasum), and English Gigaword (eg).
- Machine Translation: WMT14 English-French (wmt14-fr-en), WMT16 Romanian-English (wmt16-ro-en), WMT16 German-English (wmt16-de-en), WMT19 Czech-English (wmt19-cs-en), WMT13 Spanish-English (wmt13-es-en), WMT19 Chinese-English (wmt19-zh-en), and WMT19 Russian-English (wmt19-ru-en).
- Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), SGD (sgd), Topical-Chat (tc), Wizard of Wikipedia (wow), Movie Dialog (md), Cleaned OpenSubtitles Dialogs (cos), Empathetic Dialogues (ed), Curiosity (curio), CMU Document Grounded Conversations (cmudog), MuTual (mutual), OpenDialKG (odkg), and DREAM (dream).
- Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), ToTTo (totto), ENT-DESC (ent), AGENDA (agenda), GenWiki (genwiki), TEKGEN (tekgen), LogicNLG (logicnlg), WikiTableT (wikit), and WEATHERGOV (wg).
- Question generation: SQuAD (squadqg), CoQA (coqaqg), NewsQA (newsqa), HotpotQA (hotpotqa), MS MARCO (marco), MSQG (msqg), NarrativeQA (nqa), and QuAC (quac).
- Story generation: ROCStories (roc), WritingPrompts (wp), Hippocorpus (hc), WikiPlots (wikip), and ChangeMyView (cmv).
- Question answering: SQuAD (squad), CoQA (coqa), Natural Questions (nq), TriviaQA (tqa), WebQuestions (webq), NarrativeQA (nqa), MS MARCO (marco), NewsQA (newsqa), HotpotQA (hotpotqa), MSQG (msqg), and QuAC (quac).
- Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz), MetaLWOZ (metalwoz), KVRET (kvret), WOZ (woz), CamRest676 (camres676), Frames (frames), TaskMaster (taskmaster), Schema-Guided (schema), and MSR-E2E (e2e_msr).
- Chinese generation: LCSTS (lcsts), CSL (csl), and ADGEN (adgen).
- Commonsense generation: CommonGen (cg).
- Paraphrase generation: Quora (quora) and ParaNMT-small (paranmt).
- Text style transfer: GYAFC-E&M and F&R (gyafc-em, gyafc-fr).
- Text simplification: WikiAuto + Turk/ASSET (wia-t).

These datasets can be downloaded at https://huggingface.co/RUCAIBox.

### Leaderboard

For each dataset, we report their details including the dataset description, basic statistics, and training/validation/testing samples. In addition, we build a [leaderboard](https://github.com/RUCAIBox/TextBox/tree/2.0.0/Leaderboard) for each dataset by collecting the automatic results and generated texts of the latest research. We also encourage community users to collaboratively maintain the leaderboard and submit their model results.

### Dataset Parameters

Specify the dataset with`--dataset=<xxx>`. Dataset full list is [here](###Dataset-list).

`src_len`, `tgt_len`, and `truncate` restricts the maximal length of source/target sentence and the positional to be truncated (`head` or `tail`). For some models used for translation task like m2m100, you need to specify source language and target language:

```
# m2m100: en -> zh
src_lang: 'en'
tgt_lang: 'zh'
```

### Prompting

To prompt (instruction) at `prefix` or `suffix`, pass strings to the following parameters:

```
prefix_prompt: 'Summarize: '
suffix_prompt: ' (Write a story)'
```

### New Dataset 

We also support you to run our model using your own dataset. 

Usage

1. Create a new folder under the `dataset` folder (i.e. `dataset/YOUR_DATASET`) to put your own corpus files (`train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, `test.tgt`) which include a sequence per line.
2. Write a YAML configuration file using the same file name to set the hyper-parameters of your dataset, e.g. `textbox/properties/dataset/YOUR_DATASET.yaml`.

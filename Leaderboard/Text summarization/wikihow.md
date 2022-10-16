# WikiHow

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1810.09305)

Repository: [Official](https://github.com/mahnazkoupaee/WikiHow-Dataset)

WikiHow is a dataset of more than 230,000 article and summary pairs extracted and constructed from an online knowledge base written by different human authors. The articles span a wide range of topics and represent high diversity styles.

### Overview

| Dataset | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| ------- | --------- | --------- | -------- | ------------------- | ------------------- |
| WikiHow | $157,252$ | $5,599$   | $5,577$  | $502.6$             | $45.6$              |

### Data Sample

title

```
How to Be an Organized Artist1
```

headline

```
\nKeep related supplies in the same area.,\nMake an effort to clean a dedicated workspace after every session.,\nPlace loose supplies in large, clearly visible containers.,\nUse clotheslines and clips to hang sketches, photos, and reference material.,\nUse every inch of the room for storage, especially vertical space.,\nUse chalkboard paint to make space for drafting ideas right on the walls.,\nPurchase a label maker to make your organization strategy semi-permanent.,\nMake a habit of throwing out old, excess, or useless stuff each month.
```

text

```
o. Paints should be kept with brushes, cleaner, and canvas, print supplies should be by the ink, etc. Make broader groups and areas for your supplies to make finding them easier, limiting your search to a much smaller area. Some ideas include:\n\n\nEssential supplies area -- the things you use every day.\nInspiration and reference area.\nDedicated work area .\nInfrequent or secondary supplies area, tucked out of the way.;\n, This doesn't mean cleaning the entire studio, it just means keeping the area immediately around the desk, easel, pottery wheel, etc. clean each night. Discard trash or unnecessary materials and wipe down dirty surfaces. Endeavor to leave the workspace in a way that you can sit down the next day and start working immediately, without having to do any work or tidying.\n\n\nEven if the rest of your studio is a bit disorganized, an organized workspace will help you get down to business every time you want to make art.\n\n, As visual people, a lot of artist clutter comes from a desire to keep track of supplies visually instead of tucked out of sight. By using jars, old glasses, vases, and cheap, clear plastic drawers, you can keep things in sight without leaving it strewn about haphazardly. Some ideas, beyond those just mentioned, include:\n\n\nCanvas shoe racks on the back of the door\nWine racks with cups in each slot to hold pens/pencils.\nPlastic restaurant squirt bottles for paint, pigment, etc., Simply string up the wires across a wall or along the ceiling and use them to hold essential papers that you don't want to cut or ruin with tacks or tape. Cheap and easy, this is also a good way to handle papers and ideas you touch regularly or need to pin up and down for inspiration., Shelving is an artist's best friend and is a cheap and easy way to get more room in your studio or art space. Don't be afraid to get up high either, especially for infrequently used supplies. The upper reaches of the room are often the most under-utilized, but provide vital space for all your tools and materials., Turning one wall into a chalkboard gives you a perfect space for ideas, sketches, and planning without requiring extra equipment or space. You can even use it for smaller areas. Paint over jars or storage equipment, allowing you to relabel them with chalk as your needs change.\n\n, A lot of disorganization comes when you keep moving the location of things, trying to optimize your space by reorganizing frequently. This usually has the opposite effect, leading to lost items and uncertainty when cleaning, but an afternoon with a label maker can solve everything. Instead of spending all of your mental energy looking for or storing things, you can just follow the labels, freeing your mind to think about art., Once a month, do a purge of your studio. If it isn't essential or part of a project, either throw it out or file it away for later. Artists are constantly making new things, experimenting, and making a mess. This is a good thing, but only if you set aside time to declutter. It may not be fun at the moment, but it is a lot more fun than spending 30 minutes digging through junk to find the right paint or an old sketch.\n\n\nDon't be sentimental here. If you haven't used it in the last six months there is little chance you'll use it in the next six months. Toss it.\n\n"
```

## LeaderBoard

Descending order by ROUGE-2.

| Model                                                        | ROUGE-1 | ROUGE-2 | ROUGE-L | Repository                                             | Generated Text |
| ------------------------------------------------------------ | ------- | ------- | ------- | ------------------------------------------------------ | -------------- |
| [PEGASUS](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf) | $43.06$ | $19.71$ | $34.80$ | [Official](https://github.com/google-research/pegasus) |                |
| [BertSum](https://arxiv.org/pdf/2008.09676.pdf)              | $35.91$ | $13.9$  | $34.82$ |                                                        |                |
| [Transformer](http://proceedings.mlr.press/v119/zhang20ae/zhang20ae.pdf) | $32.48$ | $10.53$ | $23.86$ |                                                        |                |
| [Pointer-generator + coverage](https://arxiv.org/pdf/1810.09305v1.pdf) | $28.53$ | $9.23$  | $26.54$ |                                                        |                |
| [Pointer-generator](https://arxiv.org/pdf/1810.09305.pdf)    | $27.30$ | $9.10$  | $26.54$ |                                                        |                |
| [MatchSum](https://arxiv.org/pdf/2004.08795v1.pdf)           | $31.85$ | $8.98$  | $29.58$ | [Official](https://github.com/maszhongming/MatchSum)   |                |
| [TextRank](https://arxiv.org/pdf/1810.09305.pdf)             | $27.53$ | $7.4$   | $20.00$ |                                                        |                |
| [LEAD3](https://arxiv.org/pdf/1810.09305.pdf)                | $26.00$ | $7.24$  | $24.25$ |                                                        |                |
| [Seq2Seq+Attention](https://arxiv.org/pdf/1810.09305.pdf)    | $22.04$ | $6.27$  | $20.87$ |                                                        |                |

## Citation

```
@article{wikihow,
  title={Wikihow: A large scale text summarization dataset},
  author={Koupaee, Mahnaz and Wang, William Yang},
  journal={arXiv preprint arXiv:1810.09305},
  url={http://arxiv.org/abs/1810.09305},
  year={2018}
}
```
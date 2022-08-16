# MS MARCO

## Dataset

### Instruction

Paper: [Paper](https://arxiv.org/abs/1611.09268)

Homepage: [Official](https://microsoft.github.io/msmarco/)

The MS MARCO (Microsoft MAchine Reading Comprehension) is a collection of datasets focused on deep learning in search. The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Over time the collection was extended with a 1,000,000 question dataset, a natural language generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search.

### Overview

| Dataset  | Num Train | Num Valid | Num Test | Source Length (Avg) | Target Length (Avg) |
| -------- | --------- | --------- | -------- | ------------------- | ------------------- |
| MS MARCO | $681,445$ | $77,580$  | -        | $68.7$              | $13.3$              |

### Data Sample

Answers

```
[ "Approximately $15,000 per year." ]
```

Passages

```
{ 
	"is_selected": [ 1, 0, 0, 0, 0, 0 ], 
	"passage_text": [ "The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed.", "The average revenue in 2011 of a Starbuck Store was $1,078,000, up from $1,011,000 in 2010. The average ticket (total purchase) at domestic Starbuck stores in No … vember 2007 was reported at $6.36. In 2008, the average ticket was flat (0.0% change).", "In fiscal 2014, Walgreens opened a total of 184 new locations and acquired 84 locations, for a net decrease of 273 after relocations and closings. How big are your stores? The average size for a typical Walgreens is about 14,500 square feet and the sales floor averages about 11,000 square feet. How do we select locations for new stores? There are several factors that Walgreens takes into account, such as major intersections, traffic patterns, demographics and locations near hospitals.", "th store in 1984, reaching $4 billion in sales in 1987, and $5 billion two years later. Walgreens ended the 1980s with 1,484 stores, $5.3 billion in revenues and $154 million in profits. However, profit margins remained just below 3 percent of sales, and returns on assets of less than 10 percent.", "The number of Walgreen stores has risen from 5,000 in 2005 to more than 8,000 at present. The average square footage per store stood at approximately 10,200 and we forecast the figure to remain constant over our review period. Walgreen earned $303 as average front-end revenue per store square foot in 2012.", "Your Walgreens Store. Select a store from the search results to make it Your Walgreens Store and save time getting what you need. Your Walgreens Store will be the default location for picking up prescriptions, photos, in store orders and finding deals in the Weekly Ad." ], 
	"url": [ "http://www.indeed.com/cmp/Walgreens/salaries", "http://www.answers.com/Q/What_is_the_average_gross_sales_volume_of_a_single_Walgreen's_Store", "http://news.walgreens.com/fact-sheets/frequently-asked-questions.htm", "http://www.babson.edu/executive-education/thought-leadership/retailing/Documents/walgreens-strategic-evolution.pdf", "http://www.trefis.com/stock/wag/articles/199532/key-trends-driving-walgreens-business/2013-08-07", "http://www.walgreens.com/storelocator/find.jsp?requestType=locator" ] 
}
```

Query

```
walgreens store sales average
```

Query_id

```
9,652
```

## LeaderBoard

Descending order by METRICL.

| Model | METRIC | Repository | Generated Text |
| ----- | ------ | ---------- | -------------- |
|       |        |            |                |
|       |        |            |                |
|       |        |            |                |

## Citation

```
@inproceedings{marco,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  booktitle = {CoCo@NIPS},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {1773},
  publisher = {CEUR-WS.org},
  url={http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf},
  year      = {2016}
}
```

 
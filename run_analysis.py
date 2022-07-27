import argparse
import pickle
import numpy as np
import scipy
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from pyecharts import options as opts
from pyecharts.charts import Page, Grid, Boxplot, WordCloud
from pyecharts.globals import ThemeType, SymbolType
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts


from textbox.config.configurator import Config
from textbox.evaluator import BaseEvaluator
from textbox.data.utils import AbstractDataset

sentspace_database = ['NRC_Valence', 'Concreteness', 'Log_Lexical_Frequency']
default_text_style = {'font_family': 'Times New Roman', 'font_weight':'bold'}

def show_results(models, results, charts):
    table = Table()
    header = ['Models'] + [k.title() for k in results[0].keys()]
    rows = []

    p = [3 for _ in range(len(results[0]))]
    for ri, (model, result) in enumerate(zip(models, results)):
        row = [model]
        if ri + 1 == len(results):
            for i, v in enumerate(result.values()):
                row.append(str(round(np.mean(v), 2)) + '*' * p[i])
        else:
            for i, (v1, v2) in enumerate(zip(result.values(), results[-1].values())):
                row.append(round(np.mean(v1), 2))

                pvalue = scipy.stats.ttest_ind(v2, v1, equal_var=False, alternative='greater').pvalue if isinstance(v1, list) and len(v1) > 1 else 1
                if pvalue < 0.005:
                    p[i] = min(p[i], 3)
                elif pvalue < 0.01:
                    p[i] = min(p[i], 2)
                elif pvalue < 0.05:
                    p[i] = min(p[i], 1)
                else:
                    p[i] = 0
        rows.append(row)

    table.add(header, rows)
    table.set_global_opts(
        title_opts=ComponentTitleOpts(title="Main Results")
    )
    charts.add(table)

def load_lexical_database(features):
    databases = {}
    path='packages/'
    for feature in features:
        if feature in sentspace_database:
            with open(path+feature+'.pkl', 'rb') as f:
                d = pickle.load(f)
                databases[feature] = d
    return databases

def lexical_analysis(models, dataset, corpora, charts):
    models = models + ['Target']
    corpora = corpora + [dataset.target_text]

    max_ngram = 4
    src_sents = [word_tokenize(s) for s in dataset.source_text]
    src_ngrams = [[set([' '.join(n) for n in ngrams(sent, i)]) for sent in src_sents] for i in range(1, max_ngram + 1)]

    features = ['Length', 'NRC_Valence', 'Concreteness', 'Log_Lexical_Frequency', 'Ngram Overlap']
    databases = load_lexical_database(features)
    lemmatizer = WordNetLemmatizer()
    model_features = []
    for texts in corpora:
        feas = []
        sents_tokens = [word_tokenize(t) for t in texts]
        sents_tags = [[tag[0].lower() if tag[0] in ['V', 'N', 'R'] else ('a' if tag[0] == 'J' else '') for _, tag in pos_tag(tokens)] for tokens in sents_tokens]
        sents_lemmas = [[lemmatizer.lemmatize(token, tag) if tag != '' else token for token, tag in zip(tokens, tags)] for tokens, tags in zip(sents_tokens, sents_tags)]
        for feature in features:
            if feature == 'Length':
                feas.append([[len(tokens) for tokens in sents_tokens]])
            elif feature == 'Ngram Overlap':
                fea = []
                for i in range(1, max_ngram + 1):
                    overlaps = []
                    for src_ngram, tgt_tokens in zip(src_ngrams[i - 1], sents_tokens):
                        tgt_ngram = [' '.join(n) for n in ngrams(tgt_tokens, i)]
                        overlap = sum([n in src_ngram for n in tgt_ngram]) / len(tgt_ngram) if len(tgt_ngram) else 0
                        overlaps.append(overlap)
                    fea.append(overlaps)
                feas.append(fea)
            elif feature in sentspace_database:
                fea = []
                feature_dict = databases[feature]
                for tokens, lemmas in zip(sents_tokens, sents_lemmas):
                    f = []
                    for token, lemma in zip(tokens, lemmas):
                        if token in feature_dict:
                            f.append(feature_dict[token])
                        elif lemma in feature_dict:
                            f.append(feature_dict[lemma])
                        else:
                            f.append(0)
                    fea.append(np.mean(f))
                feas.append([fea])
        model_features.append(feas)

    grid = Grid(init_opts=opts.InitOpts(theme=ThemeType.SHINE, width="900px", height="400px"))
    grid_num = 0
    for i, feature in enumerate(features):
        if feature != 'Ngram Overlap':
            chart = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.SHINE))
            chart.add_xaxis([feature])
        else:
            chart = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.SHINE, width="900px", height="400px"))
            chart.add_xaxis([str(i) for i in range(1, max_ngram + 1)])
        for model, features in zip(models, model_features):
            chart.add_yaxis(model, chart.prepare_data(features[i]), tooltip_opts=opts.TooltipOpts(textstyle_opts=opts.TextStyleOpts(color='white', **default_text_style)))
        title_name = 'Lexcial Features' if i == 0 else (feature if feature == 'Ngram Overlap' else '')
        chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title_name, title_textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
            legend_opts=opts.LegendOpts(pos_top=25, textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
            xaxis_opts=opts.AxisOpts(name='Ngram' if feature == 'Ngram Overlap' else '', axislabel_opts=opts.LabelOpts(**default_text_style), name_textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(**default_text_style))
        )
        if feature != 'Ngram Overlap':
            left = round((grid_num / (len(features) - 1)) * 100, 2)
            right = 100 - 100 / (len(features) - 1) - left
            left = str(left + 4) + '%'
            right = str(right) + '%'
            grid_num += 1
            grid.add(chart, grid_opts=opts.GridOpts(pos_left=left, pos_right=right))
        else:
            charts.add(chart)
    charts.add(grid)

def length_metric_analysis(models, dataset, results, charts):
    src_text = dataset.source_text
    tgt_text = dataset.target_text
    src_len = np.array([len(word_tokenize(s)) for s in src_text])
    tgt_len = np.array([len(word_tokenize(t)) for t in tgt_text])

    group_num = 5
    for length, name in zip([src_len, tgt_len], ['Source', 'Target']):
        idx = np.argsort(length)
        idx = np.array_split(idx, group_num)
        x = []
        for i in idx:
            x.append(f"{length[i].min()} - {length[i].max()}")
        for metric in results[0].keys():
            if not isinstance(results[0][metric], list) or len(results[0][metric]) == 1:
                continue
            chart = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.SHINE, width="900px", height="400px"))
            chart.add_xaxis(x)
            for result, model in zip(results, models):
                y = []
                for i in idx:
                    y.append(np.array(result[metric])[i].tolist())
                chart.add_yaxis(model, chart.prepare_data(y), tooltip_opts=opts.TooltipOpts(textstyle_opts=opts.TextStyleOpts(color='white', **default_text_style)))
            chart.set_global_opts(
                title_opts=opts.TitleOpts(title=f"{metric.title()} acc. {name} Length", title_textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
                legend_opts=opts.LegendOpts(pos_top=25, textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
                xaxis_opts=opts.AxisOpts(name='Length', axislabel_opts=opts.LabelOpts(**default_text_style), name_textstyle_opts=opts.TextStyleOpts(**default_text_style)), 
                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(**default_text_style))
            )
            charts.add(chart)

def wordcloud(models, dataset, corpora, charts):
    models = models + ['Target']
    corpora = corpora + [dataset.target_text]
    stops = set(stopwords.words('english'))

    for model, texts in zip(models, corpora):
        tokens = [word_tokenize(t.lower()) for t in texts]
        tokens = sum(tokens, [])
        word_count = list(Counter(tokens).items())
        word_count = [(word, count) for word, count in word_count if word not in stops]
        word_count.sort(key=lambda x: x[1], reverse=True)
        chart = WordCloud()
        chart.add("Count", word_count[:200], word_size_range=[20, 100], tooltip_opts=opts.TooltipOpts(textstyle_opts=opts.TextStyleOpts(**default_text_style)), textstyle_opts=opts.TextStyleOpts(**default_text_style))
        chart.set_global_opts(title_opts=opts.TitleOpts(title=f"WordCloud-{model}", title_textstyle_opts=opts.TextStyleOpts(**default_text_style)))
        charts.add(chart)

def run_analysis(config: Config):
    dataset = AbstractDataset(config, 'test')
    evaluator = BaseEvaluator(config, config["metrics"])
    model_names = []
    generate_corpora = []
    model_results = []
    config['rouge_type'] = 'rouge-score'
    config['corpus_bleu'] = False
    for gen_file in config['gen_files']:
        model_names.append(gen_file.split('/')[-1].split('-')[0])
        with open(gen_file, 'r') as f:
            texts = [t.strip() for t in f]
        generate_corpora.append(texts)
        model_results.append(evaluator.evaluate(texts.copy(), dataset.target_text.copy(), avg=False))
    
    charts = Page(page_title='Analysis', layout=Page.SimplePageLayout)
    show_results(model_names, model_results, charts)
    lexical_analysis(model_names, dataset, generate_corpora, charts)
    length_metric_analysis(model_names, dataset, model_results, charts)
    wordcloud(model_names, dataset, generate_corpora, charts)
    charts.render('test.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='samsum', help='name of datasets')
    parser.add_argument('--gen_files', '-g', nargs='+', default=[], help='list of generated files')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    config_dict = {'gen_files': args.gen_files}
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config = Config(model="", dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)
    
    run_analysis(config)
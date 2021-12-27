from flask import Flask, render_template, request, redirect, make_response, send_file
import os
import pandas as pd
import numpy as np
import models as algorithms
import ploting as plotfun
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask_bootstrap import Bootstrap

app = Flask(__name__)

bootstrap = Bootstrap(app)

def datasetList():
    datasets = [x.split('.')[0] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    folders = [f for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    return datasets, extensions, folders

#Load columns of the dataset
def loadColumns(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'), nrows=0)
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0)
        return df.columns

#Load Dataset
def loadDataset(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        return df


@app.route('/', methods = ['GET', 'POST'])
def index():
    datasets,_,folders = datasetList()
    originalds = []
    featuresds = []
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
        else: featuresds += [datasets[i]]
    if request.method == 'POST':
            f = request.files['file']
            f.save(os.path.join('datasets', f.filename))
            return redirect('/')
    return render_template('index.html', originalds = originalds, featuresds = featuresds)

@app.route('/datasets/')
def datasets():
    return redirect('/')

@app.route('/datasets/<dataset>')
def dataset(description = None, head = None, dataset = None):
    df = loadDataset(dataset)
    try:
        description = df.describe().round(2)
        head = df.head(5)
    except: pass
    return render_template('dataset.html',
                           description = description.to_html(classes='table table-striped table-hover'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)



@app.route('/datasets/<dataset>/preprocessing')
def preprocessing(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('preprocessing.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
def preprocessed_dataset(dataset):
    numFeatures = request.form.get('nfeatures')
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    response = request.form.get('response')
    dropsame = request.form.get('dropsame')
    dropna = request.form.get('dropna')

    df = loadDataset(dataset)

    if dropna == 'all':
        df = df.dropna(axis=1, how='all')
    elif dropna == 'any':
        df.dropna(axis=1, how='any')

    filename = dataset + '_'
    try:
        nf = int(numFeatures)
        from sklearn.feature_selection import SelectKBest, chi2
        X = df.drop(str(response), axis=1)
        y = df[str(response)]
        kbest = SelectKBest(chi2, k=nf).fit(X,y)
        mask = kbest.get_support()
        # List of K best features
        best_features = []
        for bool, feature in zip(mask, list(df.columns)):
            if bool: best_features.append(feature)
        #Reduced Dataset
        df = pd.DataFrame(kbest.transform(X),columns=best_features)
        df.insert(0, str(response), y)

        filename += numFeatures + '_' + 'NA' + dropna + '_Same' + dropsame + '.csv'

    except:
        df = df[manualFeatures]
        filename += str(datasetName) + '_' + str(response) + '.csv'

    if dropsame == 'Yes':
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)
    df.to_csv(os.path.join('preprocessed', filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])


@app.route('/datasets/<dataset>/graphs')
def graphs(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('graphs.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/graphprocess/', methods=['POST'])
def graph_process(dataset = dataset):
    histogram = request.form.getlist('histogram')
    boxplotcat = request.form.get('boxplotcat')
    boxplotnum = request.form.get('boxplotnum')
    corr = request.form.getlist('corr')
    corrcat = request.form.get('corrcat')

    if corrcat != '': corr += [corrcat]
    ds = loadDataset(dataset)
    import plotfunctions as plotfun
    figs = {}
    if histogram != [''] and histogram != []:
        figs['Histograms'] = str(plotfun.plot_histsmooth(ds, histogram), 'utf-8')
    if corr != [''] and corr != []:
        figs['Correlations'] = str(plotfun.plot_correlations(ds, corr, corrcat), 'utf-8')
    if boxplotcat != '' and boxplotnum != '':
        figs['Box Plot'] = str(plotfun.plot_boxplot(ds, boxplotcat, boxplotnum), 'utf-8')
    if figs == {}: return redirect('/datasets/' + dataset + '/graphs')
    return render_template('drawgraphs.html', figs = figs, dataset = dataset)

if __name__ == '__main__':
    app.run(debug=False)

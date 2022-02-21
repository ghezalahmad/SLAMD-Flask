from flask import Flask, render_template, request, redirect, make_response, send_file
import os
import pandas as pd
import numpy as np
#import models as algorithms
import discovery as algorithms
import ploting as plotfun
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask_bootstrap import Bootstrap
from slamd import *

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
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
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0, sep=",")
        return df.columns

#Load Dataset
def loadDataset(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), sep=",")
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
                           description = description.to_html(classes='table table-striped table-hover card-body'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)

@app.route('/datasets/<dataset>/models')
def models(dataset = dataset):
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    #predmodels = algorithms.regressionModels()
    return render_template('models.html', dataset = dataset,
                           clfmodels = clfmodels,
                           #predmodels = predmodels,
                           columns = columns)


@app.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
def model_process(dataset = dataset):
    algscore = request.form.get('model')
    res = request.form.get('response')
    strategy = request.form.get('strategies')
    variables = request.form.getlist('variables')

    df = loadDataset(dataset)
    y = df[str(res)]
    print(y)

    if variables != [] and '' not in variables: df = df[list(set(variables + [res]))]
    X = df.drop(str(res), axis=1)
    try: X = pd.get_dummies(X)
    except: pass

    predictors = X.columns
    if len(predictors)>10: pred = str(len(predictors))
    else: pred = ', '.join(predictors)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn import preprocessing

    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gpr.fit(X, y)
    Expected_Pred, Uncertainty= gpr.predict(X, return_std=True)
    y_samples = gpr.sample_y(X, n_samples=5)
    print('y_samples', y_samples)
    Expected_Pred = pd.DataFrame(Expected_Pred.squeeze())
    Uncertainty = pd.DataFrame(Uncertainty.squeeze())

    ep = Expected_Pred.set_axis(['Prediction'], axis=1)
    un = Uncertainty.set_axis(['Uncertainty'], axis=1)
    var_data = pd.concat([X, y], axis=1)
    pre_data = pd.concat([un, ep], axis=1)
    all_data = pd.concat([var_data, pre_data], axis=1)
    ser = Expected_Pred + Uncertainty
    # Normalize the utility
    if strategy == 'MEI (exploit)':
        scaler = preprocessing.StandardScaler().fit(ser)
        ser_scaled = scaler.transform(ser)
        pdscaled = pd.DataFrame(data=ser_scaled)
        pdscaled = pdscaled.set_axis(['Utility'], axis=1)
        print(type(ser_scaled))
    fig = plotfun.gpr_graph()
    all_data = pd.concat([all_data, pdscaled], axis=1)
    all_data = all_data.head(10)


    return render_template('scores.html', dataset = dataset, algscore=algscore, res = res, gpr=gpr,
         all_data=all_data.to_html(classes='table table-striped table-hover card-body'), fig=fig)
    #,   #kfold = kfold, response = str(fig, 'utf-8'))

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
    scatter = request.form.getlist('scatter1')
    scat = request.form.getlist('scatter2')
    print(histogram)
    if corrcat != '': corr += [corrcat]
    ds = loadDataset(dataset)
    import ploting as plotfun
    figs = {}
    if histogram != [''] and histogram != []:
        figs['Histograms'] = str(plotfun.plot_histsmooth(ds, histogram), 'utf-8')
    if corr != [''] and corr != []:
        figs['Correlations'] = str(plotfun.plot_correlations(ds, corr, corrcat), 'utf-8')
    if boxplotcat != '' and boxplotnum != '':
        figs['Box Plot'] = str(plotfun.plot_boxplot(ds, boxplotcat, boxplotnum), 'utf-8')

    if scatter != [''] and scat !=[]:
        figs['scatter'] = (plotfun.plot_scatter(ds, scatter, scat), 'utf-8')
    if figs == {}: return redirect('/datasets/' + dataset + '/graphs')


    return render_template('drawgraphs.html', figs = figs, dataset = dataset)


################### SEQUENCIAL LEARNING ###############################3####

@app.route('/datasets/<dataset>/sequential')
def sequential(dataset):
    columns = loadColumns(dataset=dataset)
    return render_template('sequential.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/sequentialprocess', methods=['POST', 'GET'])
def sequential_process(dataset=dataset):
    features = request.form.getlist('features')
    fixedtargets = request.form.getlist('fixedtargets')

    targets = request.form.getlist('targets')
    initial_sample = request.form.get('initial_sample')
    iterations = request.form.get('iterate')
    models = request.form.get('models')
    #strategy = request.form.get('strategy')
    dist = request.form.get('dist')

    min_or_max_target = {}
    for t in targets:
        x = 'R_'+t
        min_or_max_target[t]= request.form.get(x)
    print(min_or_max_target)
    print(targets)

    check_to_use_threshold = {}
    for t in targets:
        x = 'C_'+t
        check_to_use_threshold[t]= request.form.get(x)
    print(check_to_use_threshold)


    target_selected_number1 = {}
    for t in targets:
        x = 'N1_'+t
        target_selected_number1[t]= request.form.get(x)
    print(target_selected_number1)

    target_selected_number2 = {}
    for t in targets:
        x = 'N2_'+t
        target_selected_number2[t]= request.form.get(x)
    print(target_selected_number2)


#---------------------------------
    min_or_max_fixedtarget = {}
    for t in fixedtargets:
        x = 'R1_'+t
        min_or_max_fixedtarget[t]= request.form.get(x)
    print(min_or_max_fixedtarget)
    print(targets)

    check_to_use_threshold_ft = {}
    for t in fixedtargets:
        x = 'C1_'+t
        check_to_use_threshold_ft[t]= request.form.get(x)
    print(check_to_use_threshold_ft)


    fixedtarget_selected_number1 = {}
    for t in fixedtargets:
        x = 'N11_'+t
        fixedtarget_selected_number1[t]= request.form.get(x)
    print(fixedtarget_selected_number1)

    fixedtarget_selected_number2 = {}
    for t in fixedtargets:
        x = 'N22_'+t
        fixedtarget_selected_number2[t]= request.form.get(x)
    print(fixedtarget_selected_number2)

    dataset = loadDataset(dataset)

    features = dataset[features]
    target_selction = dataset.columns[~dataset.columns.isin(features)]


    fixedtargets = dataset[fixedtargets]
    target_name = targets
    targets = dataset[targets]


    initial_sample_size = int(initial_sample)
    #print('initial_sample', initial_sample_size)

    iterationen = int(iterations)
    #print('iterations', iterationen)

    #initial_sample_size=4 # Done
    dist= dist # range 1 - 100 -
    target_quantile = 80
    sample_quantile=50# smaller than the target qualtile # not neccessary
    #iterationen=3 # Done
    std=2 # sigma factor
    #dist=1 # it is for MEID, MLID only. # prediction_quantile
    model=None
    strategy='MEI (exploit)'
    #print(type(strategy))

    s = sequential_learning(dataset,initial_sample_size,target_quantile,iterationen,sample_quantile,std,dist,model,
                            strategy, features, targets, fixedtargets, target_name)
    if models == "Decision Trees (DT)":
        dt=DT(models,s,targets)
        s.model=dt
        s = s.main()
    elif models == "lolo Random Forrest (RF)":
        rf = RF(models, s, targets)
        s.model=rf
    elif models == "Random Forrest (RFscikit)":
        rfscikit = RFscikit(models, s, targets)
        s.model=rfscikit
    elif models == "Gaussian Process Regression (GPR)":
        gpr = GPR(models, s, targets)
        s.model=gpr
    else:
        print('Select a model')


    return render_template('sequential.html', s=s, features=features, fixedtargets=fixedtargets, targets=targets,
                                                    dataset=dataset,target_selection=target_selection)

################### SEQUENCIAL LEARNING ###############################3####



@app.route('/datasets/<dataset>/tutorial')
def tutorial():
    return render_template('tutorial.html')


@app.errorhandler(500)
def internal_error(e):
    return render_template('error500.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error404.html')



if __name__ == '__main__':
    app.jinja_env.auto_reload=True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)

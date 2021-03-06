from flask import Flask, render_template, request, redirect, make_response, send_file
import os
import pandas as pd
import numpy as np
#import models as algorithms
#import discovery as algorithms
import ploting as plotfun
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from flask_bootstrap import Bootstrap
from slamd import *
from material_discovery import *
import plotly.graph_objects as go



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
    head_df = pd.DataFrame(df)
    try:
        description = df.describe().round(2)
        #head = df.head(5)
        des = head_df.describe().round(2)

    except: pass
    return render_template('dataset.html', head_df=head_df, des = des, dataset=dataset)





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
    plot_name = request.form.get('plot_name')
    x_ax = request.form.get('x_axis')
    y_ax = request.form.get('y_axis')
    hue = request.form.get('hue')
    size = request.form.get('size')
    #scatter = request.form.get('sc') # only scatter
    #scatter_matrix = request.form.getlist('sm')
    #heatmap = request.form.getlist('hm')
    attribute_feature = request.form.getlist('attribute_feature')

    #scat = request.form.getlist('scatter2')

    ds = loadDataset(dataset)
    #----Scatter plot paramter -------------------------------


    #----------------------------------------------------------------

    #heatmap = ds[attribute_feature]


    #columns = loadColumns(dataset)
    attribute_f = ds[attribute_feature]
    import json
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go

    if plot_name == 'sc':
        x_axis = ds[x_ax]
        x_axis = x_axis.to_numpy()
        x_axis = x_axis.flatten()

        y_axis = ds[y_ax]
        y_axis= y_axis.to_numpy()
        y_axis = y_axis.flatten()

        hue = ds[hue]
        hue = hue.to_numpy()
        hue = hue.flatten()

        size = ds[size]
        size = size.to_numpy()
        size = size.flatten()
        ds = ds.to_numpy()
        ds = ds.flatten(order='C')
        fig =  px.scatter(ds, x=x_axis, y=y_axis, color=hue, size=size, marginal_x="rug", marginal_y="box", title="Scatter plot with margin plots")
        fig.update_layout(
            dragmode='select',
            width=1600,
            height=1000,
            hovermode='closest',
        )
        #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    elif plot_name == 'hm':
        df = attribute_f.corr()
        fig = px.imshow(df, aspect="auto")#x = df.columns, y = df.index, z = np.array(df))
        #fig = fig.show()
        #graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        fig.update_layout(
            title='Correlation Heatmap',
            dragmode='select',
            width=1100,
            height=1100,
            hovermode='closest',

        )

    elif plot_name == 'sm':
        print('att col', attribute_f.columns)
        col = attribute_f.columns[0]
        fig = px.scatter_matrix(attribute_f)
        fig.update_layout(
            title='Scatter Matrix',
            dragmode='select',
            width=2000,
            height=2000,
            hovermode='closest',

        )





    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('drawgraphs.html', graphJSON=graphJSON, fig=fig)


"""
@app.route('/datasets/<dataset>/graphprocess/', methods=['POST'])
def heatmap_corr(dataset = dataset):
    scatter_matrix = request.form.getlist('sm')
    attribute_feature = request.form.getlist('attribute_feature')
    ds = loadDataset(dataset)
    smatrix = ds[attribute_feature]

    if scatter_matrix != ['']:
        df = ds.corr()
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(x = df.columns,y = df.index,z = np.array(df))
        )
        fig = fig.show()
    return render_template('drawgraphs.html', fig=fig)"""
################### SEQUENCIAL LEARNING ###############################3####

@app.route('/datasets/<dataset>/sequential')
def sequential(dataset):
    columns = loadColumns(dataset=dataset)
    return render_template('sequential.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/sequentialprocess', methods=['POST', 'GET'])
def sequential_process(dataset=dataset):
    dataframe = loadDataset(dataset)
    initial_sample_size = int(request.form.get('initial_sample_size'))
    batch_size = int(request.form.get('batch_size'))
    target_treshhold = int(request.form.get('target_treshhold'))
    number_of_executions = int(request.form.get('number_of_executions'))
    sigma = request.form.get('sigma_factor')
    model = request.form.get('models')
    strategy = request.form.get('strategy')
    target_df = request.form.getlist('targets')
    fixed_targets_idx = request.form.getlist('fixedtargets')
    feature_df = request.form.getlist('features')
    tquantile = int(request.form.get('tquantile'))





# --- This is the min_max of benchmarking ---------
    min_or_max_target = {}
    for t in target_df:
        x = 'R_'+t
        min_or_max_target[t]= request.form.get(x)


    check_to_use_threshold_t = {}
    for t in target_df:
        x = 'C_'+t
        check_to_use_threshold_t[t]= request.form.get(x)


    target_selected_number1 = {}
    for t in target_df:
        x = 'N1_'+t
        target_selected_number1[t]= request.form.get(x)

    target_selected_number2 = {}
    for t in target_df:
        x = 'N2_'+t
        target_selected_number2[t]= request.form.get(x)


#---------------------------------
    min_or_max_fixedtarget = {}
    for t in fixed_targets_idx:
        x = 'R1_'+t
        min_or_max_fixedtarget[t]= request.form.get(x)


    check_to_use_threshold_ft = {}
    for t in fixed_targets_idx:
        x = 'C1_'+t
        check_to_use_threshold_ft[t]= request.form.get(x)


    fixedtarget_selected_number1 = {}
    for t in fixed_targets_idx:
        x = 'N11_'+t
        fixedtarget_selected_number1[t]= request.form.get(x)

    fixedtarget_selected_number2 = {}
    for t in fixed_targets_idx:
        x = 'N22_'+t
        fixedtarget_selected_number2[t]= request.form.get(x)
# ------------------------------------------------



    feature_df = dataframe[feature_df]
    #target_selction = dataframe.columns[~dataframe.columns.isin(features)]
    #fixed_targets_idx = dataframe[fixed_targets_idx]
    #target_name = targets
    #target_df = dataframe[target_df]

    #initial_sample_size = int(initial_sample_size)
    #print('initial_sample', initial_sample_size)


    #print('iterations', iterationen)

    #initial_sample_size=4 # Done
    #dist= dist # range 1 - 100 -
    #target_quantile = 80
    #sample_quantile=50# smaller than the target qualtile # not neccessary
    #iterationen=3 # Done
    #std=2 # sigma factor
    dist=1 # it is for MEID, MLID only. # prediction_quantile
    #model=None
    #strategy='MEI (exploit)'
    #print(type(strategy))





    s = sequential_learning(dataframe,initial_sample_size,batch_size, target_treshhold, number_of_executions,
             sigma, dist, model, strategy, target_df, fixed_targets_idx, feature_df, min_or_max_target, check_to_use_threshold_t,
             target_selected_number1,target_selected_number2, min_or_max_fixedtarget, check_to_use_threshold_ft,
             fixedtarget_selected_number1, fixedtarget_selected_number2, tquantile)

    s.main()
    #plt = s.start_sequential_learning()
    """
    if model == "Decision Trees (DT)":
        dt=DT(model,s,targets)
        s.model=dt
        s = s.main()
    elif model == "lolo Random Forrest (RF)":
        rf = RF(model, s, targets)
        s.model=rf
    elif model == "Random Forrest (RFscikit)":
        rfscikit = RFscikit(model, s, targets)
        s.model=rfscikit
    elif model == "Gaussian Process Regression (GPR)":
        gpr = GPR(model, s, targets)
        s.model=gpr
    else:
        print('Select a model')
    """

    return render_template('sequential.html', s=s, dataset=dataset, plt=plt)

################### SEQUENCIAL LEARNING ###############################3####
@app.route('/datasets/<dataset>/models')
def models(dataset = dataset):
    columns = loadColumns(dataset)
    #clfmodels = algorithms.classificationModels()
    #predmodels = algorithms.regressionModels()
    return render_template('models.html', dataset = dataset, columns = columns)

@app.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
def model_process(dataset = dataset):
    model = request.form.get('models')
    target_df = request.form.getlist('targets')
    feature_df = request.form.getlist('feature_df')
    fixed_target_df = request.form.getlist('fixedtargets')
    strategy = request.form.get('strategies')
    #distance = request.form.get('initial_sample')
    sigma = float(request.form.get('sigma_factor'))

    dataframe = loadDataset(dataset)

# --- This is the min_max of benchmarking ---------
    min_or_max_target = {}
    for t in target_df:
        x = 'Rd_'+t
        min_or_max_target[t]= request.form.get(x)

    target_selected_number2 = {}
    for t in target_df:
        x = 'Nd_'+t
        target_selected_number2[t]= int(request.form.get(x))
#---------------------------------
    min_or_max_fixedtarget = {}
    for t in fixed_target_df:
        x = 'Rd1_'+t
        min_or_max_fixedtarget[t]= request.form.get(x)


    fixedtarget_selected_number2 = {}
    for t in fixed_target_df:
        x = 'Nd1_'+t
        fixedtarget_selected_number2[t]= int(request.form.get(x))
# ------------------------------------------------

    l = learn(dataframe, model, target_df, feature_df, fixed_target_df, strategy, sigma, target_selected_number2, fixedtarget_selected_number2, min_or_max_target, min_or_max_fixedtarget)
    l.start_learning()
    n = l.start_learning()

    df_table = pd.DataFrame(n)
    df_column = df_table.columns
    #df_table2 = df_table1[1:]
    print(df_column)
    df_only_data = df_table
    print('new_df', df_only_data)
    import seaborn as sns
    import matplotlib.pyplot as plt
    print('df column', df_only_data.columns)


    return render_template('scores.html', dataset=dataset,df_column=df_column,  df_only_data=df_only_data, n=n.to_html(index=False, classes='table table-striped table-hover table-responsive', escape=False))

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

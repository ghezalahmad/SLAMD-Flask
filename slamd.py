import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as SKRFR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from lolopy.learners import RandomForestRegressor

from app import *


#df=pd.read_csv(r'datasets\fi.csv',error_bad_lines=False,sep =",",decimal=".")
#features= pd.core.indexes.base.Index(['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'SO3',
#       'TiO2', 'P2O5', 'SrO', 'Mn2O3', 'LOI', 'Fine modulus (m2/kg)',
#       'Specific gravity', 'Cement (kg/m3)', 'FA (kg/m3)', 'GGBFS (kg/m3)',
#       'SF (kg/m3)', 'Kaolin (kg/m3)', 'Other SCM (kg/m3)', 'Na2O (l)',
#       'Sio2 (l)', 'H2O', 'Superplasticizer', 'water -eff',
#       'Initial curing time (day)', 'Initial curing temp (C)',
#       'Initial curing rest time (day)', 'Final curing temp (C)',
#       'Cube D (mm)', 'f_c 28d (MPA)'],
#      dtype='object')
#targets = pd.core.indexes.base.Index(['f_c 28d (MPA)'],
#                                     dtype='object')
optimization_targets = ['max','min']
#fixedtargets = pd.core.indexes.base.Index(['Final curing temp (C)'], dtype='object')
optimization_fixed_targets=['min']



#Utility Methods
def decide_max_or_min(optimization,columns,dataframe):
        for goal in optimization:
                    for column in columns:
                            if (goal == "min"):
                                dataframe[column]=dataframe[column]*(-1)

def extend(list_of_2dms_arrays_to_extend):
    np_array=np.array(list_of_2dms_arrays_to_extend)
    max_cols=max(map(len,np_array))
    result_list=[]
    for i in np_array:
                    if(len(i) == max_cols):
                        result_list.append(i)
                    elif (len(i) != max_cols):
                        how_often=max_cols-len(i)
                        matrix_to_extend=np.tile(i[:][-1], (how_often, 1))
                        i=np.concatenate((i, matrix_to_extend))
                        result_list.append(i)


    return result_list

result_df = pd.DataFrame(columns=['Requ. experiments (mean)','Requ. experiments (std)',
                                  'Requ. experiments (90%)','Requ. experiments (max)',
                                  'Algorithm','Utlity Function','σ Factor',
                                  '5 cycle perf.','10 cycle perf.',
                                  'qant. (distance utility)','# SL runs',
                                  'Initial Sample','# of samples in the DS',
                                  '# Features','# Targets',
                                  'Target threshold','Sample threshold',
                                  'Features name','Targets name',
                                  'Req. experiments (all)'])



from abc import ABC, abstractmethod, ABCMeta

class Model(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, purchase):
        pass


class DT(Model):
    def __init__(self,name,sequential_learning,targets):
        self.name=name
        self.sequential_learning=sequential_learning
        self.targets=targets


    def fit(self):
        td,tl=self.sequential_learning.jk_resampling()
        y_pred=[]

        for i in range(len(td)):
            dtr = DecisionTreeRegressor()
            dtr.fit(td[i], tl[i])
            y_pred.append(dtr.predict(self.sequential_learning.features_df.iloc[self.sequential_learning.PredIdx]))


        #quick bug fix
        if(self.sequential_learning.strategy=="MEID (exploit)"):
            y_pred=np.array(y_pred)
            Expected_Pred = y_pred.mean(axis=0)
            Uncertainty = y_pred.std(axis=0)
        elif(self.sequential_learning=="MLID (explore & exploit)"):
            y_pred=np.array(y_pred)
            Expected_Pred = y_pred.mean(axis=0)
            Uncertainty = y_pred.std(axis=0)
        else:
            y_pred=np.array(y_pred)
            y_pred=y_pred.T
            Expected_Pred = y_pred.mean(axis=1)
            Uncertainty = y_pred.std(axis=1)

        return Expected_Pred, Uncertainty

class RF(Model):
    def __init__(self,name,sequential_learning,targets):
        self.name=name
        self.sequential_learning=sequential_learning
        self.targets=targets


    def fit(self):
        dtr = RandomForestRegressor()
        dtr.fit(self.features_df.iloc[self.sequential_learning.SampIdx].to_numpy(), self.sequential_learning.dataframe[targets].iloc[self.sequential_learning.SampIdx].sum(axis=1).to_frame().to_numpy())
        self.Expected_Pred, self.Uncertainty = dtr.predict(self.sequential_learning.features_df.iloc[self.sequential_learning.PredIdx].to_numpy(), return_std=True)

class RFscikit(Model):

    def __init__(self,name,sequential_learning,targets):
        self.name=name
        self.sequential_learning=sequential_learning
        self.targets=targets



    def fit(self):
        td,tl=self.sequential_learning.jk_resampling()
        self.y_pred_dtr=[]


        for i in range(len(td)):
            ## alternative Ensamble Learners below:
            dtr = SKRFR (n_estimators=10)
            dtr.fit(td[i], tl[i])
            self.y_pred_dtr.append(dtr.predict(self.sequential_learning.features_df.iloc[self.sequential_learning.PredIdx]))


        if(self.sequential_learning.strategy=="MEID (exploit)"):
            y_pred=np.array(y_pred_dtr)
            Expected_Pred = y_pred.mean(axis=0)
            Uncertainty = y_pred.std(axis=0)
        elif(self.sequential_learning=="MLID (explore & exploit)"):
            y_pred=np.array(y_pred_dtr)
            Expected_Pred = y_pred.mean(axis=0)
            Uncertainty = y_pred.std(axis=0)
        else:
            y_pred=np.array(y_pred_dtr)
            y_pred=y_pred.T
            Expected_Pred = y_pred.mean(axis=1)
            Uncertainty = y_pred.std(axis=1)

        return Expected_Pred, Uncertainty

class GPR(Model):

    def __init__(self,name,sequential_learning,targets):
        self.name=name
        self.sequential_learning=sequential_learning
        self.targets=targets

    def fit(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        dtr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        dtr.fit(self.sequential_learning.features_df.iloc[self.sequential_learning.SampIdx].to_numpy(), self.sequential_learning.dataframe[targets].iloc[self.sequential_learning.SampIdx].sum(axis=1).to_frame().to_numpy())
        self.Expected_Pred, self.Uncertainty= dtr.predict(self.features_df.iloc[self.PredIdx], return_std=True)
        return Expected_Pred.squeeze(), Uncertainty.squeeze()

class sequential_learning:

    min_distances_list=[]
    y_pred_dtr_mean=None
    y_pred_dtr_std=None
    y_pred_dtr=None
    SampIdx=None
    PredIdx=None
    count=0
    index_sum_randomized=None
    rand_tars=[]
    rand_fixed_tars=[]

    def __init__(self,dataframe,init_sample_size,target_treshhold,number_of_executions,sample_treshold,sigma,distance,model,
                                                strategy, feature, targets, fixedtargets, target_name):  #constructor
        self.dataframe= dataframe
        self.feature=feature
        self.target_name = target_name
        self.targets = targets
        self.fixedtargets=fixedtargets
        self.init_sample_size=init_sample_size
        self.target_treshhold = target_treshhold/100
        self.sample_treshold=sample_treshold/100
        self.number_of_executions=number_of_executions
        self.tries_list=np.empty(number_of_executions)
        self.tries_list_rand_pick=np.empty(number_of_executions)
        self.sigma=sigma
        self.distance=distance
        self.model=model
        self.strategy = strategy
        self.apply_feature_selection_to_df(self.dataframe)
        #print('self.feature', self.feature)
        self.apply_target_selection_to_df(self.dataframe)
                #print('self.dataframe', self.dataframe)
        #print('self.fixedtargets', self.fixedtargets)
        if(len(self.fixedtargets.values.tolist())>0):
            self.target_df= pd.concat([self.target_df, self.fixedtargets], axis=1)   #self.target_df.join(self.fixedtargets)
            print('self.target_df', self.target_df)
        self.standardize_data()
        init_sample_set=self.init_sampling()


    def apply_feature_selection_to_df(self,dataframe):
        self.features_df = self.feature

    def apply_target_selection_to_df(self,dataframe):
        self.target_df= self.targets

    def standardize_data(self):
        dataframe_norm=(self.dataframe-self.dataframe.mean())/self.dataframe.std()
        target_df_norm=(self.target_df-self.target_df.mean())/self.target_df.std()
        features_df_norm=(self.features_df-self.features_df.mean())/self.features_df.std()
        self.features_df=features_df_norm
        self.target_df=target_df_norm
        self.dataframe=dataframe_norm





    def init_sampling(self):
        s = pd.concat([self.dataframe[self.target_name], self.fixedtargets], axis=1)
        print('s',s)
        sum_ = s.sum(axis=1)
        #sum_ = self.dataframe.loc[self.targets]


        #sum_ = sum_.sum(axis=1)

        print('sum_', sum_)
        samp_q_t=sum_.quantile(self.sample_treshold) #target threshold here
        Index_label=np.where(sum_ < samp_q_t )
        Index_label=Index_label[0]
        init_sample_set = np.ones((0,self.init_sample_size))

        for i in range(self.number_of_executions):

                    init_sample_set=np.vstack([init_sample_set, np.random.choice(Index_label,self.init_sample_size)])

        return init_sample_set

    def start_sequential_learning(self):
            import time
            self.tries_list=np.empty(self.number_of_executions)
            self.tries_list.fill(np.nan)
            self.tries_list_rand_pick=np.empty(self.number_of_executions)
            self.tries_list_rand_pick.fill(np.nan)
            self.count=0

            distances=[]
            targt_perfs=[]

            fixed_targets=[]
            targets_as_list=[]

            current_distances_list=[]
            current_targt_perf_list=[]

            print('Sequential Learning is running...')

            global result_df

            #decide_max_or_min(optimization_targets,self.dataframe.columns,self.dataframe)
            #decide_max_or_min(optimization_fixed_targets,self.fixedtargets,self.dataframe)

            init_sample_set=self.init_sampling()
            s = pd.concat([self.dataframe[self.target_name], self.fixedtargets], axis=1)
            print('s',s)
            sum_ = s.sum(axis=1)

            fixed_targets_index=self.fixedtargets
            print(type(fixed_targets_index))
            s2 = pd.concat([self.dataframe[self.target_name], fixed_targets_index.to_frame()], axis=1)
            print('s2', s2)
            sum_ = s2.sum(axis=1)
            #sum_ = self.dataframe[self.target_name].sum(axis=1).to_frame()+self.dataframe[fixed_targets_index].sum(axis=1).to_frame()

            targ_q_t= sum_.quantile(self.target_treshhold)
            schwellwert=sum_.quantile(self.target_treshhold)
            Index_c=np.where(sum_ >= schwellwert )
            Index_c=Index_c[0]

            for i in range(self.number_of_executions):

                    self.perform_random_pick(i)
                    self.SampIdx=init_sample_set[i].astype(int)
                    self.PredIdx=self.dataframe
                    self.PredIdx = self.PredIdx.drop(self.PredIdx.index[self.SampIdx]).index

                    self.Expected_Pred, self.Uncertainty=self.model.fit()

                    self.tries_list[i]=self.init_sample_size
                    distance=distance_matrix(self.dataframe.loc[self.SampIdx],self.dataframe.iloc[Index_c])
                    distance=distance.min()
                    current_distances_list=[distance]


                    targt_perf=sum_.loc[self.SampIdx].max().item()
                    current_targt_perf_list=[targt_perf]

                    max_targt_perf_index=np.argmax(sum_.loc[self.SampIdx].values, axis=0)
                    Idx_of_best_value=self.SampIdx[max_targt_perf_index]
                    best_value=self.dataframe.iloc[Idx_of_best_value]


                    #current_fixed_target_list=np.array(best_value[self.fixedtargets].to_numpy()[0])
                    #current_prediction_target=np.array(best_value[self.targets].to_numpy()[0])
                    current_fixed_target_list=np.array([self.fixedtargets][0])
                    current_prediction_target=np.array([self.targets][0])



                    while sum_.loc[self.SampIdx].max() < targ_q_t:

                                    self.update_strategy(self.strategy)

                                    #Train Model
                                    self.Expected_Pred, self.Uncertainty=  self.model.fit()

                                    schwellwert=sum_.quantile(self.target_treshhold)
                                    Index_c=np.where(sum_ >= schwellwert )
                                    Index_c=Index_c[0]
                                    distance= distance_matrix(self.dataframe.loc[self.SampIdx],self.dataframe.iloc[Index_c])
                                    distance=distance.min()
                                    current_distances_list.append(distance)

                                    #targt_perf=sum_.loc[self.SampIdx].max().values.tolist()
                                    targt_perf=sum_.loc[self.SampIdx]
                                    targt_perf=max(targt_perf)

                                    current_targt_perf_list.append(targt_perf)


                                    max_targt_perf_index=np.argmax(sum_.loc[self.SampIdx].values, axis=0)
                                    Idx_of_best_value=self.SampIdx[max_targt_perf_index]
                                    best_value=self.dataframe.iloc[Idx_of_best_value]

                                    #current_prediction_target=np.vstack([current_prediction_target,best_value[targets].to_numpy()[0]])
                                    #current_fixed_target_list=np.vstack([current_fixed_target_list,best_value[fixedtargets].to_numpy()[0]])
                                    current_prediction_target=np.vstack([current_prediction_target,self.targets])[0]
                                    current_fixed_target_list=np.vstack([current_fixed_target_list,self.fixedtargets])[0]


                                    self.tries_list[i]=self.tries_list[i]+1

                    distances.append(current_distances_list)
                    targt_perfs.append(current_targt_perf_list)
                    fixed_targets.append(current_fixed_target_list)
                    targets_as_list.append(current_prediction_target)




                ##Distance Plot

                    fig1,axs = plt.subplots(1,2,figsize=(15, 6))
                    axs[0].set_title('Optimization progress in input space')
                    axs[0].set_xlabel('development cycles')
                    axs[0].set_ylabel("Minimum distance from sampled data to target")
                    axs[0].axhline(y=0, color='k', linestyle=':',label='Target')
                    axs[0].legend()

                    axs[1].set_title('Optimization progress in output space')
                    axs[1].set_xlabel('development cycles')
                    axs[1].set_ylabel("Maximum sampled property")
                    #axs[1].axhline(y=targ_q_t.values, color='k', linestyle=':',label='Target (normalized)')
                    axs[1].axhline(y=targ_q_t, color='k', linestyle=':',label='Target (normalized)')
                    axs[1].legend()


                    for runs in range(len(distances)):
                            axs[0].plot(distances[runs],linewidth=8, alpha=0.4)

                    for runs in range(len(targt_perfs)):

                        axs[1].plot(targt_perfs[runs],linewidth=8, alpha=0.4)


                    time.sleep(1.0)
                    fig2=plt.figure(figsize=(15, 5))
                    plt.xlabel('Number of required Experiments')
                    plt.ylabel("Frequency")
                    plt.title("Performance histogram for %s with strategy %s "%(self.model,self.strategy))
                    plt.hist([self.tries_list_rand_pick],range=(1, len(self.features_df)),label=['Random Process'],alpha=0.4)
                    plt.hist([self.tries_list],label=['SL'],range=(1, len(self.features_df)),alpha=0.4)
                    plt.legend()
                    plt.show()



        #Extend values of perfs
            lengths_of_perfs=[]
            for runs in range(len(targt_perfs)):
                            current_len_of_perf=len(targt_perfs[runs])
                            lengths_of_perfs.append(current_len_of_perf)

            for runs in range(len(targt_perfs)):
                                    if(len(targt_perfs[runs])!=max(lengths_of_perfs)):
                                        size_of_values_to_add =max(lengths_of_perfs)-len(targt_perfs[runs])
                                        targt_perfs[runs].extend(np.full(size_of_values_to_add, max(targt_perfs[runs])))

            targt_perfs_as_array=np.array(targt_perfs)
            mean_performances=np.mean(targt_perfs_as_array,axis=0)

            rel_perform_after_5=1.0
            rel_perform_after_10=1.0

            max_performance=np.max(sum_)

            min_performance=np.min(sum_)

            if(len(mean_performances) > 5 and len(mean_performances) <10):
                perform_after_5=mean_performances[4]
                rel_perform_after_5=(perform_after_5-min_performance)/(max_performance-min_performance)

            if(len(mean_performances)==5):
                perform_after_5=mean_performances[4]
                rel_perform_after_5=(perform_after_5-min_performance)/(max_performance-min_performance)

            if(len(mean_performances)>=10):
                perform_after_5=mean_performances[4]
                perform_after_10=mean_performances[9]
                print(perform_after_10/max_performance)
                rel_perform_after_5=(perform_after_5-min_performance)/(max_performance-min_performance)
                rel_perform_after_10=(perform_after_10-min_performance)/(max_performance-min_performance)



            if self.strategy== 'MEI (exploit)':
                self.sigma=0
                self.distance=0
            elif self.strategy=='MU (explore)':
                self.sigma=1
                self.distance=0
            elif self.strategy=='MLI (explore & exploit)':
                self.distance=0
            elif self.strategy=='MEID (exploit)':
                self.sigma=0


            to_append=([np.mean(self.tries_list),np.std(self.tries_list),np.quantile(self.tries_list,0.90),
                        np.quantile(self.tries_list,1),self.model, self.strategy,self.sigma,rel_perform_after_5,rel_perform_after_10,self.distance,
                        self.number_of_executions,self.init_sample_size,len(self.dataframe.index),len(self.feature),
                        len(self.targets),self.target_treshhold,self.sample_treshold,
                        self.feature,self.targets,self.tries_list])

####SL-Results

            print('Performance summary:')
            print('requ. experiments with optimzation (mean):',np.mean(self.tries_list))
            print("requ. experiments without optimzation (mean): {}".format(np.mean(self.tries_list_rand_pick)))
            print('### Result plots:')


#Plot targets
### Fixed Targets
            if(len(self.fixedtargets.values.tolist())>0):
                    anzahl_plots=len(self.fixedtargets)
                    fig3,axs_fixed = plt.subplots(anzahl_plots,figsize=(8,5*anzahl_plots),squeeze=False)
                    axs_fixed=axs_fixed.flatten()

                    fixed_targets_extended=extend(fixed_targets)

                    mean_fixed_targets_extended=np.mean(fixed_targets_extended,axis=0)

                    fixed_rand_extended=extend(self.rand_fixed_tars)

                    mean_fixed_rand_extended=np.mean(fixed_rand_extended,axis=0)

#Plot fixed targets
                    for fixed_target in range(anzahl_plots):
                            axs_fixed[fixed_target].set_title('Created value for %s'%(self.fixedtargets[fixed_target]))
                            axs_fixed[fixed_target].set_xlabel('development cycles')
                            axs_fixed[fixed_target].set_ylabel("Best sampled property")
                            axs_fixed[fixed_target].set_xlim([0,len(mean_fixed_targets_extended[:,0])])

                    for one_tar in range(anzahl_plots):

                        axs_fixed[one_tar].plot(mean_fixed_targets_extended[:,one_tar],linewidth=8, alpha=0.9, color='k',label='With optimization')
                        axs_fixed[one_tar].plot(mean_fixed_rand_extended[:,one_tar],linewidth=8, alpha=0.9, color='g',label='Without optimization')
                        axs_fixed[one_tar].axvline(x=round(np.mean(self.tries_list)-self.init_sample_size), color='k', linestyle=':',label='Average SL cycles to success')
                        axs_fixed[one_tar].legend()

                    for sl_run in range(len(targets_as_list)):
                            for one_tar in range(anzahl_plots):
                                        axs_fixed[one_tar].plot(fixed_targets[sl_run][:,one_tar],linewidth=2, alpha=0.1,color='k')




            #with out_results_SL:
            anzahl_plots=len(targets)
            fig4,axs_pred = plt.subplots(anzahl_plots,figsize=(8,5*anzahl_plots), squeeze=False)
            axs_pred=axs_pred.flatten()
            targets_extended=extend(targets_as_list)
            mean_targets_extended=np.mean(targets_extended,axis=0)
            rand_extended=extend(self.rand_tars)
            mean_rand_extended=np.mean(rand_extended,axis=0)


            for pred_target in range(anzahl_plots):
                            axs_pred[pred_target].set_title('Created value for %s'%(targets_as_list[pred_target]))
                            axs_pred[pred_target].set_xlabel('development cycles')
                            axs_pred[pred_target].set_ylabel("Best sampled property")
                            axs_pred[pred_target].set_xlim([0,len(mean_targets_extended[:,0])])


            for pred_target in range(anzahl_plots):

                        axs_pred[pred_target].plot(mean_targets_extended[:,pred_target],linewidth=8, alpha=0.9, color='k',label='With optimization')
                        axs_pred[pred_target].plot(mean_rand_extended[:,pred_target],linewidth=8, alpha=0.9, color='g',label='Without optimization')
                        axs_pred[pred_target].axvline(x=round(np.mean(self.tries_list)-self.init_sample_size), color='k', linestyle=':',label='Average SL cycles to success')
                        axs_pred[pred_target].legend()


            for sl_run in range(len(targets)):
                            for one_tar in range(anzahl_plots):
                                        axs_pred[one_tar].plot(targets_as_list[sl_run][:,one_tar],linewidth=2, alpha=0.1,color='k')
                                        plt.xlim([0,len(mean_targets_extended[:,one_tar])])

            plt.show()


            print('History:')

            a_series = pd.Series(to_append, index = result_df.columns)
            result_df=result_df.append(a_series, ignore_index=True)

            print(result_df)
            print('done ✅')





    def perform_random_pick(self,acutal_iter):
        s3 = pd.concat([self.dataframe[self.target_name], self.fixedtargets.to_frame()], axis=1)
        sum_ = s3.sum(axis=1)
        print('sum s3', sum_)
        index_sum=sum_.index.to_numpy()
        index_sum_randomized=np.random.choice(index_sum,len(index_sum),False)
        targ_q_t= sum_.quantile(self.target_treshhold)
        self.tries_list_rand_pick[acutal_iter]=1
        run=0

        best_value=self.dataframe.iloc[index_sum_randomized[run]]

        current_fixed_rand_tars=np.array([self.fixedtargets])
        current_pred_rand_tars=np.array([self.targets])

        while sum_.iloc[index_sum_randomized[run]].astype(float).item() < targ_q_t.item():
            self.tries_list_rand_pick[acutal_iter]=self.tries_list_rand_pick[acutal_iter]+1


            temp_index=np.argmax(sum_.iloc[index_sum_randomized[0:run+1]])
            max_index=index_sum_randomized[temp_index]
            best_value=self.dataframe.iloc[max_index]
            #current_pred_rand_tars=np.vstack([current_pred_rand_tars,best_value[self.targets].to_numpy()])
            current_pred_rand_tars=np.vstack([current_pred_rand_tars,self.targets])
            #current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,best_value[self.fixedtargets].to_numpy()])
            current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,self.fixedtargets])

            run=run+1


        temp_index=np.argmax(sum_.iloc[index_sum_randomized[0:run+1]])
        max_index=index_sum_randomized[temp_index]
        best_value=self.dataframe.iloc[max_index]
        #current_pred_rand_tars=np.vstack([current_pred_rand_tars,best_value[self.targets].to_numpy()])
        #current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,best_value[self.fixedtargets].to_numpy()])
        current_pred_rand_tars=np.vstack([current_pred_rand_tars,self.targets])
        current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,self.fixedtargets])

        self.rand_fixed_tars.append(current_fixed_rand_tars)
        self.rand_tars.append(current_pred_rand_tars)


    def update_strategy(self, strategy):
        if strategy=='MEI (exploit)':
            self.updateIndexMEI()
        elif strategy=='MU (explore)':
            self.updateIndexMU()
        elif strategy=='MLI (explore & exploit)':
            self.updateIndexMLI()
        elif strategy=='MEID (exploit)':
            self.updateIndexMEID()
        elif strategy=='MLID (explore & exploit)':
            self.updateIndexMLID()


    def updateIndexMEI(self):
            #fixed_targets_in_prediction=self.dataframe[self.fixedtargets].iloc[self.PredIdx].sum(axis=1).to_frame()
            fixed_targets_in_prediction=self.fixedtargets.iloc[self.PredIdx]

            index_max = np.argmax(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze())
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.PredIdx=new_PredIdx


    def updateIndexMEID(self):
            #fixed_targets_in_prediction=self.dataframe[self.fixedtargets].iloc[self.PredIdx].sum(axis=1).to_frame()
            fixed_targets_in_prediction=self.fixedtargets.iloc[self.PredIdx]
            schwellwert=np.quantile(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze(),self.distance/100)
            Index_=np.where(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze()>=schwellwert )
            Index_=Index_[0]
            distance= distance_matrix(self.dataframe.loc[self.SampIdx],self.dataframe.iloc[Index_])
            min_distances=distance.min(0)
            result = np.where(distance == min_distances.max())

            # zip the 2 arrays to get the exact coordinates
            listOfCordinates = list(zip(result[0], result[1]))
            index_max=Index_[result[1]]
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.PredIdx=new_PredIdx



    def updateIndexMLID(self):
            fixed_targets_in_prediction=self.dataframe[fixedtargets].iloc[self.PredIdx].sum(axis=1).to_frame()
            schwellwert=np.quantile((fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze()+self.sigma*self.Uncertainty.squeeze()),self.distance/100)
            Index_=np.where(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze()+self.sigma*self.Uncertainty.squeeze()>=schwellwert )
            Index_=Index_[0]
            distance= distance_matrix(self.dataframe.loc[self.SampIdx],self.dataframe.iloc[Index_])
            min_distances=distance.min(0)
            result = np.where(distance == min_distances.max())
            # zip the 2 arrays to get the exact coordinates
            listOfCordinates = list(zip(result[0], result[1]))
            index_max=Index_[result[1]]
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.PredIdx=new_PredIdx


    def updateIndexMU(self):
            index_max = np.argmax(self.Uncertainty)
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.PredIdx=new_PredIdx


    def updateIndexMLI(self):
            fixed_targets_in_prediction=self.dataframe[fixedtargets].iloc[self.PredIdx].sum(axis=1).to_frame()
            index_max = np.argmax(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze()+self.sigma*self.Uncertainty.squeeze())
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.PredIdx=new_PredIdx



    def jk_resampling(self):
        from resample.jackknife import resample as b_resample
        td=[x for x in b_resample(self.features_df.iloc[self.SampIdx])]
        #tl=[x for x in b_resample(self.dataframe[targets].iloc[self.SampIdx].sum(axis=1).to_frame())]
        t = self.targets.iloc[self.SampIdx]#.sum(axis=1).to_frame()
        tl=[x for x in b_resample(t)]
        td=np.array(td)
        tl=np.array(tl)
        return td,tl



    def main(self):
        self.start_sequential_learning()



#initial_sample_size=4
#target_quantile=80
#sample_quantile=50
#iterationen=3
#std=2
#dist=1
#model=None
#strategy='MEI (exploit)'


#s = sequential_learning(df,initial_sample_size,target_quantile,iterationen,sample_quantile,std,
#                        dist,model,strategy)

#dt=DT("Decision Tree",s,targets)
#s.model=dt
#s.main()

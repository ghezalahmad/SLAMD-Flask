import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor as SKRFR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from lolopy.learners import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from operator import add
from app import *

def extend(list_of_2dms_arrays_to_extend):
    np_array=np.array(list_of_2dms_arrays_to_extend, dtype=object)
    #print('np_array', np_array)
    max_cols=max(map(len,np_array))
    #print('max_cols', max_cols)
    result_list=[]
    for i in np_array:
        if(len(i) == max_cols):
            result_list.append(i)
        elif (len(i) != max_cols):
            how_often=max_cols-len(i)
            #print('how_often', how_often)
            matrix_to_extend=np.tile(i[:][-1], (how_often, 1))
            #print('i', i)
            #print('matrix', matrix_to_extend.shape)
            #print(matrix_to_extend)
            i=np.concatenate([i, matrix_to_extend])
            result_list.append(i)
    return result_list

def flatten_list(nested_list):
    for sublist in nested_list:
        flatlist=[element for element in sublist]
    return flatlist

## Benchmarking
class sequential_learning:
    #dataframe = df_converter()
    #features_df=df_converter()
    #target_df=df_converter()

    min_distances_list=[]
    y_pred_dtr_mean=None
    y_pred_dtr_std=None
    y_pred_dtr=None
    SampIdx=None
    PredIdx=None
    treshIdx=None
    index_sum_randomized=None
    rand_tars=[]
    rand_fixed_tars=[]

    def __init__(self,dataframe,initial_sample_size,batch_size, target_treshhold, number_of_executions,
             sigma, dist, model, strategy, target_df, fixed_targets_idx, feature_df, min_or_max_target, check_to_use_threshold_t,
             target_selected_number1,target_selected_number2, min_or_max_fixedtarget, check_to_use_threshold_ft,
             fixedtarget_selected_number1, fixedtarget_selected_number2, tquantile):  #constructor

        self.dataframe= dataframe
        self.features_df = feature_df
        self.init_sample_size=initial_sample_size
        self.batch_size=batch_size
        self.target_treshhold = target_treshhold/100
        self.number_of_executions=number_of_executions
        self.tries_list=np.empty(number_of_executions)
        self.tries_list_rand_pick=np.empty(number_of_executions)
        self.sigma=sigma
        self.distance=dist
        self.model=model
        self.strategy = strategy
        self.targets_idx=target_df
        self.fixed_targets_idx=fixed_targets_idx
        self.min_or_max_target = min_or_max_target
        self.check_to_use_threshold_t = check_to_use_threshold_t
        self.target_selected_number1 = target_selected_number1
        self.target_selected_number2 = target_selected_number2
        self.min_or_max_fixedtarget = min_or_max_fixedtarget
        self.check_to_use_threshold_ft = check_to_use_threshold_ft
        self.fixedtarget_selected_number1 = fixedtarget_selected_number1
        self.fixedtarget_selected_number2 = fixedtarget_selected_number2
        self.tquantile = tquantile/100

        #print(self.dataframe[self.targets_idx])
    def apply_feature_selection_to_df(self,dataframe):
        self.features_df = self.dataframe[self.features_df]

    def apply_target_selection_to_df(self,dataframe):
        self.target_df= self.dataframe[self.targets_idx]

    #self werte return macht wenig sinn
    def standardize_data(self):
        dataframe_norm=(self.dataframe-self.dataframe.mean())/self.dataframe.std()
        target_df_norm=(self.target_df-self.target_df.mean())/self.target_df.std()
        features_df_norm=(self.features_df-self.features_df.mean())/self.features_df.std()
        self.features_df=features_df_norm
        self.target_df=target_df_norm
        self.dataframe=dataframe_norm
        return self.features_df, self.target_df, self.dataframe

    def init_sampling(self):
        targets = self.targets_idx
        fixed_targets=(self.fixed_targets_idx)
        #df_unnorm=df_converter()
        df_unnorm=self.dataframe
        df=(df_unnorm-df_unnorm.mean())/(df_unnorm.std())
        sum_ = self.dataframe[self.targets_idx].sum(axis=1).to_frame()+self.dataframe[self.fixed_targets_idx].sum(axis=1).to_frame()
        checked_targets=[]
        treshholded_idx=self.check_input_variables()
        if(treshholded_idx):
            for row in self.check_to_use_threshold_t:
                if (self.check_to_use_threshold_t[row] == True):
                    checked_targets.append(row)

            for row in self.check_to_use_threshold_ft:
                if (self.check_to_use_threshold_ft[row] == True):
                    checked_targets.append(row)

            if not(len(checked_targets)==len(self.check_to_use_threshold_t)+len(self.check_to_use_threshold_ft)):
                sum_without_checked_targets=df.drop(columns=checked_targets).sum(axis=1)
                targ_q = (self.tquantile/100)
                targ_q_t= np.quantile(sum_without_checked_targets.iloc[treshholded_idx], 0.5)
                #targ_q_t= sum_without_checked_targets.iloc[treshholded_idx].quantile(1)
                tempIndex=np.where(sum_without_checked_targets.iloc[treshholded_idx] >= targ_q_t )
                tempIndex=tempIndex[0]
                Index_c=sum_without_checked_targets.iloc[treshholded_idx].iloc[tempIndex].index
                #Sample IDX
                Index_samp=np.delete(sum_.index, Index_c)
            else:
                Index_c=treshholded_idx
                Index_samp=np.delete(sum_.index, treshholded_idx)
        else:
            targ_q = self.tquantile/100
            targ_q_t= sum_.quantile(targ_q)
            Index_samp=np.where(sum_ < targ_q_t )
            Index_samp=Index_samp[0]
            Index_c=np.where(sum_ >= targ_q_t )
            Index_c=Index_samp[0]

        init_sample_set = np.ones((0,self.init_sample_size))
        for i in range(self.number_of_executions):
            Index_samp = 1 # we have to ask cr
            init_sample_set=np.vstack([init_sample_set, random.choice(Index_samp,self.init_sample_size)])
        return init_sample_set

    def start_sequential_learning(self):
        self.tries_list=np.empty(self.number_of_executions)
        self.tries_list.fill(np.nan)
        self.tries_list_rand_pick=np.empty(self.number_of_executions)
        self.tries_list_rand_pick.fill(np.nan)
        distances=[]
        targt_perfs=[]
        fixed_targets=[]
        targets=[]
        current_distances_list=[]
        current_targt_perf_list=[]
        #with out_perform_experiment:
        #        display(Markdown('Sequential Learning is running...'))
        print('Sequential Learning is running...')
        global result_df

        result_df = pd.DataFrame(columns=['Req. dev. cycle (mean)','Req. dev. cycle (std)','Req. dev. cycle (90%)',
                                  'Req. dev. cycle (max)','5 cycle perf.','10 cycle perf.','Batch size','Algorithm','Utlity function','σ factor',
                                  'qant. (distance utility)','# SL runs','Initial sample','# of samples in the DS',
                                  '# Features','# Targets', 'Target threshold','Features name','Targets name','A-priori information',
                                  'Req. experiments (list)'])


        self.dataframe=self.decide_max_or_min(self.min_or_max_target, self.dataframe)
        self.dataframe=self.decide_max_or_min(self.min_or_max_fixedtarget, self.dataframe)

        init_sample_set=self.init_sampling()
        fixed_targets_index=self.fixed_targets_idx

        sum_ = self.dataframe[self.targets_idx].sum(axis=1).to_frame()+self.dataframe[fixed_targets_index].sum(axis=1).to_frame()

        targ_q_t= sum_.quantile(self.tquantile)
        schwellwert=sum_.quantile(self.target_treshhold)
        Index_c=np.where(sum_ >= schwellwert )
        Index_c=Index_c[0]

        for i in range(self.number_of_executions):
            print('number of exec', self.number_of_executions)
            self.perform_random_pick(i)
            self.SampIdx=init_sample_set[i].astype(int)
            self.PredIdx=self.dataframe
            self.PredIdx = self.PredIdx.drop(self.PredIdx.index[self.SampIdx]).index
            self.decide_model(self.model)
            self.tries_list[i]=0
            #self.init_sample_size
            distance=distance_matrix(self.dataframe.iloc[self.SampIdx],self.dataframe.iloc[self.treshIdx])
            distance=distance.min()
            #print("distance",distance)
            current_distances_list=[distance]
            #max value summe
            targt_perf=sum_.loc[self.SampIdx].max().item()
            current_targt_perf_list=[targt_perf]
            max_targt_perf_index=np.argmax(sum_.loc[self.SampIdx].values, axis=0)
            Idx_of_best_value=self.SampIdx[max_targt_perf_index]
            best_value=self.dataframe.iloc[Idx_of_best_value]
            current_fixed_target_list=np.array(best_value[self.fixed_targets_idx].to_numpy()[0])
            current_prediction_target=np.array(best_value[self.targets_idx].to_numpy()[0])

            while np.any(np.in1d(self.SampIdx,self.treshIdx ))== True:

                batch_size=self.batch_size
                for batch in range(batch_size):
                    #print('samp check', self.SampIdx.size)
                    #print('batch check', self.treshIdx)
                    if(self.SampIdx.size<batch_size):

                        batch_size=self.SampIdx.size
                        self.update_strategy(self.strategy)
                    else:
                        #print('to check while lloop 2')
                        self.update_strategy(self.strategy)
                        #Train Model
                self.decide_model(self.model)
                distance= distance_matrix(self.dataframe.iloc[self.SampIdx],self.dataframe.iloc[self.treshIdx])
                distance=distance.min()
                current_distances_list.append(distance)
                targt_perf=sum_.loc[self.SampIdx].max().values.tolist()
                targt_perf=max(targt_perf)

                current_targt_perf_list.append(targt_perf)
                max_targt_perf_index=np.argmax(sum_.loc[self.SampIdx].values, axis=0)
                Idx_of_best_value=self.SampIdx[max_targt_perf_index]
                best_value=self.dataframe.iloc[Idx_of_best_value]

                current_prediction_target=np.vstack([current_prediction_target,best_value[self.targets_idx].to_numpy()[0]])
                current_fixed_target_list=np.vstack([current_fixed_target_list,best_value[self.fixed_targets_idx].to_numpy()[0]])

                self.tries_list[i]=self.tries_list[i]+1

            distances.append(current_distances_list)
            targt_perfs.append(current_targt_perf_list)

            best_value=self.dataframe.iloc[self.treshIdx]
            current_prediction_target=np.vstack([current_prediction_target,best_value[self.targets_idx].to_numpy()[0]])
            current_fixed_target_list=np.vstack([current_fixed_target_list,best_value[self.fixed_targets_idx].to_numpy()[0]])
            targets.append(current_prediction_target)
            fixed_targets.append(current_fixed_target_list)
    ## Live Plots

            #with out_perform_experiment:
            fig1,axs = plt.subplots(1,2,figsize=(15, 6))
            axs[0].set_title('Optimization progress in input space')
            axs[0].set_xlabel('development cycles')
            axs[0].set_ylabel("Minimum distance from sampled data to target")
            axs[0].axhline(y=0, color='k', linestyle=':',label='Target')
            axs[0].legend()

            axs[1].set_title('Optimization progress in output space')
            axs[1].set_xlabel('development cycles')
            axs[1].set_ylabel("Maximum sampled property")
            axs[1].axhline(y=self.tquantile, color='k', linestyle=':',label='Target (normalized)')
            axs[1].legend()
            #Plotting
            for runs in range(len(distances)):
                axs[0].plot(distances[runs],linewidth=8, alpha=0.4)

            for runs in range(len(targt_perfs)):
                axs[1].plot(targt_perfs[runs],linewidth=8, alpha=0.4)

                #with out_perform_experiment:
                #out_perform_experiment.clear_output(wait=True)
            #time.sleep(1.0)
            fig2=plt.figure(figsize=(15, 5))
            plt.xlabel('Number of required Experiments')
            plt.ylabel("Frequency")
            plt.title("Performance histogram for {} with strategy {}".format(self.model, self.strategy))
                #plt.hist([self.tries_list,self.tries_list_rand_pick],bins=len(self.tries_list),label=['SL Tries', 'Random Pick Tries'])
            plt.hist([self.tries_list_rand_pick],range=(1, len(self.features_df)),label=['Random Process'],alpha=0.4)
            plt.hist([self.tries_list],label=['SL'],range=(1, len(self.features_df)),alpha=0.4)
            plt.legend()
            plt.show()
        #plt.close(fig2)

            #with out_perform_experiment:
            #    display(Markdown('current iteration {}'.format(i)))
            #    display(Markdown(" "))
            print('current iteration {}'.format(i))

                        #self.number_of_executions
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
            #print(perform_after_10/max_performance)
            rel_perform_after_5=(perform_after_5-min_performance)/(max_performance-min_performance)
            rel_perform_after_10=(perform_after_10-min_performance)/(max_performance-min_performance)

        if self.strategy=='MEI (exploit)':
            self.sigma=0
            self.distance=0
        elif self.strategy=='MU (explore)':
            self.sigma=1
            self.distance=0
        elif self.strategy=='MLI (explore & exploit)':
            self.distance=0
        elif self.strategy=='MEID (exploit)':
            self.sigma=0


        ##Appending Performance intensiv --> List comprehension
        to_append=([np.mean(self.tries_list),np.std(self.tries_list),np.quantile(self.tries_list,0.90),
                    np.quantile(self.tries_list,1),rel_perform_after_5,rel_perform_after_10,self.batch_size,self.model, self.strategy,self.sigma,self.distance,
                    self.number_of_executions,self.init_sample_size,len(self.dataframe.index),len(self.features_df.columns),
                    len(self.targets_idx),self.target_treshhold,
                    self.features_df.columns,self.targets_idx,self.fixed_targets_idx,self.tries_list])

####SL-Results
        #with out_results_SL:
        #out_results_SL.clear_output(wait=True)
        #display(Markdown('#### Performance summary:'))
        print("Performance summary")
        print("req. development cycles with optimzation (mean):  {} ".format(np.mean(self.tries_list)))
        print("req. development cycles  without optimzation (mean): {}".format(np.mean(self.tries_list_rand_pick)))
        #display(Markdown('req. development cycles with optimzation (mean):  {} '.format(np.mean(self.tries_list))))
        #display(Markdown("req. development cycles  without optimzation (mean): {}".format(np.mean(self.tries_list_rand_pick))))

        #with out_results_SL:
        #display(Markdown(" "))
        #display(Markdown('#### Log:'))
        #display(Markdown(" "))
        a_series = pd.Series(to_append, index = result_df.columns)
        result_df= result_df.append(a_series, ignore_index=True)
        #display(Markdown(result_df.to_markdown()))
        #display((create_download_link(result_df,'Download Log-File','results_sl')))
        print("Result DF", result_df)

#Plot targets
### Fixed Targets
        if(len(self.fixed_targets_idx)>0):
            anzahl_plots=len(self.fixed_targets_idx)
            fig3,axs_fixed = plt.subplots(anzahl_plots,figsize=(8,5*anzahl_plots),squeeze=False)
            axs_fixed=axs_fixed.flatten()

            fixed_targets_extended=extend(fixed_targets)
            mean_fixed_targets_extended=np.mean(fixed_targets_extended, axis=0)
            #print('fixed_exted', fixed_targets_extended)
            fixed_rand_extended=extend(self.rand_fixed_tars)
            mean_fixed_rand_extended=np.mean(fixed_rand_extended, axis=0)

#Plot fixed targets
            """ """
            for fixed_target in range(anzahl_plots):
                axs_fixed[fixed_target].set_title('Optimization progress for %s'%(self.fixed_targets_idx[fixed_target]))
                axs_fixed[fixed_target].set_xlabel('development cycles')
                axs_fixed[fixed_target].set_ylabel("Best sampled property")
                axs_fixed[fixed_target].set_xlim([0,len(mean_fixed_targets_extended[:,0])-1])

            for one_tar in range(anzahl_plots):
                axs_fixed[one_tar].plot(mean_fixed_targets_extended[:,one_tar],linewidth=8, alpha=0.9, color='k',label='With optimization')
                axs_fixed[one_tar].plot(mean_fixed_rand_extended[:,one_tar],linewidth=8, alpha=0.9, color='g',label='Without optimization')
                axs_fixed[one_tar].axvline(x=round(np.mean(self.tries_list)-self.init_sample_size), color='k', linestyle=':',label='Average dev. cycles to success')
                axs_fixed[one_tar].legend()

            for sl_run in range(len(fixed_targets)):
                for one_tar in range(anzahl_plots):
                    axs_fixed[one_tar].plot(fixed_targets[sl_run][:,one_tar],linewidth=2, alpha=0.1,color='k')

        #with out_results_SL:
        #display(Markdown(" "))
        #display(Markdown('#### Result plots:'))

        anzahl_plots=len(self.targets_idx)
        fig4,axs_pred = plt.subplots(anzahl_plots,figsize=(8,5*anzahl_plots), squeeze=False)
        axs_pred=axs_pred.flatten()

        targets_extended=extend(targets)
        #print('type', targets_extended)
        #xlist = [float(i) for i in targets_extended]
        mean_targets_extended=np.mean(targets_extended,axis=0)


        plt.setp(axs_pred, xlim=[0,len(mean_targets_extended[:,0])-1])

        rand_extended=extend(self.rand_tars)
        mean_rand_extended=np.mean(rand_extended,axis=0)
        """ """
        for pred_target in range(anzahl_plots):
            axs_pred[pred_target].set_title('Optimization progress for %s'%(self.targets_idx[pred_target]))
            axs_pred[pred_target].set_xlabel('development cycles')
            axs_pred[pred_target].set_ylabel("Best sampled property")

        for pred_target in range(anzahl_plots):
            axs_pred[pred_target].plot(mean_targets_extended[:,pred_target],linewidth=8, alpha=0.9, color='k',label='With optimization')
            axs_pred[pred_target].plot(mean_rand_extended[:,pred_target],linewidth=8, alpha=0.9, color='g',label='Without optimization')
            axs_pred[pred_target].axvline(x=round(np.mean(self.tries_list)-self.init_sample_size), color='k', linestyle=':',label='Average dev. cycles to success')
            axs_pred[pred_target].legend()

        for sl_run in range(len(targets)):
            for one_tar in range(anzahl_plots):
                axs_pred[one_tar].plot(targets[sl_run][:,one_tar],linewidth=2, alpha=0.1,color='k')

        plt.show()


    def perform_random_pick(self,acutal_iter):
        #print("self.dataframe sollte min max hjaben")
        #print(self.dataframe)
        sum_ = self.dataframe[self.targets_idx].sum(axis=1).to_frame()+self.dataframe[self.fixed_targets_idx].sum(axis=1).to_frame()
        index_sum=sum_.index.to_numpy()
        index_sum_randomized=np.random.choice(index_sum,len(index_sum),False)
        targ_q_t= sum_.quantile(self.target_treshhold)
        self.tries_list_rand_pick[acutal_iter]=1

        run=0
        best_value=self.dataframe.iloc[index_sum_randomized[run]]
        current_fixed_rand_tars=np.array(best_value[self.fixed_targets_idx].to_numpy())
        current_pred_rand_tars=np.array(best_value[self.targets_idx].to_numpy())


        while np.any(np.in1d(index_sum_randomized[run],self.treshIdx ))== False:
            self.tries_list_rand_pick[acutal_iter]=self.tries_list_rand_pick[acutal_iter]+1
            temp_index=np.argmax(sum_.iloc[index_sum_randomized[0:run+1]])
            max_index=index_sum_randomized[temp_index]
            best_value=self.dataframe.iloc[max_index]
            current_pred_rand_tars=np.vstack([current_pred_rand_tars,best_value[self.targets_idx].to_numpy()])
            current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,best_value[self.fixed_targets_idx].to_numpy()])
            run=run+1

        temp_index=np.argmax(sum_.iloc[index_sum_randomized[0:run+1]])
        max_index=index_sum_randomized[temp_index]
        best_value=self.dataframe.iloc[max_index]
        current_pred_rand_tars=np.vstack([current_pred_rand_tars,best_value[self.targets_idx].to_numpy()])
        current_fixed_rand_tars=np.vstack([current_fixed_rand_tars,best_value[self.fixed_targets_idx].to_numpy()])

        self.rand_fixed_tars.append(current_fixed_rand_tars)
        self.rand_tars.append(current_pred_rand_tars)




    #Refactor idee: Model klasse mit name und checkbox description
    def decide_model(self,model):
        if model== 'lolo Random Forrest (RF)':
                    self.fit_RF_wJK()
        elif model == 'Decision Trees (DT)':
                    self.fit_DT_wJK()
        elif model == 'Random Forrest (RFscikit)':
                    self.fit_TE_wJK()
        elif model == 'Gaussian Process Regression (GPR)':
                    self.fit_GP()


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
        #print('fixed',self.dataframe[self.fixed_targets_idx].iloc[self.PredIdx])
        fixed_targets_in_prediction=self.dataframe[self.fixed_targets_idx].iloc[self.PredIdx].to_numpy()
        if(len(self.fixed_targets_idx)>0):
            for weights in range(len(fixedtarget_selected_number2)):
                fixed_targets_in_prediction[weights]= fixed_targets_in_prediction[weights]*fixtedtarget_selected_number2[weights]

        fixed_targets_in_prediction=fixed_targets_in_prediction.sum(axis=1)


        if(self.Expected_Pred.ndim>1):
            index_max = np.argmax(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.sum(axis=1).squeeze())
        else:
            index_max = np.argmax(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze())
            new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
            self.SampIdx=new_SampIdx
            new_PredIdx = np.delete(self.PredIdx, index_max)
            self.Expected_Pred = np.delete(self.Expected_Pred.squeeze(),index_max)
            self.PredIdx=new_PredIdx

    def updateIndexMEID(self):
        fixed_targets_in_prediction=self.dataframe[self.fixed_targets_idx].iloc[self.PredIdx].to_numpy()
        if(len(self.fixed_targets_idx)>0):
            for weights in range(len(ft.weights)):
                fixed_targets_in_prediction[weights]=fixed_targets_in_prediction[weights]*ft.weights[weights].value
        fixed_targets_in_prediction=fixed_targets_in_prediction.sum(axis=1)

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
        self.Expected_Pred = np.delete(self.Expected_Pred.squeeze(),index_max)
        self.PredIdx=new_PredIdx


    def updateIndexMLID(self):
        fixed_targets_in_prediction=self.dataframe[self.fixed_targets_idx].iloc[self.PredIdx].to_numpy()
        if(len(self.fixed_targets_idx)>0):
            for weights in range(len(ft.weights)):
                fixed_targets_in_prediction[weights]=fixed_targets_in_prediction[weights]*ft.weights[weights].value
        fixed_targets_in_prediction=fixed_targets_in_prediction.sum(axis=1)

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
        self.Expected_Pred = np.delete(self.Expected_Pred.squeeze(),index_max)
        self.Uncertainty=np.delete(self.Uncertainty.squeeze(),index_max)
        self.PredIdx=new_PredIdx

    def updateIndexMU(self):
        index_max = np.argmax(self.Uncertainty)
        new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
        self.SampIdx=new_SampIdx
        new_PredIdx = np.delete(self.PredIdx, index_max)
        self.Expected_Pred = np.delete(self.Expected_Pred.squeeze(),index_max)
        self.Uncertainty=np.delete(self.Uncertainty.squeeze(),index_max)
        self.PredIdx=new_PredIdx

    def updateIndexMLI(self):
        fixed_targets_in_prediction=self.dataframe[self.fixed_targets_idx].iloc[self.PredIdx].to_numpy()
        if(len(self.fixed_targets_idx)>0):
            for weights in range(len(ft.weights)):
                fixed_targets_in_prediction[weights]=fixed_targets_in_prediction[weights]*ft.weights[weights].value
        fixed_targets_in_prediction=fixed_targets_in_prediction.sum(axis=1)

        index_max = np.argmax(fixed_targets_in_prediction.squeeze()+self.Expected_Pred.squeeze()+self.sigma*self.Uncertainty.squeeze())
        new_SampIdx=np.append(self.SampIdx,self.PredIdx[index_max])
        self.SampIdx=new_SampIdx
        new_PredIdx = np.delete(self.PredIdx, index_max)
        self.Expected_Pred = np.delete(self.Expected_Pred.squeeze(),index_max)
        self.Uncertainty = np.delete(self.Uncertainty.squeeze(),index_max)
        self.PredIdx=new_PredIdx


    def fit_DT_wJK(self):
        td,tl=self.jk_resampling()
        #print('tl', tl)
        #print(self.features_df.iloc[self.PredIdx])
        self.y_pred_dtr=[]
        for i in range(len(td)):
            dtr = DecisionTreeRegressor()
            dtr.fit(td[i], tl[i])
            self.y_pred_dtr.append(dtr.predict(self.features_df.iloc[self.PredIdx]))
            #print('prediction', dtr.predict(self.features_df.iloc[self.PredIdx]))

        self.y_pred_dtr=np.array(self.y_pred_dtr)
        self.Expected_Pred = self.y_pred_dtr.mean(axis=0)
        self.Uncertainty = self.y_pred_dtr.std(axis=0)
        #multiply Prediction with factor
        #print('expected',self.Expected_Pred, self.Expected_Pred.shape)
        self.weight_Pred()
        return self.Expected_Pred, self.Uncertainty


    def fit_TE_wJK(self):
        td,tl=self.jk_resampling()
        self.y_pred_dtr=[]
        for i in range(len(td)):
            ## alternative Ensamble Learners below:
            dtr = SKRFR(n_estimators=10)
            dtr.fit(td[i], tl[i])
            self.y_pred_dtr.append(dtr.predict(self.features_df.iloc[self.PredIdx]))

        self.y_pred_dtr=np.array(self.y_pred_dtr)
        self.Expected_Pred = self.y_pred_dtr.mean(axis=0)
        self.Uncertainty = self.y_pred_dtr.std(axis=0)
        self.weight_Pred()
        return self.Expected_Pred, self.Uncertainty

    def jk_resampling(self):
        from resample.jackknife import resample as b_resample
        td=[x for x in b_resample(self.features_df.iloc[self.SampIdx])]
        tl=[x for x in b_resample(self.dataframe[self.targets_idx].iloc[self.SampIdx])]
        tl=np.array(tl)
        td=np.array(td)
        return td,tl

    def fit_RF_wJK(self):
        dtr = RandomForestRegressor()
        self.x=self.features_df.iloc[self.SampIdx].to_numpy()
        self.y=self.dataframe[self.targets_idx].iloc[self.SampIdx].sum(axis=1).to_frame().to_numpy()
        if self.y.shape[0]<8:
            self.x=np.tile(self.x,(4,1))
            self.y=np.tile(self.y,(4,1))
        dtr.fit(self.x, self.y)
        self.Expected_Pred, self.Uncertainty = dtr.predict(self.features_df.iloc[self.PredIdx].to_numpy(), return_std=True)
        self.weight_Pred()
        return self.Expected_Pred, self.Uncertainty

    def fit_GP(self):
        #print('sampleid', self.features_df.iloc[self.SampIdx])
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        dtr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        dtr.fit(self.features_df.iloc[self.SampIdx].to_numpy(), self.dataframe[self.targets_idx].iloc[self.SampIdx])#.sum(axis=1).to_frame().to_numpy())
        self.Expected_Pred, self.Uncertainty= dtr.predict(self.features_df.iloc[self.PredIdx], return_std=True)
        self.weight_Pred()
        return self.Expected_Pred.squeeze(), self.Uncertainty.squeeze()

    def weight_Pred(self):
        #print('number2', self.target_selected_number2)
        dt_list = []
        for d in self.target_selected_number2:
            dt_list.append(self.target_selected_number2[d])
        dt_list = np.asarray(dt_list, dtype='int')
        #print(dt_list)
        if(self.Expected_Pred.ndim>1):
            for weights in range(len(dt_list)):
                self.Expected_Pred=self.Expected_Pred*dt_list[weights]
        else:
            self.Expected_Pred=self.Expected_Pred*dt_list[0]

    def create_target_idx_after_logic_criteria(self,treshholded_idx,df,sum_):
        checked_targets=[]
        if(treshholded_idx):
            if(self.check_to_use_threshold_t is not None):
                for row in self.check_to_use_threshold_t:
                    if (self.check_to_use_threshold_t[row] == True):
                        checked_targets.append(row)

            if(self.check_to_use_threshold_ft is not None):
                for row in self.check_to_use_threshold_ft:
                    if (self.check_to_use_threshold_ft[row] == True):
                        checked_targets.append(row)

            if (len(checked_targets)!=len(self.check_to_use_threshold_t)+len(self.check_to_use_threshold_ft)):
                targets=self.dataframe[self.targets_idx]
                fixed_targets=self.dataframe[self.fixed_targets_idx]
                targets = pd.DataFrame(targets)
                fixed_targets = pd.DataFrame(fixed_targets)
                df = pd.concat([targets,  fixed_targets], axis=1)
                sum_without_checked_targets=df#.sum(axis=1)
                targ_q = self.target_treshhold #.value/100
                targ_q_t= sum_without_checked_targets.iloc[treshholded_idx].quantile(targ_q)
                tempIndex=np.where(sum_without_checked_targets.iloc[treshholded_idx] >= targ_q_t )
                tempIndex=tempIndex[0]
                Index_c=sum_without_checked_targets.iloc[treshholded_idx].iloc[tempIndex].index

                #Sample IDX
                Index_samp=np.delete(sum_.index, Index_c)
            else:
                Index_c=treshholded_idx
                Index_samp=np.delete(sum_.index, treshholded_idx)


        elif(not treshholded_idx):
            targ_q = self.tquantile/100
            targ_q_t= sum_.quantile(targ_q)
            Index_samp=np.where(sum_ < targ_q_t )
            Index_samp=Index_samp[0]
            Index_c=np.where(sum_ >= targ_q_t )
            Index_c=Index_c[0]
        return Index_samp,Index_c


    def plot_TSNE_input_space(self):
        from sklearn.manifold import TSNE
        treshholded_idx=self.check_input_variables()
        features_df,target_df,fixed_target_df,df_unnorm,df=self.load_data()
        target_df=decide_max_or_min(box_targets,self.targets_idx,target_df)
        fixed_target_df=decide_max_or_min(box_fixed_targets,self.fixed_targets_idx,fixed_target_df)
        df = decide_max_or_min(box_targets,self.targets_idx,df)
        df=decide_max_or_min(box_fixed_targets,self.fixed_targets_idx, df)
        sum_ = target_df.sum(axis=1).to_frame()+fixed_target_df.sum(axis=1).to_frame()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,random_state=1000)
        tsne_results = tsne.fit_transform(features_df)
        Index_samp,Index_c=self.create_target_idx_after_logic_criteria(treshholded_idx,df,sum_)
        with out_input_space:
                # Plot Results in reduced FS
            out_input_space.clear_output(wait=True)
            fig3= plt.figure(figsize=(10, 6))

            cmap = plt.get_cmap('cool', 200)
            cmap.set_over('lawngreen')

            if(treshholded_idx):
                vmax=np.max(sum_.iloc[Index_samp])
                sum_.iloc[Index_c]=sum_.iloc[Index_c]+1000

                sc=plt.scatter(x=tsne_results[:,0],y=tsne_results[:,1], c=sum_,
                                   cmap=cmap, vmax=vmax)
            else:
                targ_q = self.tquantile/100
                targ_q_t= sum_.quantile(targ_q)

                Index_samp=np.where(sum_ < targ_q_t )
                Index_samp=Index_samp[0]

                Index_c=np.where(sum_ >= targ_q_t )
                Index_c=Index_samp[0]

                sum_.iloc[Index_c]=np.max(sum_.iloc[Index_samp])

                sc=plt.scatter(x=tsne_results[:,0],y=tsne_results[:,1], c=sum_,
                                   cmap=cmap, vmax=np.max(sum_.iloc[Index_samp]))

            cbar=plt.colorbar(sc,extend='both')
            cbar.ax.set_yticklabels([ ])
            cbar.ax.set_ylabel('target samples (green) normalized target property', rotation=270 ,va='center')

            plt.title("Materials data in TSNE-coordinates: candidates and targets")
            plt.show()
            plt.close(fig3)


        #Utility Methods
    def decide_max_or_min(self,source, dataframe):
        for s in source:
            if (source[s] == 'min'):
                dataframe[s] = dataframe[s]*(-1)
        return dataframe

    def confirm_fixed_target(self, source):
        fix_list = []
        for i in source:
            fix_list.append(i)
        return fix_list

    def check_input_variables(self):
        """
        from collections import Counter

        united_idxs=[]
        for i in self.target_df:
            united_idxs.append(self.dataframe.columns.get_loc(i))

        if(united_idxs is not None):

            idxs_without_feature_desc= [tuple_of_feature_and_idxs for tuple_of_feature_and_idxs in united_idxs]

            for sublist in idxs_without_feature_desc:
                for item in sublist:
                    flat_list = item
            #flat_list = [item for sublist in idxs_without_feature_desc for item in sublist]

            counts = Counter(flat_list)

            compatible_idxs = [id for id in flat_list if counts[id] >= (len(idxs_without_feature_desc))]

            treshholded_idx=set(compatible_idxs)

            treshholded_idx_as_list=list(treshholded_idx)"""

        united_idxs=[]
        for i in self.targets_idx:
            united_idxs.append(self.dataframe.columns.get_loc(i))

        treshholded_idx_as_list = united_idxs
        if treshholded_idx_as_list is not None:
            return treshholded_idx_as_list

        else:
            return None


    def load_data(self):

        #features_df=(df_converter()[confirm_features(feature_selector)]-df_converter()[confirm_features(feature_selector)].mean())/df_converter()[confirm_features(feature_selector)].std()
        #target_df=(df_converter()[confirm_target(target_selection)]-df_converter()[confirm_target(target_selection)].mean())/df_converter()[confirm_target(target_selection)].std()
        #fixed_target_df=(df_converter()[confirm_fixed_target(fixed_target_selection)]-df_converter()[confirm_fixed_target(fixed_target_selection)].mean())/df_converter()[confirm_fixed_target(fixed_target_selection)].std()

        features_df=(self.features_df-self.features_df.mean())/self.features_df.std()
        target_df=(self.dataframe[self.targets_idx]-self.dataframe[self.targets_idx].mean())/self.dataframe[self.targets_idx].std()
        fixed_target_df=(self.dataframe[self.fixed_targets_idx]-self.dataframe[self.fixed_targets_idx].mean())/self.dataframe[self.fixed_targets_idx].std()
        df_unnorm=self.dataframe
        df=(df_unnorm-df_unnorm.mean())/(df_unnorm.std())

        return features_df,target_df,fixed_target_df,df_unnorm,df

    def show_input_data(self):
        treshholded_idx=self.check_input_variables()
        features_df,target_df,fixed_target_df,df_unnorm,df=self.load_data()

        with out_input_space:
            display(Markdown('Target data'))
            if df is not None:
                targ_q = self.tquantile/100
                target_df=decide_max_or_min(box_targets,self.targets_idx,target_df)
                fixed_target_df=decide_max_or_min(box_fixed_targets,self.fixed_targets_idx,fixed_target_df)

                df = decide_max_or_min(box_targets,self.targets_idx, df)
                df=decide_max_or_min(box_fixed_targets,self.fixed_targets_idx, df)

                sum_ = target_df.sum(axis=1).to_frame()+fixed_target_df.sum(axis=1).to_frame()
                Index_samp,Index_c=self.create_target_idx_after_logic_criteria(treshholded_idx,df,sum_)

                display(Markdown(df_unnorm.iloc[Index_c].to_markdown()))
            else:
                display(Markdown('Configuration is wrong/missing...'))



    def main(self):

        #self.apply_feature_selection_to_df(self.dataframe)
        #self.features_df = self.dataframe[self.features_df]
        #self.apply_target_selection_to_df(self.dataframe)
        if(len((self.fixed_targets_idx))>0):
            self.target_df=self.dataframe[self.targets_idx].join(self.dataframe[(self.fixed_targets_idx)])
        self.standardize_data()
        init_sample_set=self.init_sampling()

        #quantile_tar_slider = 10
        #targ_q = self.tquantile/100
        targ_q = self.tquantile/100

        treshholded_idx= self.check_input_variables()
        features_df, target_df, fixed_target_df, df_unnorm, df=self.load_data()

        target_df=self.decide_max_or_min(self.min_or_max_target, target_df)
        fixed_target_df=self.decide_max_or_min(self.min_or_max_fixedtarget, fixed_target_df)

        sum_ = target_df.sum(axis=1).to_frame()+fixed_target_df.sum(axis=1).to_frame()
        treshholded_idx=self.check_input_variables()
        Index_samp,Index_c=self.create_target_idx_after_logic_criteria(treshholded_idx,df,sum_)

        self.treshIdx=Index_c
        self.start_sequential_learning()

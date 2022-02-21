## Materials Discovery
class learn():
    dataframe = df_converter()
    features_df = df_converter()
    target_df=df_converter()
    fixed_target_df=df_converter()

    y_pred_dtr_mean=None
    y_pred_dtr_std=None
    y_pred_dtr=None

    def __init__(self,dataframe,model,strategy,sigma,distance):
        self.dataframe=dataframe
        self.model=model
        self.strategy=strategy
        self.sigma=sigma
        self.distance=distance

        first_selected_target=list(confirm_target(target_selection_application))[0]
        self.PredIdx = pd.isnull(self.dataframe[[first_selected_target]]).to_numpy().nonzero()[0]
        self.SampIdx = self.dataframe.index.difference(self.PredIdx)


    def scale_data(self):

        dataframe_norm=(self.dataframe-self.dataframe.mean())/self.dataframe.std()
        target_df_norm=(self.target_df-self.target_df.mean())/self.target_df.std()
        features_df_norm=(self.features_df-self.features_df.mean())/self.features_df.std()
        fixed_target_df_norm=(self.fixed_target_df-self.fixed_target_df.mean())/self.fixed_target_df.std()

        self.features_df=features_df_norm
        self.target_df=target_df_norm
        self.dataframe=dataframe_norm
        self.fixed_target_df=fixed_target_df_norm

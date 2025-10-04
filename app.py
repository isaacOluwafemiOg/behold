#importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import catboost
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import associations
from sklearn.metrics import f1_score, confusion_matrix,ConfusionMatrixDisplay
import joblib
import statistics
import shap
from streamlit_shap import st_shap


st.set_page_config(layout='wide', initial_sidebar_state='expanded')


TRAIN_SCORE = 0.977
VALID_SCORE = 0.899
TEST_SCORE = 0.893

SELECTED_FEATURES = ['koi_duration', 'koi_ror', 'koi_prad', 'koi_sma', 'koi_dor','koi_max_sngle_ev',
                      'koi_max_mult_ev', 'koi_model_snr', 'koi_num_transits','koi_fwm_stat_sig',
                        'koi_fwm_sdec', 'koi_fwm_srao', 'koi_dicco_mra','koi_dikco_msky']

MODEL_HYPERPARAMETERS = {'iterations': 945, 'learning_rate': 0.0633262487790528,'max_depth': 6,
                          'l2_leaf_reg': 0.9155184101187535,'min_data_in_leaf': 90,
                          'colsample_bylevel': 0.7583283066319525,'bootstrap_type': 'Bayesian'}
TARGET_MAP = {'CANDIDATE':1,'FALSE POSITIVE':0}
REV_TARGET_MAP = {1:'CANDIDATE',0:'FALSE POSITIVE'}
np.random.seed(42)

@st.cache_resource
def load_models():
    loaded_models = joblib.load('./models/exoplanet_models.pkl')

    return loaded_models

@st.cache_resource
def prepare():
    loaded_models = load_models()
    hold_out_test = pd.read_csv('./data/test_data.csv')
    feat_desc = pd.read_csv('./data/data_desc.csv')
    # Fits the explainer
    explainers = [shap.TreeExplainer(model,
                                    hold_out_test[SELECTED_FEATURES]) for model in loaded_models]
    return loaded_models,hold_out_test,feat_desc,explainers

def missing_data(data):
    '''
    get information on missing data and column data types

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    --------
    na_df : pd.DataFrame
    dataframe containing counts and proportion of missing values in each column together with column data type
    '''
    total = data.isna().sum()
    percent = (data.isna().sum()/data.shape[0]*100).round(2)
    tt = pd.concat([total, percent], axis=1, keys=['Missing total', 'Percentage Missing'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['DataType'] = types

    na_df = np.transpose(tt)
    return na_df

def get_col_overview(df,col,desc):
    '''
    function to give overview of a given column in the given dataframe, df

    Parameters
    -----------
    df: pd.DataFrame
    dataframe for whose columns an overview is to be provided

    col: str
    the name of the feature column

    has_desc: bool
    Is there a separate dataframe containing descriptions on what each column represents?

    Returns
    -------
    None

    '''
    description = desc[desc['Feature']==col]['Description'].item()
    st.write(f'{col} is described as ',description)
    if df[col].nunique()>20:
        st.write('number of unqiue values: ',df[col].nunique())
        st.write('count of null_values: ',df[col].isna().sum())
        if col in df.select_dtypes(include=np.number).columns:
            st.write(f'maximum value: {df[col].max()}')
            st.write(f'minimum value: {df[col].min()}')
        else:
            st.write(f'modal value: {df[col].apply(mode)}',)

    else:
        st.write(f'counts of unique values observed in the {col} feature')
        statsdf = pd.DataFrame(df[col].value_counts(dropna=False,normalize=True,))
        st.dataframe(statsdf)

def plot_feature_scatter(df, fcol,n_cols=3):
    i = 0

    num_features = df.select_dtypes(include=np.number).columns
    num_features = [col for col in num_features if col!=fcol]
    
    n_rows = int(np.ceil(len(num_features)/n_cols))
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(14,(4*n_rows)))
    
    for feature in num_features:
        i += 1
        plt.subplot(n_rows,n_cols,i)
        plt.scatter(df[fcol], df[feature], marker='+')
        plt.xlabel(fcol, fontsize=9)
        plt.ylabel(feature, fontsize=9)
    st.pyplot(fig)

def single_predict(X,models,explainers):
    preds = []
    for model in models:
        preds.append(model.predict(X[SELECTED_FEATURES])[0])
    
    final_pred = statistics.mode(preds)

    #get models which were responsible for this prediction
    relevant_explainers = [explainers[i] for i in list(range(5)) if preds[i]==final_pred]

    return final_pred,relevant_explainers


def predict_batch(pdata,models):
    copy_pdata = pdata.copy()
    preds = {}
    for i,model in enumerate(models):
        preds[i] = model.predict(pdata[SELECTED_FEATURES]).ravel()
    
    pred_df = pd.DataFrame(preds)
    copy_pdata['prediction'] = pred_df.mode(axis=1)[0]
    
    return copy_pdata
    
    


def main():
    st.sidebar.header('A World Away: Hunting for Exoplanets with AI')
    #st.sidebar.image(imag,width=80)#,use_column_width=True)
    #st.sidebar.image(imag,use_column_width=True)
    mode =st.sidebar.selectbox('Menu:',['Explore','Single Prediction','Batch Prediction'],index=0)
    st.sidebar.markdown('''
    ---
    Created by [Team Behold](https://www.spaceappschallenge.org/2025/find-a-team/behold/)
    ''')
    
    
    st.title('Exoplanet Classifier')
    st.write('This web application functions on the backbone of advances in artificial intelligence\
              and machine learning.\
            The work of analyzing and training on large sets of data collected by the Kepler missions to\
            identify exoplanets  has culminated in this system that analyzes new data to accurately\
            identify exoplanets.')
    st.write('There are three modes of this application and it can be toggled using the drop-down on the\
            left side-bar')

    with st.spinner('loading resources... please wait'):
        loaded_models,hold_out_test,feat_desc,explainers = prepare()
        with_prediction = predict_batch(hold_out_test,loaded_models)
    
        

    if mode == 'Explore':
        with st.spinner("loading data and performance statistics... Pleae wait."):
            
            train = pd.read_csv('./data/train_data.csv')
            train['koi_pdisposition'] = train['koi_pdisposition'].map(REV_TARGET_MAP)
            
            feature_imp_df = pd.read_csv('./data/feature_importance.csv')
            feature_imp_df.columns = ['Feature','Importance']

        st.header('Explore the Model')

        st.write('The predictions of this web application are powered by a series of 5 Catboost\
                 Classifier models trained each on a different fold of 5 folds of training data\
                  obtained from the\
                  [Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) datatest\
                  of the NASA Exoplanet Archive')
        
        st.write('The catboost classifier model optimal hyperparameters values for the training process\
                  was:')
        st.write(MODEL_HYPERPARAMETERS)
        
        st.write('Below are the results of the model evaluation:')
        col1, col2, col3 = st.columns(3,gap='medium')
    
        col1.metric('Train F1 score:',
                     str(round(TRAIN_SCORE*100,2))+'%',
                     help='Performance of model on train data')
        col2.metric('Cross Validation F1 score:',
                     str(round(VALID_SCORE*100,2))+'%',
                     help='Performance of model using a 5-fold stratified cross-validation')
        col3.metric('Test F1 score:',
                     str(round(TEST_SCORE*100,2))+'%',
                     help='Performance of model on test data of 1913 observations')
        
        if st.button('Further Explore the Model'):
        
            st.subheader('Confusion Matrix ("Using Hold-Out Test Data")')
            cm = confusion_matrix(with_prediction['koi_pdisposition'],with_prediction['prediction'])

            # Create and plot the ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=['FALSE POSITIVE','CANDIDATE'])
            

            fig, ax = plt.subplots(figsize=(6, 6)) # Create a figure and an axes object
            disp.plot(cmap=plt.cm.Blues, ax=ax) # Plot on the created axes
            plt.title("Confusion Matrix")
            st.pyplot(fig) # Display the plot in Streamlit
        
        
            st.write("")
            st.write("")
            
            st.subheader('Feature Importance')
            st.write('The ML system makes use of a relevant subset of 22 features to predict the disposition\
                    of an observation. The following are the weights assigned to each feature.')
            
            feature_imp_df
            figfi,axfi = plt.subplots(figsize=(6,6))
            sns.barplot(feature_imp_df.head(5),ax=axfi,
                        y='Feature',
                        x='Importance')
            st.pyplot(figfi)


        st.markdown('''

         ---
        ''')

        st.header('Explore the Data')
        st.write('The data used to train the model was obtained by an 80-20 stratified split of the Kepler\
                  Objects of Interest dataset. 80% which corresponds to 7650 observations were used for\
                 training the model')
        st.write('Below is a chart that shows the balance in the target distribution:')
        target_dist = train['koi_pdisposition'].value_counts()
        fig_tdist,ax_tdist = plt.subplots(figsize=(3,3))
        sns.barplot(target_dist,ax=ax_tdist,palette='Set3')

        st.pyplot(fig_tdist)

        st.write('The Table below shows the number of missing values for each feature')
        na_df = missing_data(train[SELECTED_FEATURES])
        st.dataframe(na_df)

        st.write('')
        #Correlation with target

        correlation_matrix = associations(train[SELECTED_FEATURES+['koi_pdisposition']],
                                           compute_only=True)["corr"]
        cortab = pd.DataFrame(correlation_matrix['koi_pdisposition'].abs().sort_values(ascending=False))
        cortab = cortab.drop('koi_pdisposition')
        st.write('Correlation of Features with the target variable')
        st.dataframe(cortab)


        if st.button("Further Explore the Data"):   
            chosen_feature = st.selectbox('Select a Feature to dive deeper:',SELECTED_FEATURES,index=1)

            get_col_overview(train,chosen_feature,feat_desc)

            if chosen_feature in train.select_dtypes(include=np.number).columns:
                
                with st.spinner('Constructing Scatterplot... please wait'):
                    st.write(f'scatter plot of {chosen_feature} feature vs other features')
                    plot_feature_scatter(train[SELECTED_FEATURES+['koi_pdisposition']],chosen_feature)

        st.write('''***''')


    if mode == 'Single Prediction':
        st.header('Predicting on a single input')
        st.write('Specify the features values of the observation whose disposition you want to predict')
        
        fcol1, fcol2, fcol3 = st.columns(3,gap='medium')
        koi_duration = fcol1.number_input('Transit Duration [hrs]',
                                   min_value=0.00,value=3.01,step=0.01)

        koi_ror = fcol1.number_input('Planet-Star Radius Ratio',
                                   min_value=0.00,value=0.027,step=0.001)

        koi_prad = fcol1.number_input('Planetary Radius [Earth radii]',
                                   min_value=0.00,value=3.17,step=0.01)
        
        koi_sma = fcol1.number_input('Orbit Semi-Major Axis [au]',
                                   min_value=0.00,value=0.092,step=0.001)

        koi_dor = fcol1.number_input('Planet-Star Distance over Star Radius',
                                   min_value=0.00,value=15.06,step=0.01)
        
        koi_max_sngle_ev = fcol1.number_input('Maximum Single Event Statistic',
                                   min_value=0.00,value=7.235,step=0.001)

        koi_max_mult_ev = fcol2.number_input('Maximum Multiple Event Statistic',
                                   min_value=0.00,value=36.839,step=0.001)
        
        koi_model_snr = fcol2.number_input('Transit Signal-to-Noise',
                                   min_value=0.00,value=42.10,step=0.01)
        
        koi_num_transits = fcol2.number_input('Number of Transits',
                                   min_value=0.00,value=128.00,step=0.01)

        koi_fwm_stat_sig = fcol2.number_input('FW Offset Significance [percent]',
                                   min_value=0.00,value=00.00,step=0.01)

        koi_fwm_sdec = fcol3.number_input('FW Source &delta;(OOT) [deg]',
                                   min_value=0.00,value=42.96,step=0.01)
        
        koi_fwm_srao = fcol3.number_input('FW Source &Delta;&alpha;(OOT) [sec]',
                                   min_value=-800.00,value=0.34,step=0.01)
        
        koi_dicco_mra = fcol3.number_input('PRF &Delta;&alpha;<sub>SQ</sub>(OOT) [arcsec]',
                                   min_value=-50.0,value=-0.02,step=0.01)
        
        koi_dikco_msky = fcol3.number_input('PRF &Delta;&theta;<sub>SQ</sub>(KIC) [arcsec]',
                                   min_value=0.00,value=0.06,step=0.001)
        
        inp = pd.DataFrame({'koi_duration': [koi_duration],
                            'koi_ror': [koi_ror],
                            'koi_prad' : [koi_prad],'koi_sma': [koi_sma],
                            'koi_dor': [koi_dor],'koi_max_sngle_ev': [koi_max_sngle_ev],
                            'koi_max_mult_ev' : [koi_max_mult_ev],'koi_model_snr': [koi_model_snr],
                            'koi_num_transits': [koi_num_transits],
                            'koi_fwm_stat_sig': [koi_fwm_stat_sig],
                            'koi_fwm_sdec': [koi_fwm_sdec],'koi_fwm_srao': [koi_fwm_srao],
                            'koi_dicco_mra': [koi_dicco_mra],
                            'koi_dikco_msky'  : [koi_dikco_msky]
                            })

        inp_view = inp.copy()
        
        
        st.markdown('''

         ---
        ''')
        
        st.write('Below is the tabular view of your specified values:')
        st.dataframe(inp_view)

        if st.button("Predict on data input"):
            with st.spinner("Predicting... Please wait."):
                predic,relevant_explainers = single_predict(inp,loaded_models,explainers)

            disposition = REV_TARGET_MAP[predic]
            st.subheader('Results')
            st.write('The input observation is predicted to be a ' + disposition + ' exoplanet')

            st.write('Discover the features that influenced the predicted result:')
            with st.spinner("Inspecting model decision... please wait"):
                choice_explainer = relevant_explainers[0]
                shap_values = choice_explainer(inp)
                # use the first relevant explainer to for the plot
                st_shap(shap.plots.waterfall(shap_values[0]))
    


    if mode == 'Batch Prediction':
        st.header('Predicting on Uploaded File')

        st.write('This mode requires you to upload a csv file containing the Kepler Object of Interes\
                  observations whose dispositions you want to predict')
        st.write('Below is a table containing a guide on how the table should be structured')
        
        test_copy = hold_out_test.drop('koi_pdisposition',axis=1)
        
        data_type = list(np.squeeze(test_copy[SELECTED_FEATURES].dtypes))
        
        data_example = [str(test_copy[i].dropna().unique().tolist()[:5]).replace('[','').replace(']','') for i in SELECTED_FEATURES]
        
        featdesc_dict = dict(zip(feat_desc['Feature'].tolist(),feat_desc['Description'].tolist()))
        
        illust = pd.DataFrame({'Column Name':SELECTED_FEATURES,
                               'Data Type':data_type,'Examples':data_example})
        illust['Description'] = illust['Column Name'].map(featdesc_dict)
        
        illust['Data Type'] = np.where(illust['Data Type']=='object','String',illust['Data Type'])
        
        illust_ordered = illust[['Column Name','Description','Data Type','Examples']]
        st.dataframe(illust_ordered)
        st.write('Warning!!!')
        st.write('Uploading data with wrong data types will lead to unexpected dangerous results')

        u_data = st.file_uploader('Kindly provide a csv file containing the koi observations on which you want to predict',
                                  type='csv')
        
        if u_data is not None:
            data = pd.read_csv(u_data)

            st.subheader('Dataset Preview')
            data_ordered = data[SELECTED_FEATURES]
            st.write('Below is the data you uploaded')
            st.dataframe(data_ordered)           
             
            
            
            if st.button("Predict on uploaded data"):
                
                with st.spinner("Predicting... Please wait."):
                    predic = predict_batch(data_ordered,loaded_models)

                st.subheader('Results')
                num_candidates = predic['prediction'].sum()
                
                bpred_view = pd.DataFrame(predic['prediction'].map(REV_TARGET_MAP))
                
                bpred_view = bpred_view.rename(columns={'prediction':'Predicted disposition'})
                
                st.dataframe(bpred_view)
                total = bpred_view.shape[0]
                
                st.write(f'It is predicted that {num_candidates} out of the {total} observations \
                         are exoplanet candidates')


 

        else:
            st.write('No dataset Uploaded')



if __name__ == '__main__':
    main()
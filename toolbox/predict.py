import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import lime
import time
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, Booster
from lime import lime_text
import lime.lime_tabular
import joblib

# Données
path_df = "traindata.csv"

# Modèles sauvegardés   
modelXGboost = "XGboostS.sav"
modelKdtree = "MytreeS"

def load_modele(path):
    '''Renvoie le modele en tant qu\'objet à partir du chemin'''
    return joblib.load(modelXGboost)


def predict_update(ID, dataframe, feature, value):
    '''Renvoie la prédiction à partir d\'un vecteur X'''
    ID = int(ID)

    X = dataframe[dataframe.index == ID]
    X[feature] = value

    X = X.drop(['TARGET'], axis=1)

    XGboostMod = joblib.load(modelXGboost)
    
    prediction = XGboostMod.predict(np.array(X))
    pred = prediction[0]
    
    probabilite = XGboostMod.predict_proba(np.array(X))
    prob = probabilite[:,0][0]
       
    return pred, prob


def predict_flask(ID, dataframe):
    '''Fonction de prédiction utilisée par l\'API flask :
    a partir de l'identifiant et du jeu de données
    renvoie la prédiction à partir du modèle'''

    ID = int(ID)
    
    XGboostMod = joblib.load(modelXGboost)
    
    dataframeCopy = dataframe.copy()
    
    cols_when_model_builds = XGboostMod.get_booster().feature_names
    dataframe = dataframe[cols_when_model_builds]
    
    dataframe['SK_ID_CURR'] = dataframeCopy['SK_ID_CURR']
    #dataframe['TARGET'] = dataframeCopy['TARGET']
    
    dataframe.set_index(['SK_ID_CURR'],inplace=True)
       
    X = dataframe[dataframe.index == ID]

    #X = X.drop(['TARGET'], axis=1)
    
    prediction = XGboostMod.predict(np.array(X))
    pred = prediction[0]
    
    probabilite = XGboostMod.predict_proba(np.array(X))
    prob = probabilite[:,0][0]
    
    return pred, prob 
    

def clean_map(string):
    '''nettoyage des caractères de liste en sortie de LIME as_list'''
    signes = ['=>', '<=', '<', '>']
    for signe in signes :
        if signe in string :
            signe_confirme = signe
        string = string.replace(signe, '____')
    string = string.split('____')
    if string[0][-1] == ' ':
        string[0] = string[0][:-1]

    return (string, signe_confirme)


def interpretation(ID, dataframe, model, sample=False):
    '''Fonction qui fait appel à Lime à partir du modèle de prédiction et du jeu de données'''
    #préparation des données
    print('\n\n\n\n======== Nouvelle Instance d\'explicabilité ========')
    start_time = time.time()
    
    XGboostMod = joblib.load(modelXGboost)
	
    ID = int(ID)
    class_names = ['OK', 'default']
    

    dataframe_complet = dataframe.copy()
    dataframecompletResetIndex = dataframe.reset_index()
    
    
    print('ID client: {}'.format(ID))
     
    print('Temps initialisation : ', time.time() - start_time)
    start_time = time.time()

    #si on souhaite travailler avec un volume réduit de données    
    if sample is True :
        dataframe_reduced = dataframe[dataframe.index ==int(ID)]
        dataframe = pd.concat([dataframe_reduced, dataframe.sample(2000, random_state=20)], axis=0)
        del dataframe_reduced


    X = dataframe[dataframe.index == ID]
    X = X.drop(['TARGET'], axis=1)
    dataframe = dataframe.drop(['TARGET'], axis=1)
    

    #création de l'objet explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = np.array(dataframe.sample(int(0.1*dataframe.shape[0]), random_state=20)),
            feature_names = dataframe.columns,
            training_labels = dataframe.columns.tolist(),
            verbose=1,
            random_state=20,
            mode='classification')

    print('Temps initialisation explainer : ', time.time() - start_time)
    start_time = time.time()

    #explication du modèle pour l'individu souhaité
    exp = explainer.explain_instance(data_row = X.sort_index(axis=1).iloc[0:1,:].to_numpy().ravel(),
        predict_fn = model.predict_proba)
    
    print('Temps instance explainer : ', time.time() - start_time)
    start_time = time.time()

    #traitement des données et comparaison
    fig = exp.as_pyplot_figure()

    df_map = pd.DataFrame(exp.as_list())
    print(df_map)

    df_map['feature'] = df_map[0].apply(lambda x : clean_map(x)[0][0])
    df_map['signe'] = df_map[0].apply(lambda x : clean_map(x)[1])
    df_map['val_lim'] = df_map[0].apply(lambda x: clean_map(x)[0][-1])
    df_map['ecart'] = df_map[1]

    df_map = df_map[['feature', 'signe', 'val_lim', 'ecart']]
    #global
    df_map['contribution'] = 'normal'
    df_map.loc[df_map['ecart']>=0, 'contribution'] = 'default'
    
    df_map['customer_values'] = [X[feature].mean() for feature in df_map['feature'].values.tolist()]
    df_map['moy_global'] = [dataframe_complet[feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_en_regle'] = [dataframe_complet[dataframe_complet['TARGET'] == 0][feature].mean() for feature in df_map['feature'].values.tolist()]
    #clients en règle
    df_map['moy_defaut'] = [dataframe_complet[dataframe_complet['TARGET'] == 1][feature].mean() for feature in df_map['feature'].values.tolist()]
    #20 plus proches voisins
    index_plus_proches_voisins = nearest_neighbors(X, dataframe_complet, 20)
    # df_map['moy_voisins'] = [dataframe_complet[dataframe_complet['Unnamed: 0'].isin(index_plus_proches_voisins)][feature].mean() for feature in df_map['feature'].values.tolist()]
    list_SK_ID_CURR_20_Vois = [ dataframecompletResetIndex.loc[i]["SK_ID_CURR"]   for i in list(index_plus_proches_voisins) ]
    # df_map['moy_voisins'] = [dataframe_complet[dataframe_complet.index.isin(index_plus_proches_voisins)][feature].mean() for feature in df_map['feature'].values.tolist()]
    df_map['moy_voisins'] = [dataframecompletResetIndex[dataframecompletResetIndex["SK_ID_CURR"].isin (list_SK_ID_CURR_20_Vois) ][feature].mean() for feature in df_map['feature'].values.tolist()]

    print('Temps calcul données comparatives : ', time.time() - start_time)
    start_time = time.time()
    df_map = pd.concat([df_map[df_map['contribution'] == 'default'].head(3),
        df_map[df_map['contribution'] == 'normal'].head(3)], axis=0)

    return df_map.sort_values(by='contribution')



def correspondance_feature(feature_name):
    
    '''A partir du nom d\'une feature, trouve sa correspondance en français'''
    # df_correspondance = pd.read_csv(path_correspondance_features)
    # df_correspondance['Nom origine'] = df_correspondance['Nom origine'].str[1:]
    # try:
    #     return df_correspondance[df_correspondance['Nom origine'] == feature_name]['Nom français'].values[0]
    # except:
    #     print('correspondance non trouvée')
    return feature_name


def df_explain(dataframe):
    '''Ecrit une chaine de caractéres permettant d\'expliquer l\'influence des features dans le résultat de l\'algorithme '''

    chaine = '##Principales caractéristiques discriminantes##  \n'
    df_correspondance = pd.DataFrame(columns=['Feature','Nom francais'])
    for feature in dataframe['feature']:

        chaine += '### Caractéristique : '+ str(feature) + '('+ correspondance_feature(feature) +')###  \n'
        chaine += '* **Prospect : **'+ str(dataframe[dataframe['feature']==feature]['customer_values'].values[0])
        chaine_discrim = ' (seuil de pénalisation : ' + str(dataframe[dataframe['feature']==feature]['signe'].values[0])
        chaine_discrim +=  str(dataframe[dataframe['feature']==feature]['val_lim'].values[0])

        if dataframe[dataframe['feature']==feature]['contribution'].values[0] == 'default' :
            chaine += '<span style=\'color:red\'>' + chaine_discrim + '</span>  \n' 
        else : 
            chaine += '<span style=\'color:green\'>' + chaine_discrim + '</span>  \n' 

        #chaine += '* **Clients Comparables:**'+str(dataframe[dataframe['feature']==feature]['moy_voisins'].values[0])+ '  \n'
        #chaine += '* **Moyenne Globale:**'+str(dataframe[dataframe['feature']==feature]['moy_global'].values[0])+ '  \n'
        #chaine += '* **Clients réguliers :** '+str(dataframe[dataframe['feature']==feature]['moy_en_regle'].values[0])+ '  \n'
        #chaine += '* ** Clients avec défaut: **'+str(dataframe[dataframe['feature']==feature]['moy_defaut'].values[0])+ '  \n'
        #chaine += ''
        df_correspondance_line = pd.DataFrame(data = np.array([[feature, correspondance_feature(feature)]]), columns = ['Feature', 'Nom francais'])
        #df_correspondance_line = pd.DataFrame(data = {'Feature' : feature, 'Nom francais' : correspondance_feature(feature)})
        df_correspondance = pd.concat([df_correspondance, df_correspondance_line], ignore_index=True)
    return chaine, df_correspondance


def nearest_neighbors(X, dataframe, n_neighbors):
    '''Determine les plus proches voisins de l\'individu X 
    considere a partir d\'un KDTree sur 5 colonnes représentatives de caractéristiques intelligibles
    Renvoie en sortie les indices des k plus proches voisins'''
    
    cols = ['DAYS_BIRTH', 'AMT_INCOME_TOTAL','CODE_GENDER', 'INCOME_CREDIT_PERC']

    MytreeCopy = joblib.load(modelKdtree)
    dist, ind = MytreeCopy.query(np.array(X[cols]).reshape(1,-1), k = n_neighbors)
    return ind[0]


def graphes_streamlit(df):
    '''A partir du dataframe, affichage un subplot de 6 graphes représentatif du client comparé à d'autres clients sur 6 features'''
    f, ax = plt.subplots(2, 3, figsize=(10,10), sharex=False)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
    i = 0
    j = 0
    liste_cols = ['Client', 'Moyenne', 'En Règle', 'En défaut','Similaires']
    for feature in df['feature']:

        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(y = df[df['feature']==feature][['customer_values', 'moy_global', 'moy_en_regle', 'moy_defaut', 'moy_voisins']].values[0],
                   x = liste_cols,
                   ax = ax[i, j])
        sns.axes_style("white")

        if len(feature) >= 18:
            chaine = feature[:18]+'\n'+feature[18:]
        else : 
            chaine = feature
        if df[df['feature']==feature]['contribution'].values[0] == 'default':
            chaine += '\n(pénalise le score)'
            ax[i,j].set_facecolor('#ffe3e3') #contribue négativement
            ax[i,j].set_title(chaine, color='#990024')
        else:
            chaine += '\n(améliore le score)'
            ax[i,j].set_facecolor('#e3ffec')
            ax[i,j].set_title(chaine, color='#017320')
            
       
        if j == 2:
            i+=1
            j=0
        else:
            j+=1
        if i == 2:
            break;
    for ax in f.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
    if i!=2: #cas où on a pas assez de features à expliquer (ex : 445260)
        #
        True
    st.pyplot(f)

    return True
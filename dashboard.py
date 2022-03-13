#Lancement de cette application : streamlit run dashboard.py depuis le répertoire de l'application
import urllib
import streamlit as st
import numpy as np
import pandas as pd
import time
from urllib.request import urlopen
import json
from toolbox.predict import *
# supression librairies inutiles



# Chargement des données
path_df = "traindata.csv"
#df reduced : 10 % du jeu de donnees initial
path_df_reduced = "traindata.csv"

#mise en cache de la fonction pour exécution unique
@st.cache(allow_output_mutation=True)
def chargement_data(path):
    dataframe = pd.read_csv(path,index_col = 0)
    return dataframe

#mise en cache de la fonction pour exécution unique
@st.cache(hash_funcs={'xgboost.sklearn.XGBClassifier': id})
def chargement_explanation(id_input, dataframe, model, sample):
    return interpretation(str(id_input), 
        dataframe, 
        model, 
        sample=sample)

#mise en cache de la fonction pour exécution unique
@st.cache 
def chargement_ligne_data(id, df):
    return df[df.index==int(id)]

dataframe = chargement_data(path_df_reduced)

# liste_id = liste des clients, avec dataframe.index = ['SK_ID_CURR']
liste_id = dataframe.index.tolist()

#affichage formulaire
html_temp = """ 
<div style ="background-color:white;padding:13px"> 
<h1 style ="color:blue;text-align:center;">DASHBOARD - PREDICTION D'UNE DEMANDE DE PRET </h1> 
</div> 
"""         
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)

#st.title('DASHBOARD - SCORING CREDIT')
st.subheader("Prédiction du score d'un client")
st.subheader("vs")
st.subheader("Comparaison de ce score avec celui des autres clients ")
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
#id_input = str(st.selectbox('Choisissez un N° Client',options=dataframe.index.unique()))
chaine = "l'id Saisi est " + id_input
st.write(chaine)

# Génération d'échantillons de clients, respectivement en règle et en défaut.
sample_en_regle = str(list(dataframe[dataframe['TARGET'] == 0].sample(5)[['TARGET']].index.values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_regle = 'Exemples d\'id de clients en règle : ' +sample_en_regle
sample_en_defaut = str(list(dataframe[dataframe['TARGET'] == 1].sample(5)[['TARGET']].index.values)).replace('\'', '').replace('[', '').replace(']','')
chaine_en_defaut = 'Exemples d\'id de clients en défaut : ' + sample_en_defaut

if id_input == '': #lorsque rien n'a été saisi
    st.write(chaine_en_defaut)
    st.write(chaine_en_regle)

elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API

    #Appel de l'API : 
    
    API_url = "http://127.0.0.1:8080/credit/" + id_input
    #st.write(API_url)

    with st.spinner('Chargement du score du client...'):
        json_url = urlopen(API_url)

        API_data = json.loads(json_url.read())
        classe_predite = API_data['prediction']
        if classe_predite == 1:
            etat = 'client à risque'
        else:
            etat = 'client peu risqué'
        proba = 1-API_data['proba'] 

        #affichage de la prédictionn
        prediction = API_data['proba']
        classe_reelle = dataframe[dataframe.index == int(id_input)]['TARGET'].values[0]
        classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        chaine = 'Prédiction : **' + etat +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")

    #affichage de l'explication du score
    with st.spinner('Chargement des détails de la prédiction...'):
        explanation = chargement_explanation(str(id_input), 
        dataframe, 
        load_modele(modelXGboost), 
        sample=True)
    
    #Affichage des graphes    
    st.write(graphes_streamlit(explanation))

    st.subheader("Définition des groupes")
    st.markdown("\
    \n\
    * Client : la valeur pour le client considéré\n\
    * Moyenne : valeur moyenne pour l'ensemble des clients\n\
    * En Règle : valeur moyenne pour l'ensemble des clients en règle\n\
    * En Défaut : valeur moyenne pour l'ensemble des clients en défaut\n\
    * Similaires : valeur moyenne pour les 20 clients les plus proches du client\
    considéré sur les critères sexe/âge/revenu/durée/montant du crédit\n\n\
    ")

    st.sidebar.header("Modifier le profil client")
    st.sidebar.markdown('Cette section permet de modifier une des valeurs les plus caractéristiques du client et de recalculer son score')
    features = explanation['feature'].values.tolist()
    liste_features = tuple([''] + features)
    feature_to_update = ''
    feature_to_update = st.sidebar.selectbox('Quelle caractéristique souhaitez vous modifier', liste_features)

    if feature_to_update != '':
        default_value = explanation[explanation['feature'] == feature_to_update]['customer_values'].values[0]


        min_value = float(dataframe[feature_to_update].min())
        max_value = float(dataframe[feature_to_update].max())

        if (min_value, max_value) == (0,1): 
            step = float(1)
        else :
            step = float((max_value - min_value) / 20)

        update_val = st.sidebar.slider(label = 'Nouvelle valeur (valeur d\'origine : ' + str(default_value)[:4] + ')',
            min_value = float(dataframe[feature_to_update].min()),
            max_value = float(dataframe[feature_to_update].max()),
            value = float(default_value),
            step = step)

        if update_val != default_value:
            time.sleep(0.5)
            update_predict, proba_update = predict_update(id_input, dataframe, feature_to_update, update_val)
            if update_predict == 1:
                etat = 'client à risque'
            else:
                etat = 'client peu risqué'
            chaine = 'Nouvelle prédiction : **' + etat +  '** avec **' + str(round((1 - proba_update)*100)) + '%** de risque de défaut (classe réelle : '+str(classe_reelle) + ')'
            st.sidebar.markdown(chaine)

    #st.subheader('Informations relatives au client')
    df_client = chargement_ligne_data(id_input, dataframe).T
    df_client['nom_fr'] = [correspondance_feature(feature) for feature in df_client.index]
    #st.write(df_client)
        

else: 
    st.write('Identifiant non reconnu')
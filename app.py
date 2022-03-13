#APP FLASK (commande : flask run)

from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_wtf import Form
from wtforms.fields import StringField,BooleanField, PasswordField, TextAreaField
from wtforms.widgets import TextArea
import validators
from wtforms.validators import DataRequired
from toolbox.predict import *
import pandas as pd
import xgboost
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
#import socketio



# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b7776a'

#formulaire d'appel à l'API (facultatif)
class SimpleForm(Form):
    form_id = StringField('formulaire_id:', validators=[DataRequired()])
   
@app.route('/formulaire_id', methods=['GET', 'POST'])
def form():
    form = SimpleForm(request.form)
    print(form.errors)
    
    if request.method == 'POST':
        form_id=request.form['formulaire_id']
        print(form_id)
        return(redirect('credit/'+form_id)) 
    
    if form.validate():
        # Affichage de l'ID
        flash('Vous avez demandé l\'ID : ' + form_id)
        redirect('')
    else:
        flash('Veuillez compléter le champ. ')
    
    return render_template('formulaire_id.html', form=form)


# Chargement des données
#pathname = os.path.join("root", "directory1", "directory2")


path_df = "traindata.csv"
dataframe = pd.read_csv(path_df)

@app.route('/credit/<form_id>', methods=['GET'])
def credit(form_id):
   
    #calcul prédiction défaut et probabilité de défaut
    prediction, proba = predict_flask(form_id, dataframe)

    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba)
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)


#lancement de l'application
#if __name__ == "__main__":
#    port = os.environ.get("PORT",8080)
#    app.run(debug=False, host="0.0.0.1", port=port)
if __name__ == "__main__":
    app.run(debug=True,port=8080,use_reloader=False)

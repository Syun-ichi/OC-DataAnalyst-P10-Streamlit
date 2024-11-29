import streamlit as st
import pandas as pd
import numpy as np

import joblib

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression


# Charger le modèle sauvegardé : GradientBoostingRegressor
loaded_gbr : GradientBoostingRegressor = joblib.load('gradient_boosting_model.pkl')

# Charger le modèle sauvegardé : PolynomialFeatures
loaded_poly : PolynomialFeatures = joblib.load('poly_transformer.pkl')

# Charger le modèle sauvegardé : StandardScaler
loaded_scaler : StandardScaler = joblib.load('scaler.pkl')

# Charger les caractéristiques sélectionnées
selected_features_linear = joblib.load('selected_features.pkl')

# Charger le modèle sauvegardé : LogisticRegression
model_predicteur : LogisticRegression = joblib.load('best_logistic_regression_model.pkl')

# Charger les caractéristiques sélectionnées pour le modèle de prédiction
columns_predicteur = joblib.load('selected_features_logistic_regression.pkl')

# Charger le transformateur PolynomialFeatures pour le modèle de prédiction
loaded_poly_log : PolynomialFeatures = joblib.load('poly_transformer_logistic_regression.pkl')

# Charger le scaler pour le modèle de prédiction
loaded_scaler_log : StandardScaler = joblib.load('scaler_logistic_regression.pkl')


if __name__ == "__main__":

    st.title('Application de détection de faux billets')

    file = st.file_uploader('Télécharger le fichier de données', type=['csv'])

    data = None

    if file is not None:
        data = pd.read_csv(file, sep=',', encoding='utf-8')

        if data.columns.to_list() != ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length','id']:
            st.warning('Le fichier doit contenir les colonnes suivantes: diagonal, height_left, height_right, margin_low, margin_up, length, id')
            data = None

    if data is not None:

        if data['margin_low'].isna().sum() > 0:
            # Prédire les valeurs cibles pour les données où 'margin_low' est null
            columns = ['diagonal', 'height_left', 'height_right', 'margin_up', 'length']
            data_null_margin_low = data[data['margin_low'].isna()]
            X_null = data_null_margin_low[columns]
            X_null_poly = loaded_poly.transform(X_null)
            X_null_scaled = loaded_scaler.transform(X_null_poly)
            pred = loaded_gbr.predict(X_null_scaled[:, selected_features_linear])

            # Remplacer les valeurs manquantes
            data.loc[data['margin_low'].isna(), 'margin_low'] = pred

        # Prédire la variable is_genuine
        columns = ['margin_low', 'length', 'margin_up', 'diagonal', 'height_left', 'height_right']
        X = data[columns]
        X_poly_log = loaded_poly_log.transform(X)
        X_scaled_log = loaded_scaler_log.transform(X_poly_log)
        data['is_genuine'] = model_predicteur.predict(X_scaled_log[:, columns_predicteur])
        data['Taux de confiance'] = model_predicteur.predict_proba(X_scaled_log[:, columns_predicteur]).max(axis=1)*100

        st.subheader('Statistiques des données & prédictions')

        col1, col2 ,col3, col4= st.columns(4)

        col1.metric('Nombre de billets total', data.shape[0])

        col2.metric('Taux de vrai billets', f"{data['is_genuine'].mean()*100:.2f} %")

        col3.metric('Nombre de faux billets', data.loc[data['is_genuine']==False,'is_genuine'].count())

        col4.metric('Confiance moyenne', f"{data['Taux de confiance'].mean():.2f} %")

        with st.expander('Voir le détail des faux billets'):
            st.dataframe(data.loc[data['is_genuine']==False], 
                         column_config={
                             'Taux de confiance': st.column_config.NumberColumn(
                                 label='Taux de confiance (%)',
                                 format='%.2f %%')
                             })
        
        with st.expander('Voir le détail de tout les billets'):
            st.dataframe(data, 
                         column_config={
                             'Taux de confiance': st.column_config.NumberColumn(
                                 label='Taux de confiance (%)',
                                 format='%.2f %%')
                             })


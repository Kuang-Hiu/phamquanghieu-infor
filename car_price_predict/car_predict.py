import numpy as np
import pandas as pd
from datetime import date
from sklearn.cluster import KMeans
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load


def processing_data(df):
    # Fill Null value in df
    df['New_Price'].fillna(0, inplace=True)
    df['New_Price'] = df['New_Price'].apply(lambda x: 1 if x != 0 else 0)

    df['Mileage'].fillna(0, inplace=True)
    df['Engine'].fillna(0, inplace=True)
    df['Power'].fillna(0, inplace=True)
    df['Seats'].fillna(0, inplace=True)

    df['age'] = 2020- df['Year']
    df['age'] = df['age'].apply(lambda x: 15 if x > 14 else x)
    df = df.drop(['Year'], axis=1)

    #cluster = KMeans(n_clusters=5, random_state=0).fit(df[['Kilometers_Driven', 'age']])
    df['Kilometers_Driven_rate'] = 2

    df['Fuel_Type'] = df['Fuel_Type'].apply(lambda x: "Clean_Fuel" if x not in ['Diesel', 'Petrol'] else x)

    varlist = ['Location', 'age', 'Kilometers_Driven_rate', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Seats',
               'Company', 'Mileage', 'Engine', 'Power']
    X = df[varlist].copy()

    X.Transmission.replace({'Manual': 0, 'Automatic': 1}, inplace=True)
    X.Owner_Type.replace({'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}, inplace=True)

    location = {'Ahmedabad': 0,
                 'Bangalore': 1,
                 'Chennai': 2,
                 'Coimbatore': 3,
                 'Delhi': 4,
                 'Hyderabad': 5,
                 'Jaipur': 6,
                 'Kochi': 7,
                 'Kolkata': 8,
                 'Mumbai': 9,
                 'Pune': 10}

    X.Location.replace(location, inplace=True)

    Fuel_Type = {'Clean_Fuel': 0, 'Diesel': 1, 'Petrol': 2}
    X.Fuel_Type.replace(Fuel_Type, inplace=True)

    Company = {'AMBASSADOR': 0,
                 'AUDI': 1,
                 'BENTLEY': 2,
                 'BMW': 3,
                 'CHEVROLET': 4,
                 'DATSUN': 5,
                 'FIAT': 6,
                 'FORCE': 7,
                 'FORD': 8,
                 'HONDA': 9,
                 'HYUNDAI': 10,
                 'ISUZU': 11,
                 'JAGUAR': 12,
                 'JEEP': 13,
                 'LAMBORGHINI': 14,
                 'LAND': 15,
                 'MAHINDRA': 16,
                 'MARUTI': 17,
                 'MERCEDES-BENZ': 18,
                 'MINI': 19,
                 'MITSUBISHI': 20,
                 'NISSAN': 21,
                 'PORSCHE': 22,
                 'RENAULT': 23,
                 'SKODA': 24,
                 'SMART': 25,
                 'TATA': 26,
                 'TOYOTA': 27,
                 'VOLKSWAGEN': 28,
                 'VOLVO': 29}

    X.Company.replace(Company, inplace=True)

    X_final = X.copy()

    scaler = load('./car_price_predict/pkl/std_scaler_machine.bin')

    machine_data = scaler.transform(X_final[['Engine', 'Power', 'Mileage']])

    pca = load('./car_price_predict/pkl/pca.bin')
    x_pca = pca.transform(machine_data)

    del X_final['Engine']
    del X_final['Power']
    del X_final['Mileage']

    sc = load('./car_price_predict/pkl/x_std_scaler.bin')
    X_final['PCA_1'] = x_pca[:, 0]
    X_final['PCA_2'] = x_pca[:, 1]
    X_final_scaled = sc.transform(X_final)

    return X_final_scaled

def car_price(X):
    model = pickle.load(open("./car_price_predict/pkl/rf_reg.pkl", 'rb'))
    results_scaled = model.predict(X)
    sc = load('./car_price_predict/pkl/y_std_scaler.bin')
    results = sc.inverse_transform([results_scaled])
    return results[0][[0]]


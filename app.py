from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
# import sklearn
app = Flask(__name__)
car_data = pd.read_csv('updated_car_data.csv')
car_eda = pd.read_csv('cleaned_eda3.csv')
# finalModel = pickle.load(open('finalModel.pkl', 'rb'))
# print(car_data)

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

model = pickle.load(open('finalModel2.pkl', 'rb'))

@app.route("/")
def home():
    name = 'devin'
    return render_template('index.html', name = name)

@app.route("/api/count", methods=["POST"])
def count():
    # data_feature = request.get_data().decode('utf-8')
    
    data_feature = request.form.get('count_feature_value')
    data_charttype = request.form.get('count_charttype_value')
    # print(request.form)
    
    
    if data_charttype == 'bar':
        if data_feature == 'Mileage':
            mileage = car_eda['bins_mileage'].value_counts().sort_index()
            appended = mileage[-1:].append(mileage[:-1])
            
            index = np.array(appended.index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(appended.values)
            json_value = json.dumps(values.tolist())  
        elif data_feature == 'Prod. year':
            year_counts = car_eda['Prod. year'].value_counts()
            appended = year_counts[0:1].append(year_counts[1:].sort_index())
            index = np.array(appended.index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(appended.values)
            json_value = json.dumps(values.tolist())  
            
        elif data_feature == 'Cylinders':
            cylinder = car_eda['Cylinders'].value_counts().sort_index()
            appended = cylinder[-1:].append(cylinder[:-1])
            
            index = np.array(appended.index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(appended.values)
            json_value = json.dumps(values.tolist())
            
        elif data_feature == 'Airbags':
            airbags = car_eda['bins_airbags'].value_counts().sort_index()
            appended = airbags[-1:].append(airbags[1:3])
            appended = appended.append(airbags[0:1])
            index = np.array(appended.index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(appended.values)
            json_value = json.dumps(values.tolist())
            
        elif data_feature == 'Doors':
            index = np.array(car_eda['Doors'].value_counts().sort_index().index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(car_eda['Doors'].value_counts().sort_index().values)
            json_value = json.dumps(values.tolist())
        elif data_feature == 'Engine_volume':
            engine_counts = car_eda['Engine_volume'].value_counts().sort_index()
            appended = engine_counts[-1:].append(engine_counts[0:-1])
            index = np.array(appended.index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(appended.values)
            json_value = json.dumps(values.tolist()) 
        else:
            index = np.array(car_eda[data_feature].value_counts().index)
            json_data = json.dumps(index.tolist())
            
            values = np.array(car_eda[data_feature].value_counts().values)
            json_value = json.dumps(values.tolist())    
    
    elif data_charttype == 'pie':
        if data_feature == 'Mileage':
            index = np.array(car_eda['bins_mileage'].value_counts().index)
            json_data = json.dumps(index.tolist())
                
            # values = np.array(car_eda['bins_mileage'].value_counts().values)
            values = np.round((car_eda['bins_mileage'].value_counts().values / car_eda['bins_mileage'].value_counts().sum())*100, 2)
            json_value = json.dumps(values.tolist())
            
        elif data_feature == 'Airbags':
            index = np.array(car_eda['bins_airbags'].value_counts().index)
            json_data = json.dumps(index.tolist())
                
            values = np.array(car_eda['bins_airbags'].value_counts().values)
            json_value = json.dumps(values.tolist())
            
        else:
            index = np.array(car_eda[data_feature].value_counts().index)
            json_data = json.dumps(index.tolist())
                
            values = np.array(car_eda[data_feature].value_counts().values)
            json_value = json.dumps(values.tolist())  
    
    print(json_value)
    
    data = {'message': json_data,
            'value' : json_value
            }
    return data

@app.route("/api/price", methods=["POST"])
def price():
    data_feature = request.get_data().decode('utf-8')
    
    if data_feature == 'Mileage':
        mileage = car_eda.groupby('bins_mileage')['Price'].mean()
        appended = mileage[-1:].append(mileage[0:-1])
        
        index = np.array(appended.index)
        json_data = json.dumps(index.tolist())
        
        average_prices = np.array(appended)
        json_value = json.dumps(average_prices.tolist())
        
    elif data_feature == 'Airbags':
        airbags = car_eda.groupby('bins_airbags')['Price'].mean()
        
        appended = airbags[-1:].append(airbags[1:3])
        appended = appended.append(airbags[0:1])
        index = np.array(appended.index)
        json_data = json.dumps(index.tolist())
        
        average_prices = np.array(appended)
        json_value = json.dumps(average_prices.tolist())
        
    elif data_feature == 'Prod. year':
        years = car_eda.groupby('Prod. year')['Price'].mean()
        appended = years[-1:].append(years[0:-1])
        
        index = np.array(appended.index)
        json_data = json.dumps(index.tolist())
        
        average_prices = np.array(appended)
        json_value = json.dumps(average_prices.tolist())
    
    elif data_feature == 'Engine_volume':
        engine = car_eda.groupby('Engine_volume')['Price'].mean()
        appended = engine[-1:].append(engine[0:-1])
        
        index = np.array(appended.index)
        json_data = json.dumps(index.tolist())
        
        average_prices = np.array(appended)
        json_value = json.dumps(average_prices.tolist())
        
    else:
        index = np.array(car_eda.groupby(data_feature)['Price'].mean().index)
        json_data = json.dumps(index.tolist())
        
        average_prices = np.array(car_eda.groupby(data_feature)['Price'].mean())
        json_value = json.dumps(average_prices.tolist())
    
    # print(json_value)
    
    data = {'message': json_data,
            'value' : json_value
            }
    return data
    

@app.route("/data")
def data():
    # categories = car_data['Model']
    
    levies = car_data['Levy'].value_counts()
    manufacturers = car_data['Manufacturer'].value_counts()
    models = car_data['Model'].value_counts()
    prod_years = car_data['Prod. year'].value_counts()
    categorys = car_data['Category'].value_counts()
    leather_interiors = car_data['Leather interior'].value_counts()
    fuel_types = car_data['Fuel type'].value_counts()
    engine_volumes = car_data['Engine volume'].value_counts()
    mileages = car_data['Mileage'].value_counts()
    cylinders = car_data['Cylinders'].value_counts()
    gear_box_types = car_data['Gear box type'].value_counts()
    drive_wheels = car_data['Drive wheels'].value_counts()
    doors = car_data['Doors'].value_counts()
    wheels = car_data['Wheel'].value_counts()
    colors = car_data['Color'].value_counts()
    airbags = car_data['Airbags'].value_counts()
    
    levies_index = levies.index 
    manufacturers_index = manufacturers.index 
    models_index = models.index 
    prod_years_index = prod_years.index 
    categorys_index = categorys.index 
    leather_interiors_index = leather_interiors.index 
    fuel_types_index = fuel_types.index 
    engine_volumes_index = engine_volumes.index 
    mileages_index = mileages.index 
    cylinders_index = cylinders.index 
    gear_box_types_index = gear_box_types.index 
    drive_wheels_index = drive_wheels.index 
    doors_index = doors.index 
    wheels_index = wheels.index 
    colors_index = colors.index 
    airbags_index = airbags.index 
    
    levies_values = levies.values 
    manufacturers_values = manufacturers.values 
    models_values = models.values 
    prod_years_values = prod_years.values 
    categorys_values = categorys.values 
    leather_interiors_values = leather_interiors.values 
    fuel_types_values = fuel_types.values 
    engine_volumes_values = engine_volumes.values 
    mileages_values = mileages.values 
    cylinders_values = cylinders.values 
    gear_box_types_values = gear_box_types.values 
    drive_wheels_values = drive_wheels.values 
    doors_values = doors.values 
    wheels_values = wheels.values 
    colors_values = colors.values 
    airbags_values = airbags.values 
    
    # print((levies_index.to_list()))
    # print(np.array(car_data['Color'].value_counts().index))
    
    # print(car_eda.columns)
    
    features = np.array(car_eda.columns[2:-3])
    
    
    return render_template('data.html', 
        features = features,
        levies_index = car_data['Levy'].unique(), 
        manufacturers_index = manufacturers_index, 
        models_index = models_index, 
        prod_years_index = prod_years_index, 
        categorys_index = categorys_index, 
        leather_interiors_index = leather_interiors_index, 
        fuel_types_index = fuel_types_index, 
        engine_volumes_index = engine_volumes_index, 
        mileages_index = mileages_index, 
        cylinders_index = cylinders_index, 
        gear_box_types_index = gear_box_types_index, 
        drive_wheels_index = drive_wheels_index, 
        doors_index = doors_index, 
        wheels_index = wheels_index, 
        colors_index = colors_index, 
        airbags_index = airbags_index,
        levies_values = levies_values, 
        manufacturers_values = manufacturers_values, 
        models_values = models_values, 
        prod_years_values = prod_years_values, 
        categorys_values = categorys_values, 
        leather_interiors_values = leather_interiors_values, 
        fuel_types_values = fuel_types_values, 
        engine_volumes_values = engine_volumes_values, 
        mileages_values = mileages_values, 
        cylinders_values = cylinders_values, 
        gear_box_types_values = gear_box_types_values, 
        drive_wheels_values = drive_wheels_values, 
        doors_values = doors_values, 
        wheels_values = wheels_values, 
        colors_values = colors_values, 
        airbags_values = airbags_values
    )


@app.route("/prediction")
def prediction():

    levies = car_data['Levy'].unique()
    manufacturers = car_data['Manufacturer'].unique()
    models = car_data['Model'].unique()
    prod_years = car_data['Prod. year'].unique()
    categorys = car_data['Category'].unique()
    leather_interiors = car_data['Leather interior'].unique()
    fuel_types = car_data['Fuel type'].unique()
    engine_volumes = car_data['Engine volume'].unique()
    mileages = car_data['Mileage'].unique()
    cylinders = car_data['Cylinders'].unique()
    gear_box_types = car_data['Gear box type'].unique()
    drive_wheels = car_data['Drive wheels'].unique()
    doors = car_data['Doors'].unique()
    wheels = car_data['Wheel'].unique()
    colors = car_data['Color'].unique()
    airbags = car_data['Airbags'].unique()
    # turbos = car_data['Turbo'].unique()

    return render_template('prediction.html',
        levies = levies,
        manufacturers = manufacturers,
        models = models,
        prod_years = prod_years,
        categorys = categorys,
        leather_interiors = leather_interiors,
        fuel_types = fuel_types,
        engine_volumes = engine_volumes,
        mileages = mileages,
        cylinders = cylinders,
        gear_box_types = gear_box_types,
        drive_wheels = drive_wheels,
        doors = doors,
        wheels = wheels,
        colors = colors,
        airbags = airbags,
        # turbos = turbos
    )

@app.route('/predict', methods=['POST'])
def predict():
    # company = request.form.get('company')
    # car_model = request.form.get('car_model')
    # year = request.form.get('year')
    # fuel_type = request.form.get('fuel_type')
    # kms_driven = int(request.form.get('kilo_driven'))
    
    Levy = request.form.get('levy')
    Manufacturer = request.form.get('manufacturer')
    Model = request.form.get('model')
    Prod_year = request.form.get('prod_year')
    Category = request.form.get('category')
    Leather_interior = request.form.get('leather_interior')
    Fuel_type = request.form.get('fuel_type')
    Engine_volume = request.form.get('engine_volume')
    Mileage = request.form.get('mileage')
    Cylinders = request.form.get('cylinder')
    Gear_box_type = request.form.get('gear_box_type')
    Drive_wheels = request.form.get('drive_wheel')
    Doors = request.form.get('door')
    Wheel = request.form.get('wheel')
    Color = request.form.get('color')
    Airbags = request.form.get('airbag')
    Turbo = request.form.get('turbo')
    
    test = pd.DataFrame(columns=['Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage', 'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color', 'Airbags', 'Turbo'], data = np.array([Levy, Manufacturer, Model, Prod_year, Category, Leather_interior, Fuel_type, Engine_volume, Mileage, Cylinders, Gear_box_type, Drive_wheels, Doors, Wheel, Color, Airbags, Turbo]).reshape(1,17))
    
    test['Levy'] = test['Levy'].astype('float64')
    test['Prod. year'] = test['Prod. year'].astype('int64')
    test['Leather interior'] = test['Leather interior'].astype('bool')
    test['Turbo'] = test['Turbo'].map({'False':False, 'True':True})
    test['Engine volume'] = test['Engine volume'].astype('float64')
    test['Cylinders'] = test['Cylinders'].astype('float64')
    test['Mileage'] = test['Mileage'].astype('int64')
    test['Doors'] = test['Doors'].astype('int64')
    test['Airbags'] = test['Airbags'].astype('int64')

    prediction = model.predict(test)
    print(prediction)

    return str(np.round(prediction[0], 2))
    # return ""


if __name__ == "__main__":
    app.run(debug=True)

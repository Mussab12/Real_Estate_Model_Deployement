import pickle

import pandas as pd
from flask import Flask, render_template,request

application=Flask(__name__)
model=pickle.load(open("RandomRegressorModel.pkl",'rb'))
House=pd.read_csv("Clean_data1.csv")
@application.route('/')
def index():
    Bedrooms=sorted(House['Bedrooms'].unique())
    Bathrooms=sorted(House['Bathrooms'].unique())
    Parking_Space=sorted(House['Parking Space'].unique())
    Land_Size=sorted(House['Land Size'].unique())
    Building_Size=sorted(House['Building Size'].unique())
    Features=House['Features'].unique()
    Type=House['Type'].unique()
    return render_template('index.html',Bedrooms=Bedrooms,Bathrooms=Bathrooms,Parking_Space=Parking_Space,Land_Size=Land_Size,Building_Size=Building_Size,Features=Features,Type=Type)

@application.route('/predict',methods=['POST'])
def predict():
    Bedrooms=int(request.form.get('Bedrooms'))
    Bathrooms = int(request.form.get('Bathrooms'))
    Parking_Space = int(request.form.get('Parking_Space'))
    Land_Size = float(request.form.get('Land_Size'))
    Building_Size = float(request.form.get('Building_Size'))
    Features = request.form.get('Features')
    Type = request.form.get('Type')
    print(Bedrooms,Bathrooms,Parking_Space,Land_Size,Building_Size,Features,Type)
    prediction=model.predict(pd.DataFrame([[Bedrooms,Bathrooms,Parking_Space,Land_Size,Building_Size,Features,Type]],columns=['Bedrooms','Bathrooms','Parking Space','Land Size','Building Size','Features','Type']))
    print(prediction)


    return str(prediction[0])

if __name__=="__main__":
    application.run(debug=True)
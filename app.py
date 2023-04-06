from flask import Flask, request, render_template
import pandas as pd
import joblib
import math as p
import pickle
import json
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# Declare a Flask app
app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("Home.html")


@app.route("/crop")
def crop():
    return render_template("crop.html")


@app.route('/main', methods=['GET', 'POST'])
def main():

    # If a form is submitted
    if request.method == "GET":

        # Unpickle classifier
        # knn = joblib.load("crop.pkl")

        # with open('encoder.pkl', 'rb') as f:
        #     encoder = pickle.load(f)

        # Get values through input bars
        nitrogen = request.args.get("nitrogen")
        phosphorous = request.args.get("phosphorous")
        pottassium = request.args.get("pottassium")
        tempearture = request.args.get("tempearture")
        humdity = request.args.get("humdity")

        dataset = pd.read_csv('cropdata.csv')

        # Prepare data
        X = dataset.drop(['CROP', 'N', 'P', 'K'], axis=1)
        y = dataset[['N', 'P', 'K']]
        z = dataset[['CROP']]
        
        encoder = LabelEncoder()

        data_encoded = pd.DataFrame({'fruit_encoded': encoder.fit_transform(dataset['CROP'])})

        # Concatenate the original dataframe with the encoded dataframe
        y = pd.concat([y, data_encoded], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
        # model = KNeighborsRegressor()
        # model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

        # # Put inputs to dataframe
        X = pd.DataFrame([[tempearture, humdity]], columns=["TEMPERATURE", "HUMIDITY"])

        # # Get prediction
        prediction = model.predict(X)

        val=encoder.inverse_transform([p.ceil(float(prediction[0][3]))])[0]
        li=[]
        for i in range(3):
            li.append(prediction[0][i])
        li.append(val)

        s="{nit},{pho},{pot},{cro}".format(nit=li[0],pho=li[1],pot=li[2],cro=li[3])


        # stra = ' '.join([str(elem) for elem in li])

        # json_s = json.dumps(stra)
        # jso=json.loads(json_s)

    else:
        prediction = "ERROR"

    return render_template("crop.html", da=s)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)

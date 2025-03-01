from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the pickle 

with open('zomato_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
categorical_cols =["book_table","online_order","rest_type","listed_in(type)","listed_in(city)"]
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = joblib.load(f'{col}_label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':        
        online_order = request.form['online_order']
        book_table = request.form['book_table']
        rate = float(request.form['rate'])
        votes = int(request.form['votes'])
        rest_type = request.form['rest_type']
        approx_cost = float(request.form['approx_cost'])
        listed_in_type = request.form['listed_in_type']
        listed_in_city = request.form['listed_in_city']
        num_cuisines = int(request.form['num_cuisines'])
        input_df = pd.DataFrame({"online_order":[online_order],"book_table":[book_table],"rate":[rate],"votes":[votes],"rest_type":[rest_type],"approx_cost(for two people)":[approx_cost],"listed_in(type)":[listed_in_type],"listed_in(city)":[listed_in_city],"num_cuisines":[num_cuisines]})
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])
        print(input_df)
        prediction = model.predict(input_df)
        print(prediction)
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
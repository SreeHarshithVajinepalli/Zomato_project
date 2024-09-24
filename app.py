from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the pickle 

with open('zomato_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('zomato_encoder.pkl', 'rb') as encoder_file:
    le = pickle.load(encoder_file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input from the form
        rate = float(request.form['rate'])
        votes = int(request.form['votes'])
        approx_cost = float(request.form['approx_cost'])
        rest_type = request.form['rest_type']
        listed_in_type = request.form['listed_in_type']
        listed_in_city = request.form['listed_in_city']
        num_cuisines = int(request.form['num_cuisines'])
        online_order = request.form['online_order']
        book_table = request.form['book_table']
        
        print(rest_type,type(rest_type))
        # Encode categorical variables
        a=np.array([book_table,online_order,rest_type,listed_in_type,listed_in_city])
        arr=pd.Series(a)
        book_table,online_order,rest_type,listed_in_type,listed_in_city=le.transform(arr)
        # listed_in_city = le.transform([listed_in_city])[0]
        # rest_type = le.transform([rest_type])[0]
        # listed_in_type = le.transform([listed_in_type])[0]
        
        # online_order = le.transform([online_order])[0]
        # book_table = le.transform([book_table])[0]

        print(rest_type,type(rest_type))
        
        # Create input array
        input_data = np.array([[online_order, book_table, rate, votes, approx_cost, 
                                rest_type, listed_in_type, listed_in_city, num_cuisines]])
        # input_data=np.array([[1,1,4.1,776,800,4,0,10,3]])
        # Scale the input 

        # Make prediction
        prediction = model.predict(input_data)
        print(prediction)
        return render_template('result.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
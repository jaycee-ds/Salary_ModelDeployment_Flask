from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('experienceform.html')

@app.route('/', methods=['POST'])
def predict():
    user_input = float(request.form['text'])
    prediction = model.predict([1.0, user_input])
    
    output = round(prediction[0], 2)
    
    return str(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)


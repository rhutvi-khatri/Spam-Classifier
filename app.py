from flask import Flask,render_template,request,jsonify
import joblib,pickle

app = Flask(__name__)

model = joblib.load('model.pkl')
vector = pickle.load(open('vector.pkl','rb'))
@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/results',methods=['POST'])
def pred():
    if request.method == 'POST':

        input_text = request.form['input']
        input_text = [input_text]  # Put the input into a list
        # Use the vectorizer to transform the text
        input_vectorized = vector.transform(input_text)
        prediction = model.predict(input_vectorized)

    return render_template('result.html',ans=prediction)

if __name__ == '__main__':
    app.run(debug=True)
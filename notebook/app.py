from flask import Flask,render_template,request
import pickle
tokenizer = pickle.load(open(r"notebook\model\tfidf_vectorizer.pkl","rb"))
model = pickle.load(open(r"notebook\model\multinomailNB_tfidf_tweet.pkl","rb"))
app =Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    text = request.form.get("text-content")
    tokenized_text = tokenizer.transform([text])
    prediction = model.predict(tokenized_text)
    prediction = "Disaster tweet" if prediction ==1 else "Normal tweet "
    return render_template("index.html",prediction=prediction,text=text)

if __name__=="__main__":
    app.run(debug=True)
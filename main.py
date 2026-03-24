from flask import render_template,Flask,request
from src.pipeline.pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        print("GET request hit")
        return render_template('home.html', results=None, prob=None)
    
    else:
       data = CustomData(
        Satisfaction_Score = int(request.form.get('Satisfaction_Score')),
        Churn_Score = int(request.form.get('Churn_Score')),
        Tenure_in_Months = int(request.form.get('Tenure_in_Months')),
        Monthly_Charge = float(request.form.get('Monthly_Charge')),
        Total_Long_Distance_Charges = float(request.form.get('Total_Long_Distance_Charges')),
        Number_of_Referrals = int(request.form.get('Number_of_Referrals')),
        CLTV = int(request.form.get('CLTV')),
        Age = int(request.form.get('Age')),
        Dependents = int(request.form.get('Dependents')),
        Internet_Service = int(request.form.get('Internet_Service')),
        Gender = request.form.get('Gender')
        )      
       
       pred_data = data.get_data_as_data_frame()
       prediction = PredictPipeline()
       preds, prob = prediction.predict(pred_data)

        # Extract values
       result = "Churn" if preds[0] == 1 else "No Churn"
       probability = round(prob[0][1] * 100, 2)

    return render_template(
        'home.html',
        results=result,
        prob=probability
        )
        
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


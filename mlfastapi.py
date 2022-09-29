from fastapi import FastAPI
import uvicorn
import joblib


app = FastAPI(debug=True)

@app.get('/')
def home():
    return {'text':'Claim risk prediction web app'}

@app.get('/predict')
def predict(PREMIUM_CHARGED:float, NO_CLAIM_DISCOUNT:float, SPECIAL_DISCOUNT:float, SUM_INSURED:float, 
            LOADINGS:float, FLAG_POLICY_TYPE_OF_COVER_CHANGE_IN_LAST_12_MONTHS: int, POLICY_HISTORY_AGE:int, 
            POLICY_TERM:int, COMPANY_LOADINGS:float, type_of_cover:int, transaction_type:int, 
            client_gender:int, open_restricted_flag:int, pa_driver_included:int, vehicle_condition:int, 
            flag_vehicle_condition:int, flag_vehicle_type_on_seating_capacity:int):
    loaded_model = joblib.load('rf.pkl')
    make_prediction = loaded_model.predict([[PREMIUM_CHARGED,NO_CLAIM_DISCOUNT,SPECIAL_DISCOUNT,SUM_INSURED,
                                             LOADINGS,FLAG_POLICY_TYPE_OF_COVER_CHANGE_IN_LAST_12_MONTHS,
                                             POLICY_HISTORY_AGE,POLICY_TERM,COMPANY_LOADINGS,type_of_cover,
                                             transaction_type,client_gender,open_restricted_flag,
                                             pa_driver_included,vehicle_condition,flag_vehicle_condition,
                                             flag_vehicle_type_on_seating_capacity]])
    
    output = round(make_prediction[0],2)
    if (make_prediction[0] == 0):
      return 'The policy is not going to claim'
    else:
      return 'The policy is going to be claimed'
    
    
if __name__ =='__main__':
    uvicorn.run(app)
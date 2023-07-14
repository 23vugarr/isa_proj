from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
from dataclasses import dataclass
from typing import Optional
from numpy import float64
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class VARIABLES:
    model_filename = './model.pkl'
    model_input = './df.xlsx'


class ModelInput(BaseModel):
    Annual_Income: Optional[float64] = None
    Monthly_Inhand_Salary: Optional[float64] = None
    Num_Bank_Accounts: Optional[float64] = None
    Num_Credit_Card: Optional[float64] = None
    Interest_Rate: Optional[float64] = None
    Num_of_Loan: Optional[float64] = None
    Delay_from_due_date: Optional[float64] = None
    Num_of_Delayed_Payment: Optional[float64] = None
    Changed_Credit_Limit: Optional[float64] = None
    Num_Credit_Inquiries: Optional[float64] = None
    Credit_Mix: Optional[object] = None
    Outstanding_Debt: Optional[float64] = None
    Credit_Utilization_Ratio: Optional[float64] = None
    Credit_History_Age: Optional[float64] = None
    Payment_of_Min_Amount:Optional[object]=None
    Total_EMI_per_month: Optional[float64] = None
    Amount_invested_monthly: Optional[float64] = None
    Monthly_Balance: Optional[float64] = None



model = pickle.load(open(VARIABLES.model_filename, "rb"))


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    original_data = pd.read_excel(VARIABLES.model_input)
    original_data = original_data.drop(columns = ['ID', 'Type_of_Loan', 'Payment_Behaviour', 'Age','SSN','Customer_ID', 'Occupation','Name', 'Month', 'Credit_Score'], axis = 1)
    data = pd.concat([original_data, data], axis = 0)
    data = data.reset_index(drop = True)
    
    data = pd.get_dummies(data, columns=['Payment_of_Min_Amount'], drop_first = True)
    le = LabelEncoder()
    data['Credit_Mix'] = le.fit_transform(data.Credit_Mix)


    col = data.columns
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns = col)
    print(data)

    return data.iloc[-1,:].values.reshape(1,-1)


@app.get("/predict")
def root(x_model: ModelInput):
    x_model = x_model.json()
    x_model = json.loads(x_model)
    data = pd.DataFrame.from_dict(x_model, orient='index').T

    data = preprocess(data)
    result = model.predict(data)
    result = result.tolist()

    return {
        "the result of model": result[0]
    }
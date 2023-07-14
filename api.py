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
    Annual_Income: Optional[float] = None
    Monthly_Inhand_Salary: Optional[float] = None
    Num_Bank_Accounts: Optional[float] = None
    Num_Credit_Card: Optional[float] = None
    Interest_Rate: Optional[float] = None
    Num_of_Loan: Optional[float] = None
    Delay_from_due_date: Optional[float] = None
    Num_of_Delayed_Payment: Optional[float] = None
    Changed_Credit_Limit: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = None
    Credit_Mix: Optional[object] = None
    Outstanding_Debt: Optional[float] = None
    Credit_Utilization_Ratio: Optional[float] = None
    Credit_History_Age: Optional[float] = None
    Payment_of_Min_Amount: Optional[object] = None
    Total_EMI_per_month: Optional[float] = None
    Amount_invested_monthly: Optional[float] = None
    Monthly_Balance: Optional[float] = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v

    @classmethod
    def __get_pydantic_json_schema__(cls, *, by_alias: bool = True) -> dict:
        field_schema = super().__get_pydantic_json_schema__(by_alias=by_alias)
        field_schema["type"] = "number"
        return field_schema

model = pickle.load(open(VARIABLES.model_filename, "rb"))


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    original_data = pd.read_excel(VARIABLES.model_input)
    original_data = original_data.drop(columns=['ID', 'Type_of_Loan', 'Payment_Behaviour', 'Age', 'SSN', 'Customer_ID',
                                                'Occupation', 'Name', 'Month', 'Credit_Score'], axis=1)
    data = pd.concat([original_data, data], axis=0)
    data = data.reset_index(drop=True)

    data = pd.get_dummies(data, columns=['Payment_of_Min_Amount'], drop_first=True)
    le = LabelEncoder()
    data['Credit_Mix'] = le.fit_transform(data.Credit_Mix)

    col = data.columns
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=col)

    return data.iloc[-1, :].values.reshape(1, -1)


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

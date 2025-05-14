from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

app = FastAPI()

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

class EmployeeData(BaseModel):
    Age: int
    Job_Level: int
    Monthly_Income: int
    Hourly_Rate: int
    Years_at_Company: int
    Years_in_Current_Role: int
    Years_Since_Last_Promotion: int
    Work_Life_Balance: int
    Job_Satisfaction: int
    Performance_Rating: int
    Training_Hours_Last_Year: int
    Project_Count: int
    Average_Hours_Worked_Per_Week: int
    Absenteeism: int
    Work_Environment_Satisfaction: int
    Relationship_with_Manager: int
    Job_Involvement: int
    Distance_From_Home: int
    Number_of_Companies_Worked: int
    Attrition_Yes: int
    Gender_Male: int
    Marital_Status_Married: int
    Marital_Status_Single: int
    Department_HR: int
    Department_IT: int
    Department_Sales: int
    Job_Role_Assistant: int
    Job_Role_Executive: int
    Job_Role_Manager: int
    Overtime_Yes: int

class PredictionResponse(BaseModel):
    Attrition_Prediction: str
    Probability: float

@app.get("/")
def read_root():
    return {"message": "🚀 Welcome! The Attrition Prediction API is running successfully."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: EmployeeData):
    input_data = np.array([[  
        data.Age,
        data.Job_Level,
        data.Monthly_Income,
        data.Hourly_Rate,
        data.Years_at_Company,
        data.Years_in_Current_Role,
        data.Years_Since_Last_Promotion,
        data.Work_Life_Balance,
        data.Job_Satisfaction,
        data.Performance_Rating,
        data.Training_Hours_Last_Year,
        data.Project_Count,
        data.Average_Hours_Worked_Per_Week,
        data.Absenteeism,
        data.Work_Environment_Satisfaction,
        data.Relationship_with_Manager,
        data.Job_Involvement,
        data.Distance_From_Home,
        data.Number_of_Companies_Worked,
        data.Attrition_Yes,
        data.Gender_Male,
        data.Marital_Status_Married,
        data.Marital_Status_Single,
        data.Department_HR,
        data.Department_IT,
        data.Department_Sales,
        data.Job_Role_Assistant,
        data.Job_Role_Executive,
        data.Job_Role_Manager,
        data.Overtime_Yes
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    return PredictionResponse(
        Attrition_Prediction="Yes" if prediction[0] == 1 else "No",
        Probability=probability[0]
    )

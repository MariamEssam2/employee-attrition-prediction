# employee-attrition-prediction
A machine learning project to predict employee attrition using data analysis and predictive modeling
# 🚀 Employee Attrition Prediction API

This is a **FastAPI-based REST API** that predicts employee attrition using a **Random Forest Classifier**.  
It allows organizations to proactively identify employees likely to leave by analyzing key HR metrics.

---

## 📊 Key Features Used for Prediction

| Feature                           | Description                                  |
|------------------------------------|----------------------------------------------|
| Age                                | Employee's age                               |
| Job Level                          | Level of the employee's job (1 to 5)         |
| Monthly Income                     | Monthly income of the employee               |
| Years at Company                   | Years the employee has been at the company   |
| Work-Life Balance                  | Employee's rating of work-life balance       |
| Job Satisfaction                   | Job satisfaction rating                      |
| Performance Rating                 | Employee's performance rating                |
| Training Hours Last Year           | Hours of training in the last year           |
| Project Count                      | Number of projects handled                   |
| Average Hours Worked Per Week      | Average weekly working hours                 |
| Absenteeism                        | Number of absent days in a year              |
| Distance From Home                 | Distance from home to office                 |
| Overtime                           | Whether the employee works overtime (Yes/No) |

> ℹ Other categorical features like department, job role, gender, and marital status are also encoded and used, but only main influencing variables are shown here.

---

## 🎯 Model Used

- **Random Forest Classifier**  
A powerful ensemble algorithm that builds multiple decision trees and outputs the class that is the mode of the classes (classification).

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/attrition-prediction-api.git
cd attrition-prediction-api
Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Ensure your model
Ensure the file best_model.pkl (trained Random Forest model) is in the project root.

4. Start the API
bash
Copy
Edit
uvicorn main:app --reload
5. API Docs
Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

📤 Example API Usage
Health Check
http
Copy
Edit
GET /
Response:

json
Copy
Edit
{
  "message": "🚀 Welcome! The Attrition Prediction API is running successfully."
}
Example Request:
json
Copy
Edit
{
  "Age": 30,
  "Job_Level": 2,
  "Monthly_Income": 5000,
  "Years_at_Company": 3,
  "Work_Life_Balance": 3,
  "Job_Satisfaction": 4,
  "Performance_Rating": 3,
  "Training_Hours_Last_Year": 20,
  "Project_Count": 4,
  "Average_Hours_Worked_Per_Week": 42,
  "Absenteeism": 3,
  "Distance_From_Home": 15,
  "Overtime_Yes": 1
}
Example Response:
json
Copy
Edit
{
  "Attrition_Prediction": "Yes",
  "Probability": 0.68
}
💡 Notes
Ensure the model file is trained with matching feature structure.

API supports easy testing via Swagger UI (/docs).

📃 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Would you also like me to provide a **clean `requirements.txt`** matching your project?  
If yes, just say "**Yes, requirements.**"

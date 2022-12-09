#### Linkedin User Diagnostic Application
#### Import packages
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
import streamlit as st
import plotly.graph_objects as go
from PIL import Image


#### Building the model for the app
s=pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    a=np.where(x==1,1,0)
    x=a.tolist()
    return (x)

sm_li=clean_sm(s["web1h"])
s.insert(1,"sm_li",sm_li)
Parent=clean_sm(s["par"])
s.insert(2,"Parent",Parent)
Married=clean_sm(s["marital"])
s.insert(3,"Married",Married)
Female=np.where(s["gender"] == 2, 1, 0)
s.insert(4,"Female",Female)
s=s[~(s["income"]>9)]
s=s[~(s["educ2"]>8)]
ss=s[['sm_li','income','educ2','Parent','Married','Female','age']].copy()
ss.columns=['sm_li','Income','Education','Parent','Married','Female','Age']

y = ss['sm_li']
X = ss[['Income','Education','Parent','Married','Female','Age']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=337)

lr = LogisticRegression(class_weight='balanced')

lr.fit(X_train, y_train)

image = Image.open('linkedin.png')

#### Add header to describe app
st.title("Are you a LinkedIn User?")
st.header("Let's Find Out!")
st.image(image)
st.markdown('''
Image source: [Market Path](https://www.marketpath.com/blog/a-linkedin-trick-to-update-a-links-image)
''')

st.subheader("This is a short quiz that will predict if you are a LinkedIn User.")



st.header("Let's get started by answering a few questions!")
IncomeI = st.selectbox(label="What is your household Income?",
options=("Less than $10K", "10K to under $20K", "20K to under $30K","30K to under $40K", "40K to under $50K", 
"50K to under $75K", "75K to under $100K", "100K to under $150K", "$150K or more?"))

EducationI = st.selectbox(label="What is your highest level of school/degree completed?",
options=("Less than high school (Grades 1-8 or no formal schooling)", 
"High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
 "High school graduate (Grade 12 with diploma or GED certificate)", 
 "Some college, no degree (includes some community college)", 
"Two-year associate degree from a college or university", 
"Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)", 
"Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
"Postgraduate or professional degree, including masters, doctorate, medical or law degree"))

ParentI= st.selectbox(label="Are you a parent of a child under 18 living in your home?",
options=("Yes", "No"))

MarriedI= st.selectbox(label="Are you married?",
options=("Yes", "No"))

FemaleI= st.selectbox(label="Do you idenitfy as a woman?",
options=("Yes", "No"))

Age=st.slider("How old are you?")


Income_Mapping={"Less than $10K":1
                , "10K to under $20K":2
                ,"20K to under $30K": 3
                ,"30K to under $40K": 4
                , "40K to under $50K": 5
                , "50K to under $75K": 6
                , "75K to under $100K": 7
                , "100K to under $150K": 8
                , "$150K or more?":9}

Education_Mapping={"Less than high school (Grades 1-8 or no formal schooling)":1, 
"High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":2,
 "High school graduate (Grade 12 with diploma or GED certificate)":3, 
 "Some college, no degree (includes some community college)":4, 
"Two-year associate degree from a college or university":5, 
"Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)":6, 
"Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":7,
"Postgraduate or professional degree, including masters, doctorate, medical or law degree":8}

Parent_Mapping={"Yes":1, "No":0}

Married_Mapping={"Yes":1, "No":0}

Female_Mapping={"Yes":1, "No":0}

Person2=pd.DataFrame({
    "Income":[IncomeI],
    "Education":[EducationI],
    "Parent": [ParentI],
    "Married":[MarriedI],
    "Female":[FemaleI],
    "Age":[Age]
})

Person2=Person2.assign(Income=Person2.Income.map(Income_Mapping))

Person2=Person2.assign(Education=Person2.Education.map(Education_Mapping))

Person2=Person2.assign(Parent=Person2.Parent.map(Parent_Mapping))

Person2=Person2.assign(Married=Person2.Married.map(Married_Mapping))

Person2=Person2.assign(Female=Person2.Female.map(Female_Mapping))

person=Person2.values.flatten().tolist()
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

if st.button("Show the results!"):
    if predicted_class[0]== 1:
        st.header("This model predicts that you are a LinkedIn user!")
    else:
        st.header("This model predicts that you are not a LinkedIn user!")

    st.header(f"The probability that you are LinkedIn user is {probs[0][1]}")


st.markdown('''
Data source for model: [Pew](https://www.pewresearch.org/)
''')

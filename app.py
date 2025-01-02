import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import streamlit as st



#importing model
model = tf.keras.models.load_model('model.h5')

#load Scalar encoding, One hot encoding
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('OneHot_Encoder_Geo.pkl','rb') as file:
    OneHot_Encoder_Geo = pickle.load(file)

with open('standed_scaler.pkl','rb') as file:
    standed_scaler = pickle.load(file)

#streamlite app
st.title('Customer chrin Predection')

#user inputs
geography= st.selectbox('Geography',OneHot_Encoder_Geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
credit_score=st.number_input('CreditScore')
estimated_Salary=st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('NumOfProducts',1,4)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])


#input data
input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_Salary]
})
#onehot encoding
geo=OneHot_Encoder_Geo.transform([[geography]]).toarray()
geo_encode_df=pd.DataFrame(geo,columns=OneHot_Encoder_Geo.get_feature_names_out())

#Concatination
input_data=pd.concat([input_data.reset_index(drop=True),geo_encode_df],axis=1)

#scalling data
scaled_data=standed_scaler.transform(input_data)

#model prediction
prediction=model.predict(scaled_data)

##Propability
prediction_prob=prediction[0][0]
st.write(prediction_prob)

#final result
if prediction_prob > 0.05:
    st.write('customer is likely to Churn')
else:
    st.write('customer is not likely to Churn')
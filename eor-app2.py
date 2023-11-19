import streamlit as st
import numpy as np
import pandas as pd
import Catboost
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
# Load the saved neural network model

pickled_model = joblib.load(open('catboost-model.pkl', 'rb'))
train_df = pd.read_excel('train.xlsx', sheet_name='Sheet1')


# Streamlit app
def main():
    st.title("EOR Type Prediction App")

    # Sidebar for user input
    st.sidebar.header("User Input")

    # User input for density
    density = st.sidebar.slider("Density (API)", min_value=5.0, max_value=55.0, value=35.0)

    # User input for viscosity
    viscosity = st.sidebar.slider("Viscosity (cP)", min_value=0.01, max_value=300.0, value=100.0)

    # User input for oil saturation
    oil_saturation = st.sidebar.slider("Oil Saturation", min_value=0.2, max_value=0.9, value=0.75)

    # User input for permeability
    permeability = st.sidebar.slider("Permeability (mD)", min_value=1, max_value=5000, value=2500)

    # User input for depth
    depth = st.sidebar.slider("Depth (m)", min_value=300, max_value=15000, value=5000)

    # User input for temperature
    temperature = st.sidebar.slider("Temperature (°C)", min_value=95, max_value=300, value=210)

    # Button to predict using the neural network
    if st.sidebar.button("Predict EOR Type"):
        # Create DataFrame from user inputs
        user_df = pd.DataFrame({'Density': [density], 'Viscosity': [viscosity], 'Oil Saturation': [oil_saturation],
                                'Permeability': [permeability], 'Depth': [depth], 'Temperature': [temperature]})

 

        # Apply the neural network model
        user_prediction = model.predict(user_df)
        #predicted_label = np.argmax(user_prediction.reshape(-1, user_prediction.shape[0]), axis=0)  # Assuming the output layer uses softmax activation

        # Decode predicted label
        # Display result in a table
        st.subheader("Predicted EOR Type")
        result_df = pd.DataFrame({'Density': [density], 'Viscosity': [viscosity], 'Oil Saturation': [oil_saturation],
                                  'Permeability': [permeability], 'Depth': [depth], 'Temperature': [temperature],
                                  'Predicted EOR Type': [user_prediction]})
        st.table(result_df)

        # Display result in a bar plot (using Plotly)
        st.subheader("Bar Plot of Predicted EOR Type")
        bar_plot = px.bar(x=result_df.columns, y=result_df.iloc[0].values, title="Predicted EOR Type Distribution")
        st.plotly_chart(bar_plot)

        # Create pair plot for user input and predicted result
        st.subheader("Pair Plot of User Input and Predicted EOR Type")

        combined_df = pd.concat([user_df, result_df], axis=1)
        pair_plot = px.scatter_matrix(train_df, dimensions=['Density', 'Viscosity', 'Oil Saturation', 'Permeability', 'Depth', 'Temperature'], color='Label')
    
        st.plotly_chart(pair_plot)





if __name__ == "__main__":
    main()

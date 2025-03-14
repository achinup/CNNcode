import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model

def model_prediction(test_image):
    model = load_model("model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",['Home','About','Desies Recognization'])

if(app_mode=='Home'):
    st.header("Plant")
    image_path="https://www.shutterstock.com/shutterstock/photos/2440792099/display_1500/stock-photo-plant-diseases-are-conditions-caused-by-pathogens-such-as-fungi-bacteria-viruses-pests-that-2440792099.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown(""" 
    WELCOME TO PLANT DESIES PREDECTUION
    ### HOW TO USE IT
    1.***Upload image:***          
    2.***Analysis:***OUr system will process image
    3.***Result:*** View result                     
""")
    
if(app_mode=='About'):
    st.header("About")
    st.markdown("""
    i am am from kundapura i am achinthya  i am studying in bmsce colloge of engeneering 
                currently  i am in 3rd yesr 

    """)
elif(app_mode=='Desies Recognization'):
    st.header("Desies Recognization")    
    test_image=st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    if(st.button("Predict")):
        
        st.write("Our Predection")  
        result_index=model_prediction(test_image) 
        class_name = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
        st.success("Model is predicted it 's a{}".format(class_name[result_index]))     
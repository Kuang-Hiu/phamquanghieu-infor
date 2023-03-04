import streamlit as st
import time
import pandas as pd
import numpy as np
from car_price_predict import car_predict
import base64
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_home(placeholder):
    with placeholder.container():
        st.title("My name is Pham Quang Hieu!")
        st.caption(" I'm Master of Data Science in Hanoi University of Science")
        st.text(" I have 2 years experience in Researching and implementing Computer Vision")
        st.text(" I have experience in Machine Learning too")
        st.text(" This is my details:")
        displayPDF("PhamQuangHieu-Info.pdf")
        with open("PhamQuangHieu-Info.pdf", "rb") as file:
            st.download_button(label = "Download CV as PDF",
                               data = file,
                               file_name = "CV_Pham_Quang_Hieu.pdf",
                               mime = 'x-pdf')

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 0
    placeholder = st.empty()
    st.sidebar.header("Some projects deta:")
    proj_1 = st.sidebar.button("Used Cars Price Prediction [ML]", use_container_width=True)

    proj_2 = st.sidebar.button("Covid-19 Visualization [DA]", use_container_width=True)
    proj_3 = st.sidebar.button("Extract informations [OCR]", use_container_width=True)
    proj_4 = st.sidebar.button("Famous Person Recoginze [Reg]", use_container_width=True)
    proj_5 = st.sidebar.button("Harmful Content Censorship [OD]", use_container_width=True)
    home = st.sidebar.button("Home page", use_container_width=True)
    if proj_1:
        st.session_state.page = 1
    if proj_2:
        st.session_state.page = 2
    if proj_3:
        st.session_state.page = 3
    if proj_4:
        st.session_state.page = 4
    if home:
        st.session_state.page = 0


    if st.session_state.page == 0:
        display_home(placeholder)
    elif st.session_state.page == 1:
        display_project1(placeholder)
    elif st.session_state.page == 2:
        display_project2(placeholder)
    elif st.session_state.page == 3:
        display_project3(placeholder)
    elif st.session_state.page == 4:
        display_project4(placeholder)

    st.sidebar.expander = False

def display_project2(placeholder):
    with placeholder.container():
        st.title("Introduce about project:")
        st.text("This is my major project in master")
        st.text("I visualized covid-19 in Vietnam from 01/2021 to 12/2021")
        st.text("The dataset was collected daily in public websites")
        displayPDF("covid_visualize/Bao cao Phan tich Covid.pdf")

def display_project3(placeholder):
    with placeholder.container():
        st.title("Introduce about project:")
        st.text("- Researching and implementing to extract infor for document management")
        st.text("- From the text in pdf format I use the OD model to crop the areas to extract infor")
        st.text("- I then use the finetune CRAFT model to extract the lines")
        st.text("- Information will be retrieved using NLP model such as VietOCR")
        st.caption('Results:')
        col1, col2 = st.columns(2)
        with col1:
            st.text("Input:")
            st.image('vbhc/input.png')
        with col2:
            st.text("Results:")
            #st.image('vbhc/result.png')
            st.text('- CQBH: NGÂN HÀNG NHÀ NƯỚC | VIỆT NAM')
            st.text('- SKH: Số: 8253/NHNN-CSTT')
            st.text('- Ngày BH: Hà Nội, ngày 22 tháng 11 năm 2022')
            st.text('- Loại VB: Công văn')
            st.text('- Trích yếu: V/v tăng trưởng tín dụng năm 2022')
            st.text('- Nơi nhận: Nơi nhận: | Như để gửi; | Thủ tướng Chính phủ (đế b/c); |')
            st.text('Phó Thủ tướng Lê Minh Khải (để b/c); | Ban lãnh đạo NHNN (để b/c); |')
            st.text('Văn phòng Chính phủ (để b/c); | Vụ TDCNKT, CQTTGSNH, | Vụ Truyền thông (để p/h);')
            st.text('- Lưu VP, Vụ CSTT. (LTH Yến)')
            st.text('- Người ký: Phạm Thanh Hà')

def display_project4(pl):
    with pl.container():

        st.title("Introduce about project:")
        st.text("- Researching and implementing the problem of person recognition, ")
        st.text("  applied to the lotus social network.")
        st.text("- I have focused on identifying photos of politicians and famous people in Vietnam.")
        st.text("- The dataset consists of about 500 classes.")
        st.text("- I have used facenet to extract features about human face images")
        st.text("- Then use facenet and arcface measures to classify objects")
        st.caption('Results:')

        st.image('face_r/results.png')



def display_project1(placeholder):
    with placeholder.container():
        st.title("Introduce about project:")
        st.text("Used cars have a very potential market")
        st.text("The main reason for this large market is that buying a new car ")
        st.text("and selling it in just 1 day can reduce the price of the car by up to 30%")
        st.text("The dataset is provided by Kaggle, with parameters collected in the Indian market")
        st.caption('Metrics:')
        st.text("- R2 score on Traing set: 98%")
        st.text("- R2 score on Testing set: 87%")
        st.text("- Mean squared error: 0.12")
        st.image("car_price_predict/289542.OH7AI634e545fd28eb.jpeg")

        st.title("Input car's specs  to valuation:")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            company = st.selectbox(
                'BRAND:',
                {'AMBASSADOR': 0,
                 'AUDI': 1,
                 'BENTLEY': 2,
                 'BMW': 3,
                 'CHEVROLET': 4,
                 'DATSUN': 5,
                 'FIAT': 6,
                 'FORCE': 7,
                 'FORD': 8,
                 'HONDA': 9,
                 'HYUNDAI': 10,
                 'ISUZU': 11,
                 'JAGUAR': 12,
                 'JEEP': 13,
                 'LAMBORGHINI': 14,
                 'LAND': 15,
                 'MAHINDRA': 16,
                 'MARUTI': 17,
                 'MERCEDES-BENZ': 18,
                 'MINI': 19,
                 'MITSUBISHI': 20,
                 'NISSAN': 21,
                 'PORSCHE': 22,
                 'RENAULT': 23,
                 'SKODA': 24,
                 'SMART': 25,
                 'TATA': 26,
                 'TOYOTA': 27,
                 'VOLKSWAGEN': 28,
                 'VOLVO': 29}
            )
            location = st.selectbox(
                "Location:",
                {'Ahmedabad': 0,
                 'Bangalore': 1,
                 'Chennai': 2,
                 'Coimbatore': 3,
                 'Delhi': 4,
                 'Hyderabad': 5,
                 'Jaipur': 6,
                 'Kochi': 7,
                 'Kolkata': 8,
                 'Mumbai': 9,
                 'Pune': 10}
            )
        with col2:
            year = st.selectbox("Date:",
                                [x for x in range(1998,2020)])
            transmission = st.selectbox("Transmission",
                                        ["Manual",
                                         "Automatic"])
        with col3:
            kilometers_driven = st.text_input("Kilometers Driven")
            owner_type = st.selectbox("Owner type",
                                        ['First', 'Second',"Third","Above Third"])
    
        with col4:
            fuel_type = st.selectbox("Fuel Type:",
                                     ['Diesel', 'Petrol', "Clean_Fuel"])
            engine = st.text_input("Engine(CC)")
        with col5:
            seats = st.selectbox("Seats:",
                                [x for x in range(1,11)])
            new_price = st.text_input("New Price (x1000$)")
        if kilometers_driven is not None:
            try:
                kilometers_driven = float(kilometers_driven)
            except:
                kilometers_driven = 1000
        if engine is not None:
            try:
                engine = float(engine)
            except:
                engine = 500
    
        if new_price is not None:
            try:
                new_price = float(new_price)
            except:
                new_price = None
    
        power = engine / 17.65
        mileage = engine / 37.6
    
        val = st.button("Valuation",use_container_width= True)
        if val:
            with st.spinner("In progress..."):
                time.sleep(1)
                data = {'Company' : [company],
                        'Location': [location],
                        'Year': [year],
                        'Kilometers_Driven' : [kilometers_driven],
                        'Fuel_Type': [fuel_type],
                        'Transmission' : [transmission],
                        'Owner_Type': [owner_type],
                        'Mileage': [mileage],
                        'Engine':[engine],
                        'Power': [power],
                        'Seats': [seats],
                        'New_Price':[new_price]
                }
                df = pd.DataFrame(data)

                X = car_predict.processing_data(df)

                res = car_predict.car_price(X)[0]
    
                st.title('Car price is about: {} $'.format(round(res*1000)))

if __name__ == "__main__":
    main()

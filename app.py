import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from PIL import Image
from fuzzywuzzy import fuzz

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizerUpdated.pkl','rb'))
model = pickle.load(open('modelUpdated.pkl','rb'))

st.header('Katch', divider='red')


selected3 = option_menu(None, ["Home", "Identity Detection", "Banking Usecase", "Text Scam" ,"Dating Usecase","Reporting", 'Team'], 
    icons=['house', 'cloud-upload', "list-task", "list-task","list-task",'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    })

def detect_scammer():
    
    st.title("Scammers Search Engine")
    my_list = st.text_input("Enter the details in comma separated format (FirstName, LastName, Age, Email, Country) and press enter")
    if(len(my_list)!=0):
        analysis(my_list)

def analysis(my_list):
    
    my_list = my_list.split(',')

    data = pd.read_csv('./Data/Categorised_data.csv')

    user_texts = {
    'user_first_name': my_list[0],
    'user_last_name': my_list[1],
    'user_age': my_list[2],
    'user_email': my_list[3],
    'user_country': my_list[4]
        }

    matching_scores = {user_text: [] for user_text in user_texts}

        # Columns in the CSV to compare (adjust these to match your columns)
    csv_columns = {
            'user_first_name': 'first_name',
            'user_last_name': 'last_name',
            'user_age': 'age',
            'user_email': 'email',
            'user_country': 'country'
        }

        # Initialize new columns for each user-provided text field to store the scores
    # Initialize new columns for each user-provided text field to store the scores
    for user_text in user_texts.keys():
        data[user_text + '_score'] = 0

    # Calculate the matching scores
    for user_text, csv_column in user_texts.items():
        for index, row in data.iterrows():
            # Get the user-provided text and the corresponding CSV column value
            user_value = user_texts[user_text]
            csv_value = str(row[csv_columns[user_text]])

            # Calculate the fuzzy string matching score between the user text and the CSV column value
            score = fuzz.token_set_ratio(user_value, csv_value)

        # Append the score to the corresponding user-provided text field
        matching_scores[user_text].append(score)
        data.at[index, user_text + '_score'] = score

        # Calculate the average score for each record and add it as a new column
    data['average_score'] = data[[user_text + '_score' for user_text in user_texts]].mean(axis=1)

        # Save the updated data with scores and the average score to a new CSV file
    
        # Fetch the entire record for the column with the highest average score
    highest_avg_score_index = data['average_score'].idxmax()
    highest_avg_score = data.iloc[highest_avg_score_index]['average_score']
    category = data.iloc[highest_avg_score_index]['Category']

    if st.button('Search'):
        #st.header("Matching Score", max(data['average_score']))
        if(highest_avg_score > 70):
            status = "Suspicious Profile - Strong Match"
            table_data = {
            'Matching Score':highest_avg_score
             , 'Status':status
             , 'Domain':category}
            st.table(table_data)
        if((highest_avg_score <= 70) and (highest_avg_score >= 60)):
            status = "Good Match"
            table_data = {
            'Matching Score':highest_avg_score
             , 'Status':status
             , 'Domain':category}
            st.table(table_data)
        if((highest_avg_score < 60) and (highest_avg_score >= 45)):
            status = "Possible Match"
            table_data = {
            'Matching Score':highest_avg_score
             , 'Status':status
             , 'Domain':category}
            st.table(table_data)
        if(highest_avg_score < 45):
            status = "No Match"
            table_data = {
            'Matching Score':highest_avg_score
             , 'Status':status}
            st.table(table_data)
 
def scam_detection():
    data = pd.DataFrame(columns=('Text', 'Classification', 'Attachment'))
    st.title("Upload the csv file")

    #st.header('Single File Upload')
    uploaded_file = st.file_uploader('Upload a file', type = ['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        for index, row in df.iterrows():
                # 1. preprocess
                transformed_sms = transform_text(row['text'])
                # 2. vectorize
                vector_input = tfidf.transform([transformed_sms])
                # 3. predict
                result = model.predict(vector_input)[0]
                print(result)
                # 4. Display
                if result == 0:
                    new_row = {'Text':row['text'], 'Classification':'Account Enquiry', 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)
                    #st.header("Spam")
                elif result == 1:
                    new_row = {'Text':row['text'], 'Classification':'Account Update', 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)
                    
                elif result == 2:
                    new_row = {'Text':row['text'], 'Classification':'Negative Customer Experience', 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)

                elif result == 3:
                    new_row = {'Text':row['text'], 'Classification':'Positive Customer Review'}#, 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)

                elif result == 4:
                    new_row = {'Text':row['text'], 'Classification':'Spam'}#, 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)

                else:
                    new_row = {'Text':row['text'], 'Classification':'Usage Issue'}#, 'Attachment':row['Attachment']}
                    data = data.append(new_row, ignore_index=True)
                    #st.header("Not Spam")
    # def initialize_session_state():
    #     if "write" not in st.session_state:  
    #         st.session_state.write = None
    # initialize_session_state()

    st.write(data)

    spam_data = data['Classification']=="Spam"

    count_data = data.groupby("Classification").size().reset_index(name="Count")

    st.title("Category Counts Bar Chart")
    # Create a bar chart using Plotly Express
    st.plotly_chart(px.bar(count_data, x="Classification", y="Count", title="Category Count"))


def identity_scam():
    tfidf = pickle.load(open('vectorizerIdentity.pkl','rb'))
    model = pickle.load(open('modelIdentity.pkl','rb'))
    st.title("Enter Identity Information")

    Email_Address = st.text_input("Enter the email address")

    Phone_Number = st.text_input("Enter phone number")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):

        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Fraud Email Address")
            st.header("Spam Detected !!!")
        else:
            st.header("Not Spam")


def dating_scam():
    st.title("Dating Profile Identity Scan")

    firstName = st.text_input("Enter the First Name")

    lastName  = st.text_input("Enter the Last Name")

    age  = st.text_input("Enter the Age")

    email  = st.text_input("Enter the Email address")

    country  = st.text_input("Enter the Country Name")

    # Load the CSV data
    data = pd.read_csv('./Data/Dating_data.csv')

    # User-provided text fields
    user_texts = {
        'user_first_name': firstName,
        'user_last_name': lastName,
        'user_age': age,
        'user_email': email,
        'user_country': country
    }

    # Initialize a dictionary to store the scores for each user-provided text field
    matching_scores = {user_text: [] for user_text in user_texts}

    # Columns in the CSV to compare (adjust these to match your columns)
    csv_columns = {
        'user_first_name': 'first_name',
        'user_last_name': 'last_name',
        'user_age': 'age',
        'user_email': 'email',
        'user_country': 'country'
    }

    # Initialize new columns for each user-provided text field to store the scores
    for user_text in user_texts.keys():
        data[user_text + '_score'] = 0

    # Calculate the matching scores
    for user_text, csv_column in user_texts.items():
        for index, row in data.iterrows():
            # Get the user-provided text and the corresponding CSV column value
            user_value = user_texts[user_text]
            csv_value = str(row[csv_columns[user_text]])

            # Calculate the fuzzy string matching score between the user text and the CSV column value
            score = fuzz.token_set_ratio(user_value, csv_value)

            # Append the score to the corresponding user-provided text field
            matching_scores[user_text].append(score)
            data.at[index, user_text + '_score'] = score

    # Calculate the average score for each record and add it as a new column
    data['average_score'] = data[[user_text + '_score' for user_text in user_texts]].mean(axis=1)

    def circular_tile(value):
    # Apply CSS styling to create a circular tile
        style = """
        border-radius: 50%;
        background-color: #3498db;
        color: #fff;
        text-align: center;
        width: 150px;
        height: 150px;
        line-height: 150px;
        font-size: 24px;
        display: flex;
        justify-content: center;
    """
    
    # Display the circular tile
        st.write(
            f'<div style="{style}">{value}</div>',
            unsafe_allow_html=True
        )

    if st.button('Get Matching Score'):
        #st.header("Matching Score", max(data['average_score']))
        circular_tile(max(data['average_score']))
        if(max(data['average_score']) > 70):
            st.header("Suspicious Profile - Strong Match")
        if(max(data['average_score']) <= 70 and (max(data['average_score']) >= 60)):
            st.header("Good Match")
        if(max(data['average_score']) < 60 and (max(data['average_score']) >= 45)):
            st.header("Possible Match")
        if(max(data['average_score']) < 45):
            st.header("No Match")


if selected3 == "Home":
    st.image("./Categories/Katch.jpg")
    # Define your content for each tile
    
    #st.markdown("<center><b><span style='font-size: 30px;'>Katch is the platform designed for the identification and prevention of fraudulent activities.<center><b><span style='font-size: 24px;'>")

    #st.markdown("<center><span style='font-size: 20px;'>Katch is the platform designed for the identification and prevention of fraudulent activities.<center><span style='font-size: 20px;'>", unsafe_allow_html=True)


    st.markdown("<center><b><span style='font-size: 30px;'>Targeted Industries<center><b><span style='font-size: 24px;'>", unsafe_allow_html=True)
   
    col1, col2, col3 = st.columns(3)

    col4, col5, col6 = st.columns(3)

    with col1:
        st.image("./Categories/OnlineDating.jpg")

    with col2:
        st.image("./Categories/Investment.jpg")

    with col3:
        st.image("./Categories/Banking.jpg")

    with col4:
        st.image("./Categories/Restaurant.jpg")

    with col5:
        st.image("./Categories/Healthcare.jpg")

    with col6:
        st.image("./Categories/Gaming.jpg")

if selected3 == "Banking Usecase":
    scam_detection()

if selected3 == "Identity Detection":
    detect_scammer()

if selected3 == "Text Scam":
    identity_scam()

if selected3 == "Dating Usecase":
    dating_scam()    
   
if selected3 == "Reporting":
    scam_authorities = pd.read_csv("./Data/Scam_reporting_authorities_2020-22.csv")
    scam_age = pd.read_csv("./Data/Scam_exposure_by_age_2021-22.csv")
    scam_type = pd.read_csv("./Data/Exposure_Rate_by_scam_type_2020-22.csv")
    scam_mode = pd.read_csv("./Data/Scam_exposure_by_mode_2020-2022.csv")

    # st.bar_chart(
    # scam_type, x="Scam Type", y=['2020-21 (%)', '2021-22 (%)'], color=["#FF0000", "#0000FF"]  # Optional
    # )

    st.image("./Data/acc.jpg")

    st.plotly_chart(px.bar(scam_type, x="Scam Type", y=['2020-21 (%)', '2021-22 (%)'], title="Types of scams"))

    st.plotly_chart(px.bar(scam_mode, x="Mode", y=['2020-21 (%)', '2021-22 (%)'], title="Modes of scam",barmode = 'group'))

    st.plotly_chart(px.pie(scam_age, values='Percentage', names='Age Group', title='Age Groups attacked by scammers'))

    st.plotly_chart(px.bar(scam_authorities, x='Reporting Authorities', y=['2020-21 (%)', '2021-22 (%)'], title="Scam notified to authorities",barmode = 'group'))

    # Display the chart in the Streamlit app
  #  st.plotly_chart(fig)
   # st.plotly_chart(fig1)

if selected3 == "Team":

    #st.title("Team CyberSentinels")

    st.markdown("<center><b><span style='font-size: 30px;'>Team CyberSentinels<center><b><span style='font-size: 24px;'>", unsafe_allow_html=True)
    st.write("")
    col7, col8, col9 = st.columns(3)

    col10, col11, col12 = st.columns(3)


    with col7:
        image = Image.open('./Team/Tehani.jpeg')
        st.image(image, caption='Tehani Legeay')

    with col8:
        image = Image.open('./Team/Vaib.jpeg')
        st.image(image, caption='Vaibhav Rastogi')

    with col9:
        image = Image.open('./Team/Sid.jpg')
        st.image(image, caption='Siddharth Varma')

    with col10:
        image = Image.open('./Team/yash.jpeg')
        st.image(image, caption='Yash Krishan Verma')

    with col11:
        image = Image.open('./Team/Joe.jpeg')
        st.image(image, caption='Joe Chang')

    with col12:
        image = Image.open('./Team/ayaz.jpeg')
        st.image(image, caption='Ayaz Mujawar')


    

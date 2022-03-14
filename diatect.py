import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# page configuration
st.set_page_config(
    page_title="DIATECT",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://github.com/tomisile/Diabetes-Detection/issues",
        'About': "# Diabetes Detection using *Neural Networks*."
    }
)

# App title
# st.title("DIATECT")

data_url = 'https://raw.githubusercontent.com/tomisile/Diabetes-Detection/main/diabetes_dataset.csv'


@st.cache
def load_data(url):
    data = pd.read_csv(url)
    return data


# load dataset
dataset = load_data(data_url)


# load trained model
pickle_in = open(r"C:\Users\Tomi\Documents\github\Diabetes Prediction\mlpc.pkl", 'rb')
mlpc = pickle.load(pickle_in)


def prediction(age, gender, polyuria, polydipsia, sudden_weight_loss,
               weakness, polyphagia, genital_thrush, visual_blurring, itching,
               irritability, delayed_healing, partial_paresis, muscle_stiffness,
               alopecia, obesity):
    prediction = mlpc.predict([[age, gender, polyuria, polydipsia, sudden_weight_loss,
                                weakness, polyphagia, genital_thrush, visual_blurring, itching,
                                irritability, delayed_healing, partial_paresis, muscle_stiffness,
                                alopecia, obesity]])
    # print(prediction)
    return prediction


def main():
    """ this function defines the webpage """

    # side-bar
    st.sidebar.markdown('**Diabetes** is a disease that occurs when your blood glucose, '
                        'also called blood sugar, is **_too high._**')
    st.sidebar.write('Did you know?')

    # Main page
    image1 = Image.open(r"C:\Users\Tomi\Documents\github\Diabetes Prediction\image1.jpg")
    st.image(image1, caption=None)

    # view dataset checkbox
    if st.checkbox('View raw data'):
        # Inspect the raw data
        st.subheader('Dataset')
        st.write(dataset)
        st.write(dataset.shape)

        # dataset download button
        @st.cache
        def convert_df(df):
            # convert the pandas dataframe dataset back to csv
            return df.to_csv().encode('utf-8')

        csv = convert_df(dataset)
        if st.download_button(
                label='Download data as CSV file',
                data=csv,
                file_name='diabetes_dataset.csv',
                mime='text/csv'
        ):
            st.success('Your download has started')

    st.subheader('Take a test')
    with st.expander('Answer these questions to check your diabetes status', expanded=True):

        # text boxes and check boxes for user to input data for prediction
        gender_options = {1: 'Male', 0: 'Female'}
        yes_no_options = {1: 'Yes', 0: 'No'}

        age = st.number_input('1. How old are you?', min_value=1, max_value=100, value=60)
        gender = st.radio('2. Gender', options=(1, 0), format_func=lambda x: gender_options.get(x))
        if gender == 1:
            st.write('You\'re male')
        else:
            st.write('You\'re female')
        polyuria = st.radio('3. Have you been urinating excessively?', options=(1, 0),
                            format_func=lambda x: yes_no_options.get(x))
        polydipsia = st.radio('4. Do you feel extreme thirstiness?', options=(1, 0),
                              format_func=lambda x: yes_no_options.get(x))
        sudden_weight_loss = st.radio('5. Have you had a sudden weight loss recently?', options=(1, 0),
                                      format_func=lambda x: yes_no_options.get(x))
        weakness = st.radio('6. Do you have feel extreme weakness?', options=(1, 0),
                            format_func=lambda x: yes_no_options.get(x))
        polyphagia = st.radio('7. Are you excessively hungry?', options=(1, 0),
                              format_func=lambda x: yes_no_options.get(x))
        genital_thrush = st.radio('8. Do you suffer from genital thrush?', options=(1, 0),
                                  format_func=lambda x: yes_no_options.get(x))
        visual_blurring = st.radio('9. Do you have blurred vision?', options=(1, 0),
                                   format_func=lambda x: yes_no_options.get(x))
        itching = st.radio('10. Any severe itching in the last few days?', options=(1, 0),
                           format_func=lambda x: yes_no_options.get(x))
        irritability = st.radio('11. Are you excessively irritated?', options=(1, 0),
                                format_func=lambda x: yes_no_options.get(x))
        delayed_healing = st.radio('12. Do your wounds take too long to heal?', options=(1, 0),
                                   format_func=lambda x: yes_no_options.get(x))
        partial_paresis = st.radio('13. Are your muscles weaker than normal?', options=(1, 0),
                                   format_func=lambda x: yes_no_options.get(x))
        muscle_stiffness = st.radio('14. Do you suffer from muscle stiffness?', options=(1, 0),
                                    format_func=lambda x: yes_no_options.get(x))
        alopecia = st.radio('15. Are you currently experiencing hair loss?', options=(1, 0),
                            format_func=lambda x: yes_no_options.get(x))
        obesity = st.radio('16. Are you obese?', options=(1, 0),
                           format_func=lambda x: yes_no_options.get(x))
        result = ""

        # on_click of predict button
        if st.button('PREDICT'):
            result = prediction(age, gender, polyuria, polydipsia, sudden_weight_loss,
                                weakness, polyphagia, genital_thrush, visual_blurring, itching,
                                irritability, delayed_healing, partial_paresis, muscle_stiffness,
                                alopecia, obesity)
        st.success('Your diabetes status is {}'.format(result))


# run webpage
if __name__ == '__main__':
    main()

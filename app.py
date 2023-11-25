import streamlit as st
import pickle

st.set_page_config(page_title="Text Classification" , page_icon=':flag-pa:',layout='wide')
st.title('Text Classification')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# ##################################################################################################

vect = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def prediction(text):
    text_v = vect.transform([text])
    result = model.predict(text_v)

    if result == 1:
        return 'Passive'
    else:
        return 'Active'




st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.markdown('#### Text Classification for Active vs. Passive Voice Detection')


st.write('')
st.write('')
txt = st.text_area('Write Your Sentence')


st.write('')
st.write('')
st.write('')
btn = st.button('Predict')

st.write('')
st.write('')
st.write('')

if btn:
    p_result = prediction(txt)
    st.markdown(f'# {p_result}')



st.write('')
st.write('')
st.write('')
st.write('')
st.markdown('<span>All code can be found here <a href="https://github.com/asbpintu/Text-Classification.git">Text Classification</a></span>',unsafe_allow_html=True)
st.markdown('<span>Created By <a href="https://www.linkedin.com/in/asbpintu/">asbpintu</a></span>',unsafe_allow_html=True)
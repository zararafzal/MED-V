import streamlit as st
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

sample_prompt = """You are a medical practitioner with expertise in various medical fields. Your task is to analyze provided medical data, identify any health issues, and offer a detailed response including findings, next steps, and recommendations.

Please ensure that you:

1. **Identify Anomalies and Diseases**: Describe any abnormal findings.
2. **Provide Detailed Findings**: Offer a comprehensive analysis of the patient's condition.
3. **Suggest Next Steps**: Recommend further tests, treatments, or follow-up actions.
4. **Give Recommendations**: Advise on treatment options, lifestyle changes, or preventive measures.
5. If unclear, state: "Unable to determine based on the provided information."

Note: Respond only to medical-related data. Always include this disclaimer:

**Disclaimer**: Consult with a doctor before making any decisions based on this analysis.

Now, please analyze the provided medical data and give your comprehensive report as outlined. Use bullets where necessary and provide a structured answer close to nomad terminology."""

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_model_for_analysis(filename: str, sample_prompt=sample_prompt):
    base64_image = encode_image(filename)
    
    messages = [
        {
            "role": "user",
            "content":[
                {
                    "type": "text", "text": sample_prompt
                    },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                        }
                    }
                ]
            }
        ]

    response = client.chat.completions.create(
        model = "gpt-4-vision-preview",
        messages = messages,
        max_tokens = 1500
        )

    print(response.choices[0].message.content)
    return response.choices[0].message.content

def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
    messages = [
        {
            "role": "user",
            "content": eli5_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content

st.title("Medical Help using Multimodal LLM")

with st.expander("About this App"):
    st.write("Upload an image to get an analysis from GPT-4.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Temporary file handling
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name

    st.image(uploaded_file, caption='Uploaded Image')

# Process button
if st.button('Analyze Image'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Delete the temp file after processing

# ELI5 Explanation
if 'result' in st.session_state and st.session_state['result']:
    st.info("Below you have an option for ELI5 to understand in simpler terms.")
    if st.radio("ELI5 - Explain Like I'm 5", ('No', 'Yes')) == 'Yes':
        simplified_explanation = chat_eli(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)
             

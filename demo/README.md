# Chatbot demo

This is a real time chatbot demo which talks to the deployed model endpoint over the REST API. 

## Install Python requirements

    pip install -r requirements.txt

## Deploy models

Download and deploy the following models as per instructions provided in the [docs](https://opendocs.nutanix.com/gpt-in-a-box/overview/). 

    lama2-7b-chat
    
    codellama-7b-python

## Run Chatbot app

Once the inference server is up, run

    streamlit run chat.py

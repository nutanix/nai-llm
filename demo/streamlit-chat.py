import os
import requests
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Add supported models to the list 
AVAILABLE_MODELS = ["llama2-7b-chat", "codellama-7b-python"]
#AVAILABLE_MODELS = ["llama2-7b", "mpt-7b" , "falcon-7b"]
ASSISTANT_SVG = "assistant.svg"
USER_SVG = "user.svg"
LOGO_SVG = "nutanix.svg"

llm_mode = "chat"
llm_history = "off"

if not os.path.exists(ASSISTANT_SVG):
    assistant_avatar = None
else:
    assistant_avatar = ASSISTANT_SVG

if not os.path.exists(USER_SVG):
    user_avatar = None
else:
    user_avatar = USER_SVG

# App title
st.title("Hola Nutanix")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:

    if os.path.exists(LOGO_SVG):
        _, col2, _,_ = st.columns(4)
        with col2:
            st.image(LOGO_SVG, width=150)
    
    st.title("GPT-in-a-Box")
    st.markdown("GPT-in-a-Box is a turnkey AI solution for organizations wanting to implement GPT capabilities while maintaining control of their data and applications. Read the [annoucement](https://www.nutanix.com/blog/nutanix-simplifies-your-ai-innovation-learning-curve)")

    st.subheader("Models")
    selected_model = st.sidebar.selectbox("Choose a model", AVAILABLE_MODELS, key="selected_model")
    if selected_model == "llama2-7b":
        llm = "llama2_7b"
        st.markdown("Llama2 is a state-of-the-art foundational large language model which was pretrained on publicly available online data sources. This chat model leverages publicly available instruction datasets and over 1 million human annotations.")
    elif selected_model == "mpt-7b":
        llm = "mpt_7b"
        st.markdown("MPT-7B is a decoder-style transformer with 6.7B parameters. It was trained on 1T tokens of text and code that was curated by MosaicML’s data team. This base model includes FlashAttention for fast training and inference and ALiBi for finetuning and extrapolation to long context lengths.")
    elif selected_model == "falcon-7b":
        llm = "falcon_7b"
        st.markdown("Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora.")
    elif selected_model == "codellama-7b-python":
        llm = "codellama_7b_python"
        llm_mode = "code"
        st.markdown("Code Llama is a large language model that can use text prompts to generate and discuss code. It has the potential to make workflows faster and more efficient for developers and lower the barrier to entry for people who are learning to code.")
    elif selected_model == "llama2-7b-chat":
        llm = "llama2_7b_chat"
        llm_history = "on"
        st.markdown("Llama2 is a state-of-the-art foundational large language model which was pretrained on publicly available online data sources. This chat model leverages publicly available instruction datasets and over 1 million human annotations.")
    else:
        quit()

    if "model" in st.session_state and st.session_state["model"] != llm:
        clear_chat_history()

    st.session_state["model"] = llm

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def add_message(message):
    if message["role"] == "assistant":
        avatar = assistant_avatar
    else:
        avatar = user_avatar
    if llm_mode == "code":
        with st.chat_message(message["role"], avatar=avatar):
            st.code(message["content"], language="python")
    else: 
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

# Display or clear chat messages
for message in st.session_state.messages:
    add_message(message)

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

def generate_response(prompt):
    url = f"http://localhost:8080/predictions/{llm}"
    headers = {"Content-Type": "application/text; charset=utf-8"}
    try:
        response = requests.post(url, data=prompt, timeout=120, headers=headers)
    except requests.exceptions.RequestException:
        print("Error in requests: ", url)
        return ""
    return response.content.decode("utf-8")

def generate_chat_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'." + "\n\n"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    input=f"{string_dialogue} {prompt_input}" + "\n\n"+ "Assistant: "
    output = generate_response(input)
    # Generation failed
    if len(output) <= len(input):
        return ""
    return output[len(input):]


# User-provided prompt
if prompt := st.chat_input("Ask your query"):
    message = {"role": "user", "content": prompt}
    st.session_state.messages.append(message)
    add_message(message)


# Generate a new response if last message is not from assistant
def add_assistant_response():
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Thinking..."):
                print(llm_history, llm_mode)
                if llm_history == "on":
                    response = generate_chat_response(prompt)
                else:
                    response = generate_response(prompt)
                if not response:
                    st.markdown("<p style='color:red'>Inference backend is unavailable. Please verify if the inference server is running</p>", unsafe_allow_html=True)
                    return
                if llm_mode == "code":
                    st.code(response, language="python")
                else:
                    st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

add_assistant_response()


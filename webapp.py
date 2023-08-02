import streamlit as st
import time
import torch
from model import PoemGPT
from make_pdf import make_pdf

import gtts
from io import BytesIO

from typing import Tuple, Dict
import json

if "poem" not in st.session_state:
    st.session_state.poem = ""
    st.session_state.continue_gen = False
    st.session_state.disable_context = False

def clear_states():
    st.session_state.poem = ""
    st.session_state.continue_gen = False
    # exit()


device = "cuda" if torch.cuda.is_available() else "cpu"
with open("./config/poem_gpt_config.json", "r") as f:
        config = json.load(f)

@st.cache_resource()
def load_model(poet):
    with open(f"./tokenize/{poet}_encode.json", "r") as f:
        stoi = json.load(f)
    with open(f"./tokenize/{poet}_decode.json", "r") as f:
        itos = json.load(f)
        itos = {int(k):itos[k] for k in itos}

    model = PoemGPT(config=config, vocab_size=len(stoi))
    model.load_state_dict(torch.load(f"./weights/{poet}.pt", map_location=torch.device(device=device)))
    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda t: "".join([itos[i] for i in t])
    return model, encode, decode


def type_output(content: str, delay = 0.03):
    placeholder = st.empty()
    content = content.replace("\n", "  \n")
    for letter in content:
        st.session_state["poem"] += letter
        placeholder.markdown(st.session_state["poem"])
        time.sleep(delay)

def convert_text_to_audio(text: str) -> BytesIO:
    sound_file = BytesIO()
    tts = gtts.gTTS(text, lang='en')
    tts.write_to_fp(sound_file)
    return sound_file

def main():
    st.set_page_config(
        page_title="Poetika",
        page_icon="✒️"
    )

    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer{visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.6, 0.4], gap="large")
    with col1:
        st.header("Poetika")
        poet = st.selectbox(label="**Select your poet** ✒️", 
                            options=["Shakespeare", "Wordsworth"],
                            on_change=clear_states)
        
        st.session_state.disabled = False
        input_txt = st.text_input(label="Context for the poem",
                                value="", max_chars=200,
                                key="context",
                                placeholder="You can type the start of the poem",
                                disabled=st.session_state.disable_context)
        char_len = st.slider(label="Length of the poem", min_value=200, max_value=700, value=350,
                            step=5, disabled=st.session_state.disable_context)
        placeholder = st.empty()
        generate = None
        if not st.session_state.continue_gen:
            generate = placeholder.button("Generate")
        else:
            generate = placeholder.button(label="Continue generation?",
                                          key="cont_key_1")
    
    with col2:
        st.header("The Jam Zone")
        if generate or st.session_state.continue_gen:
            model, encode, decode = load_model(poet=poet.lower())
            
            if not st.session_state.continue_gen:
                if len(input_txt) > 0:
                    context = torch.tensor([encode(input_txt)],
                                        dtype=torch.long,
                                        device=device)
                else:
                    context = torch.zeros(size=(1,1),
                                        dtype=torch.long,
                                        device=device)
            else:
                last_gen_poem = st.session_state.poem[-char_len:]
                print(last_gen_poem)
                context = encode(last_gen_poem)
                context = torch.tensor([context],
                                       dtype=torch.long,
                                       device=device)
            

            st.toast(f"{poet} is thinking...", icon="💭")
            out = model.generate(x=context, max_new_tokens=char_len) # [1, S]
            st.toast(f"{poet} is writing...", icon="✒️")
            text = ""
            if not st.session_state.continue_gen:
                text = decode(out[0].cpu().numpy())
            else:
                text = decode(out[0].cpu().numpy()[char_len:])
            type_output(text)

            

            pdf = make_pdf(poet=poet, text=st.session_state["poem"])
            btn = st.download_button(
                label="Download as PDF",
                data=bytes(pdf.output()),
                file_name="gen.pdf",
                mime="application/pdf",
                on_click=clear_states
            )
            
            st.toast(f"Generating audio for the poem...", icon="🎤")
            st.audio(convert_text_to_audio(text=st.session_state["poem"]))
            
            if not st.session_state.continue_gen:
                st.session_state.continue_gen = True
                st.session_state.disable_context = True
            generate = placeholder.button("Continue generation?",
                                          key="cont_key_2")
    

if __name__=="__main__":
    main()
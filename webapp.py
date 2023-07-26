import streamlit as st
import time
import torch
from model import PoemGPT
from make_pdf import make_pdf

import gtts
from io import BytesIO

from typing import Tuple, Dict
import json

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
    intermediate = ""
    for letter in content:
        intermediate += letter
        placeholder.markdown(intermediate)
        time.sleep(delay)

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

    st.header("Poetika")
    st.markdown("A poem by [Poet K](https://www.linkedin.com/in/sathya-krishnan-suresh-914763217/)")
    options = ["Shakespeare", "Wordsworth"]

    poet = st.radio(label="**Select your poet** ✒️",
                horizontal=True,
                options=options)
    
    st.session_state.disabled = False
    input_txt = st.text_input(label="Context for the poem",
                            value="", max_chars=200,
                            key="context",
                            placeholder="You can type the start of the poem",
                            disabled=st.session_state.disabled)
    
    generate = st.button(label="Generate")
    if generate:
        model, encode, decode = load_model(poet=poet.lower())
        if len(input_txt) > 0:
            context = torch.tensor([encode(input_txt)],
                                   dtype=torch.long,
                                   device=device)
        else:
            context = torch.zeros(size=(1,1),
                                  dtype=torch.long,
                                  device=device)
        
        st.session_state.disabled=True
        out = model.generate(x=context) # [1, S]
        text = decode(out[0].cpu().numpy())
        type_output(text)
    
        pdf = make_pdf(poet=poet, text=text)
        btn = st.download_button(
            label="Download as PDF",
            data=bytes(pdf.output()),
            file_name="gen.pdf",
            mime="application/pdf"
        )

        
        sound_file = BytesIO()
        tts = gtts.gTTS(text, lang='en')
        tts.write_to_fp(sound_file)
        st.audio(sound_file)
        
    

if __name__=="__main__":
    main()
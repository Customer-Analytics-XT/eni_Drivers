import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

st.set_page_config(page_title="Driver", page_icon="🚙")
st.title("🚙 Eni Driver")

os.environ["HF_TOKEN"] = st.secrets.huggingface.token

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login()

if st.session_state["authentication_status"]:

    # -------------- scelta modello -----------------------------------------
    options = ["", "Driver 4.0", "Driver 4.3"]           # "" = placeholder
    model_choice = st.selectbox(
        "Seleziona la versione del driver",
        options,
        index=0,
        key="model_choice",
        format_func=lambda x: "— seleziona —" if x == "" else x,
    )

    # se l’utente non ha ancora selezionato nulla, fermiamo l’esecuzione
    if model_choice == "":
        st.info("Seleziona una versione per caricare il modello e avviare la classificazione.")
        st.stop()

    # -------------- mapping & caricamento modello --------------------------
    model_mapping = {
        "Driver 4.0": "marcopoggiey/bert-driver4",
        "Driver 4.3": "marcopoggiey/eurobert-driver4-3",
    }
    model_path = model_mapping[model_choice]

    # (ri)carica modello/tokenizer solo se è cambiato
    if st.session_state.get("current_model_path") != model_path:
        st.cache_data.clear() 
        with st.spinner("Caricamento del modello…"):
            st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(
                model_path, token=st.secrets.huggingface.token, trust_remote_code=True
            )
            st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_path)
            st.session_state["current_model_path"] = model_path

    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -------------- labels dinamiche ---------------------------------------
    labels = (
        [
            "Products_B2C",
            "Products_B2B",
            "Innovation",
            "Workplace",
            "Governance",
            "Citizenship",
            "Leadership",
            "Performance",
        ]
        if model_choice != "Driver 4.0"
        else [
            "Products",
            "Innovation",
            "Workplace",
            "Governance",
            "Citizenship",
            "Leadership",
            "Performance",
        ]
    )

    # -------------- max_length dinamico ------------------------------------
    max_length = 512 if model_choice == "Driver 4.0" else 2048

    # -------------- funzione di inferenza ----------------------------------
    @st.cache_data(show_spinner=False)
    def predict_labels(text, threshold=0.5):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        preds = (probs >= threshold).astype(int)
        return probs, preds

    # -------------- UI principale ------------------------------------------
    text = st.text_area("Inserisci il testo da classificare")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Per favore, inserisci un testo valido.")
        else:
            probs, preds = predict_labels(text)

            st.markdown("### Driver selezionati")
            scelti = [l for l, p in zip(labels, preds) if p == 1]
            st.info(", ".join(scelti) if scelti else "Nessun driver selezionato.")

            # barplot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, probs, color="skyblue")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilità")
            ax.set_title("Probabilità dei driver")
            for i, v in enumerate(probs):
                xpos = v - 0.03 if v > 0.9 else v + 0.01
                ax.text(xpos, i, f"{v:.2f}", va="center")
            st.pyplot(fig)

    authenticator.logout()

elif st.session_state["authentication_status"] is False:
    st.error("Username/password errati")
elif st.session_state["authentication_status"] is None:
    st.warning("Inserisci username e password")

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

st.set_page_config(page_title="Driver", page_icon="🚙")
st.title("🐕🚙Eni Driver 4.0")

os.environ["HF_TOKEN"] = st.secrets.huggingface.token

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

if st.session_state['authentication_status']:

    st.markdown(f"### 👋Ciao, {st.session_state['name'].split(' ')[0]}!")
    authenticator.logout()

    # Carica il modello e il tokenizer
    model_path = "marcopoggiey/bert-driver4"
    if "model" not in st.session_state:
        with st.spinner("Caricamento del modello..."):
            st.session_state["model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = AutoTokenizer.from_pretrained(model_path)

    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels = ["Products", "Innovation", "Workplace", "Governance", "Citizenship", "Leadership", "Performance"]

    def predict_labels(text, threshold=0.5):
        # Preprocessa il testo
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Calcola le probabilità senza aggiornare i gradienti
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs >= threshold).astype(int)
        return probs, preds

    text = st.text_area("Inserisci il testo da classificare")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Per favore, inserisci un testo valido.")
        else:
            probs, preds = predict_labels(text)

            # Visualizza in evidenza i driver selezionati
            st.markdown("### Driver Selezionati")
            selected = False
            chosen_labels = []
            for label, prob, pred in zip(labels, probs, preds):
                if pred == 1:
                    chosen_labels.append(label)
                    selected = True
            if selected:
                st.info(", ".join(chosen_labels))
            if not selected:
                st.info("Nessun driver selezionato.")

            # Mostra un barplot orizzontale con le probabilità su scala fissa [0,1]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, probs, color="skyblue")
            ax.set_xlim(0, 1)  # Imposta l'asse x da 0 a 1
            ax.set_xlabel("Probabilità")
            ax.set_title("Probabilità dei driver")
            # Aggiungi etichette con i valori
            for i, v in enumerate(probs):
                if v > 0.9:
                    # Posiziona l'etichetta sopra la barra
                    ax.text(v - 0.03, i - 0.2, f"{v:.2f}", color="blue", ha="center", va="bottom")
                else:
                    ax.text(v + 0.01, i - 0.05, f"{v:.2f}", color="blue", va="center")
            st.pyplot(fig)

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
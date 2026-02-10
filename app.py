import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
import soundfile as sf
import whisper
import os

# Set Page Config
st.set_page_config(
    page_title="Data Selection Demo",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Data Selection for Efficient Speech Processing")
st.markdown("**Based on the survey by Azeemi et al. (2025)**")

# Sidebar for Navigation
task = st.sidebar.radio("Select Speech Task:", ["1. ASR (Diversity)", "2. TTS (Purity)", "3. Anti-Spoofing (Robustness)"])

# ==========================================
# TASK 1: ASR DEMO
# ==========================================
if task == "1. ASR (Diversity)":
    st.header("1. Automatic Speech Recognition (ASR)")
    st.subheader("The Challenge: Redundancy vs. Diversity")
    st.markdown("""
    Most training data is "easy" (clean speech). To improve efficiency, we need to find the "hard" examples (high uncertainty).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("🟢 **Random Selection** (Clean Data)")
        if st.button("Generate Clean Sample"):
            text = "The quick brown fox jumps over the lazy dog."
            tts = gTTS(text, lang='en')
            tts.save("clean.mp3")
            st.audio("clean.mp3")
            
            # Transcription (Mocked or Real)
            # We mock the 'Tiny' model behavior for speed/stability in cloud if whisper isn't cached
            # But let's try real whisper if installed
            try:
                model = whisper.load_model("tiny")
                result = model.transcribe("clean.mp3")
                st.success(f"**Whisper Output:** '{result['text']}'")
                st.caption("✅ The model learned nothing new.")
            except Exception as e:
                st.error(f"Model Error: {e}")

    with col2:
        st.error("🔴 **Intelligent Selection** (Noisy Data)")
        if st.button("Generate Noisy Sample"):
            # Create Noisy Audio
            text = "The quick brown fox jumps over the lazy dog."
            tts = gTTS(text, lang='en')
            tts.save("temp.mp3")
            data, fs = sf.read("temp.mp3")
            noise = np.random.normal(0, 0.05, len(data))
            sf.write("noisy.wav", data + noise, fs)
            st.audio("noisy.wav")

            try:
                model = whisper.load_model("tiny")
                result = model.transcribe("noisy.wav")
                st.warning(f"**Whisper Output:** '{result['text']}'")
                st.caption("⚠️ The model struggled! This sample has **High Information Value**.")
            except:
                st.error("Model Error")

# ==========================================
# TASK 2: TTS DEMO
# ==========================================
elif task == "2. TTS (Purity)":
    st.header("2. Text-to-Speech (TTS)")
    st.subheader("The Challenge: Noise is Poison")
    st.markdown("""
    Unlike ASR, training TTS on noisy data creates glitches in the synthetic voice.
    We use **Segment-Level Pruning** to cut out bad artifacts.
    """)

    # Generate Signal
    t = np.linspace(0, 2.0, 32000)
    clean_sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Inject Artifact
    glitch = np.zeros_like(clean_sig)
    glitch[10000:15000] = np.random.normal(0, 1, 5000)
    dirty_sig = clean_sig + glitch
    
    # Prune
    pruned_sig = np.concatenate([dirty_sig[:10000], dirty_sig[15000:]])

    col1, col2 = st.columns(2)
    
    with col1:
        st.error("1. Original Noisy Data (Unusable)")
        sf.write("dirty_tts.wav", dirty_sig, 16000)
        st.audio("dirty_tts.wav")
        st.caption("Contains a heavy artifact.")

    with col2:
        st.success("2. Pruned Data (Usable)")
        sf.write("pruned_tts.wav", pruned_sig, 16000)
        st.audio("pruned_tts.wav")
        st.caption("Artifact surgically removed.")

# ==========================================
# TASK 3: ANTI-SPOOFING DEMO
# ==========================================
elif task == "3. Anti-Spoofing (Robustness)":
    st.header("3. Audio Anti-Spoofing")
    st.subheader("The Challenge: The Moving Target")
    st.markdown("""
    To detect unseen attacks, we focus on **Forgetting Events**—samples the model struggles to remember during training.
    """)

    epochs = np.arange(1, 11)
    easy_acc = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    hard_acc = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, easy_acc, 'g-o', label="Easy Sample (Stable)", linewidth=3)
    ax.plot(epochs, hard_acc, 'r--x', label="Hard Sample (High Forgetting)", linewidth=3)
    ax.set_title("Training Dynamics: Forgetting Events")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correctness")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    st.caption("The 'Hard Sample' flips back and forth. This boundary case is critical for generalization.")

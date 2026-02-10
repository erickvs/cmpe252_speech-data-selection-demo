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
    layout="wide"
)

st.title("Data Selection for Efficient Speech Processing")
st.markdown("**Based on the survey by Azeemi et al. (2025)**")

# Sidebar for Navigation
task = st.sidebar.radio("Select Speech Task:", ["1. ASR (Diversity)", "2. TTS (Purity)", "3. Anti-Spoofing (Robustness)"])

# ==========================================
# TASK 1: ASR DEMO
# ==========================================
if task == "1. ASR (Diversity)":
    st.header("1. Automatic Speech Recognition (ASR)")
    
    # --- PHASE 1: DEFINITION (Slide 2) ---
    st.subheader("Part A: State-of-the-Art Definition (Slide 2)")
    st.markdown("Run OpenAI's **Whisper** model on standard, clear speech.")
    
    if st.button("🎙️ Run ASR Demo"):
        text = "Automatic Speech Recognition converts spoken language into text."
        tts = gTTS(text, lang='en')
        tts.save("def.mp3")
        st.audio("def.mp3")
        
        try:
            model = whisper.load_model("tiny")
            result = model.transcribe("def.mp3")
            st.success(f"**Transcript:** '{result['text']}'")
            st.info("✅ Result: The model understands standard data perfectly.")
        except Exception as e:
            st.error(f"Model Error: {e}")

    st.divider()

    # --- PHASE 2: THE SOLUTION (Slide 5) ---
    st.subheader("Part B: The Selection Solution (Slide 5)")
    st.markdown("""
    **The Problem:** Random selection just picks more "Easy" data (like above).
    **The Solution:** We must find the **High Uncertainty** samples.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Random Selection (Easy Data)")
        if st.button("🎲 Select Randomly"):
            text = "This is standard training data."
            tts = gTTS(text, lang='en')
            tts.save("easy.mp3")
            st.audio("easy.mp3")
            
            try:
                model = whisper.load_model("tiny")
                result = model.transcribe("easy.mp3")
                st.write(f"**Transcript:** '{result['text']}'")
                st.info("Low Entropy (Model is bored).")
            except:
                st.error("Error")

    with col2:
        st.caption("Intelligent Selection (Hard Data)")
        if st.button("🧠 Select via Entropy"):
            # Create Noisy Audio
            text = "This is standard training data."
            tts = gTTS(text, lang='en')
            tts.save("temp.mp3")
            data, fs = sf.read("temp.mp3")
            noise = np.random.normal(0, 0.05, len(data))
            sf.write("hard.wav", data + noise, fs)
            st.audio("hard.wav")

            try:
                model = whisper.load_model("tiny")
                result = model.transcribe("hard.wav")
                st.write(f"**Transcript:** '{result['text']}'")
                st.warning("High Entropy (Model is learning!).")
            except:
                st.error("Error")

# ==========================================
# TASK 2: TTS DEMO
# ==========================================
elif task == "2. TTS (Purity)":
    st.header("2. Text-to-Speech (TTS)")
    st.subheader("The Challenge: Impact of Noise on Synthesis")
    st.markdown("""
    Unlike ASR, training TTS on noisy data creates artifacts in the synthetic voice.
    We use **Segment-Level Pruning** to remove detrimental segments.
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
        st.caption("Artifact removed.")

# ==========================================
# TASK 3: ANTI-SPOOFING DEMO
# ==========================================
elif task == "3. Anti-Spoofing (Robustness)":
    st.header("3. Audio Anti-Spoofing")
    st.subheader("The Challenge: Adversarial Adaptation")
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
    st.caption("The 'Hard Sample' oscillates between correct and incorrect. This boundary case is critical for generalization.")

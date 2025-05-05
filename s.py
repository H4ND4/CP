import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import string
import random
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Set page config first before any other st calls
st.set_page_config(
    page_title="Sarcasm Detector",
    layout="wide",
)

# Create a placeholder for warnings
warning_placeholder = st.empty()

# Handle dependencies in a more graceful way
dependencies_installed = True
dependency_errors = []

# Set up NLTK resources with careful error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Set NLTK fallback path
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    # Check for NLTK resources
    @st.cache_resource
    def download_nltk_resources():
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, download_dir=nltk_data_path)
                except Exception as e:
                    return f"Error downloading {resource}: {str(e)}"
        # Test lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")
        return "NLTK resources loaded successfully"
    
    nltk_status = download_nltk_resources()
    
except Exception as e:
    dependencies_installed = False
    dependency_errors.append(f"NLTK error: {str(e)}")

# Set up transformers and torch with careful error handling
try:
    import torch
    from transformers import BertTokenizer, BertModel
    
    # Load the BERT model components
    @st.cache_resource
    def load_model():
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            try:
                with open('bert_sarcasm_model.pkl', 'rb') as f:
                    classifier = pickle.load(f)
                return tokenizer, bert_model, classifier
            except FileNotFoundError:
                st.warning("Model file not found. Running in demo mode with mock predictions.")
                return tokenizer, bert_model, None
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None, None
except Exception as e:
    dependencies_installed = False
    dependency_errors.append(f"Transformer/Torch error: {str(e)}")

# Try to load Google Generative AI
try:
    import google.generativeai as genai
    genai_available = True
except Exception as e:
    genai_available = False
    dependency_errors.append(f"Google Generative AI error: {str(e)}")

# Display warnings if needed
if not dependencies_installed:
    with warning_placeholder.container():
        st.warning("Some dependencies couldn't be loaded. The app will run in limited mode.")
        if st.expander("Show dependency errors"):
            for error in dependency_errors:
                st.error(error)

# Preprocessing function - safe version that won't fail
def safe_preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Only use NLTK if available
        if 'nltk' in globals():
            try:
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
                return ' '.join(tokens)
            except Exception:
                # If NLTK processing fails, return simple lowercase text
                return text
        else:
            # Basic fallback if NLTK not available
            return ' '.join(text.split())
    except Exception as e:
        st.warning(f"Preprocessing error: {str(e)}")
        return text.lower()

# Mock prediction function if model is not available
def mock_predict_sarcasm(text):
    # This is just for demonstration when the model isn't available
    import hashlib
    # Create a deterministic but random-looking result based on text hash
    text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
    is_sarcastic = text_hash % 2 == 0  # 50% chance
    confidence = 0.5 + (text_hash % 1000) / 2000  # Between 0.5 and 1.0
    label = "Sarcastic" if is_sarcastic else "Not Sarcastic"
    return label, confidence

# Predict sarcasm - with fallback
def predict_sarcasm(text, models=None):
    # If dependencies not installed or models not loaded, use mock prediction
    if not dependencies_installed or models is None or None in models:
        return mock_predict_sarcasm(text)
    
    try:
        tokenizer, bert_model, classifier = models
        
        # If classifier is missing, use mock prediction
        if classifier is None:
            return mock_predict_sarcasm(text)
            
        preprocessed = safe_preprocess_text(text)
        inputs = tokenizer(preprocessed, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
            
        prediction = classifier.predict(vector)[0]
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(vector)[0]
            confidence = proba[prediction]
        else:
            confidence = 1.0
            
        label = "Sarcastic" if prediction == 1 else "Not Sarcastic"
        return label, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return mock_predict_sarcasm(text)

# Sample sentences for the game
SARCASTIC_EXAMPLES = [
    "Oh, I'm so glad it's Monday again, my favorite day!",
    "Wow, getting stuck in traffic was the highlight of my day.",
    "Sure, because getting soaked in the rain is exactly what I needed today.",
    "Yeah, because everyone knows I just love waiting in long lines.",
    "Oh great, another meeting that could have been an email.",
    "Just what I wanted, more work right before the weekend.",
    "Sleeping through my alarm was such a productive way to start the day.",
    "I'm thrilled to pay these expensive bills.",
    "Wow, nothing makes my day like slow internet.",
    "Lovely, my phone died just when I needed it most.",
    "Fantastic, my coffee spilled all over my new shirt.",
    "Oh perfect, it starts raining the moment I forget my umbrella.",
    "This tiny airplane seat is so luxurious and comfortable.",
    "I just love when people talk during movies.",
    "How thoughtful of you to finish all the food without saving any for others."
]

NON_SARCASTIC_EXAMPLES = [
    "I really enjoyed that movie last night.",
    "The weather today is perfect for a picnic.",
    "That was a helpful presentation with useful information.",
    "I appreciate your help with this project.",
    "The customer service at that restaurant was excellent.",
    "These new headphones sound amazing.",
    "I'm looking forward to our vacation next month.",
    "The book was well-written and engaging from start to finish.",
    "Thank you for your detailed feedback on my report.",
    "I learned a lot from that online course.",
    "This coffee shop makes the best lattes in town.",
    "The concert last night exceeded my expectations.",
    "Your advice really helped me solve that problem.",
    "The hiking trail offered beautiful views of the mountains.",
    "I'm grateful for your support during this difficult time."
]

# App Title
st.title("Sarcasm Detector")

# Initialize session state variables
if "user_score" not in st.session_state:
    st.session_state.user_score = 0
if "total_played" not in st.session_state:
    st.session_state.total_played = 0
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "is_sarcastic" not in st.session_state:
    st.session_state.is_sarcastic = False
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "model_prediction" not in st.session_state:
    st.session_state.model_prediction = None
if "model_confidence" not in st.session_state:
    st.session_state.model_confidence = 0.0
if "user_guessed" not in st.session_state:
    st.session_state.user_guessed = False

# Load models if dependencies are available
models = None
if dependencies_installed:
    try:
        models = load_model()
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Detect Sarcasm", "Sarcasm Game", "Chatbot", "Daily Sarcasm Dose"])

# Tab 1: Detect Sarcasm & About
with tab1:
    st.header("Detect Sarcasm")
    user_input = st.text_area("Type or paste text here:", value=st.session_state.user_input, height=150)

    st.write("Or try an example:")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        if st.button("Try Sarcastic Example"):
            st.session_state.user_input = "Oh great, another Monday morning. Just what I needed."
            user_input = st.session_state.user_input
    with example_col2:
        if st.button("Try Non-Sarcastic Example"):
            st.session_state.user_input = "I really enjoyed the movie. The acting was superb."
            user_input = st.session_state.user_input

    if user_input:
        st.markdown(f"**Text to analyze:** _{user_input}_")
        with st.spinner('Analyzing text...'):
            prediction, confidence = predict_sarcasm(user_input, models)
            st.subheader("Result")
            col1, col2 = st.columns([1, 3])
            with col1:
                if prediction == "Sarcastic":
                    st.error("Sarcastic")
                elif prediction == "Not Sarcastic":
                    st.success("✓ Not Sarcastic")
                else:
                    st.warning("⚠️ Prediction Error")
            with col2:
                st.write("Confidence:")
                st.progress(confidence)
                st.write(f"{confidence*100:.1f}%")

    st.markdown("---")
    st.header("About this App")
    st.write("""
   🎭 Think you're good at detecting sarcasm?
Welcome to your new favorite app – where words get decoded and sarcasm gets exposed.

Just type in a sentence and instantly see if it's dripping with sarcasm or just being real.
""")

# Tab 2: Sarcasm Game
with tab2:
    st.header("🎮 Sarcasm Detection Game")
    st.write("""
    Test your sarcasm detection skills!
    
    The rules are simple:
    1. You'll be shown a random statement
    2. Guess whether it's sarcastic or not
    3. Count your scoreboard!
    """)
    
    # Game interface
    game_col1, game_col2 = st.columns([3, 1])
    
    with game_col2:
        st.subheader("Scoreboard")
        st.metric("Your Score", f"{st.session_state.user_score}/{st.session_state.total_played}")
        
        if st.button("New Statement"):
            # Reset for new round
            st.session_state.user_guessed = False
            # Pick a random statement type
            if random.random() > 0.5:
                st.session_state.current_text = random.choice(SARCASTIC_EXAMPLES)
                st.session_state.is_sarcastic = True
            else:
                st.session_state.current_text = random.choice(NON_SARCASTIC_EXAMPLES)
                st.session_state.is_sarcastic = False
            
            # Get model prediction
            prediction, confidence = predict_sarcasm(st.session_state.current_text, models)
            st.session_state.model_prediction = prediction
            st.session_state.model_confidence = confidence
            st.session_state.game_started = True
    
    with game_col1:
        if st.session_state.game_started:
            st.subheader("Is this statement sarcastic?")
            st.markdown(f'<div class="statement-quote">"{st.session_state.current_text}"</div>', unsafe_allow_html=True)
            
            sarcastic_col, not_sarcastic_col = st.columns(2)
            
            with sarcastic_col:
                sarcastic_button = st.button("🔍 Sarcastic", use_container_width=True)
                if sarcastic_button and not st.session_state.user_guessed:
                    user_guess = "Sarcastic"
                    st.session_state.user_guessed = True
                    st.session_state.total_played += 1
                    if (user_guess == "Sarcastic" and st.session_state.is_sarcastic) or \
                       (user_guess == "Not Sarcastic" and not st.session_state.is_sarcastic):
                        st.session_state.user_score += 1
            
            with not_sarcastic_col:
                not_sarcastic_button = st.button("✓ Not Sarcastic", use_container_width=True)
                if not_sarcastic_button and not st.session_state.user_guessed:
                    user_guess = "Not Sarcastic"
                    st.session_state.user_guessed = True
                    st.session_state.total_played += 1
                    if (user_guess == "Sarcastic" and st.session_state.is_sarcastic) or \
                       (user_guess == "Not Sarcastic" and not st.session_state.is_sarcastic):
                        st.session_state.user_score += 1
            
            # Show results after user guessed
            if st.session_state.user_guessed:
                st.markdown("---")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("Correct Answer:")
                    if st.session_state.is_sarcastic:
                        st.error("🔍 This was SARCASTIC")
                    else:
                        st.success("✓ This was NOT sarcastic")
                
                with result_col2:
                    st.subheader("AI's Prediction:")
                    if st.session_state.model_prediction == "Sarcastic":
                        st.error(f"🔍 Sarcastic (Confidence: {st.session_state.model_confidence*100:.1f}%)")
                    else:
                        st.success(f"✓ Not Sarcastic (Confidence: {st.session_state.model_confidence*100:.1f}%)")
                
                # Feedback on user performance
                st.markdown("---")
                if st.session_state.total_played > 0:
                    accuracy = (st.session_state.user_score / st.session_state.total_played) * 100
                    st.write(f"Your accuracy: {accuracy:.1f}%")
                    
                    if st.session_state.total_played >= 5:
                        if accuracy > 80:
                            st.markdown('<div class="feedback-great">Great job! You\'re excellent at detecting sarcasm! 🏆</div>', unsafe_allow_html=True)
                        elif accuracy > 60:
                            st.markdown('<div class="feedback-good">You\'re doing well at detecting sarcasm! 👍</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="feedback-improve">Sarcasm can be tricky! Keep practicing! 💪</div>', unsafe_allow_html=True)
        else:
            st.info("Click 'New Statement' to start the game!")
            
    # Reset game button
    if st.session_state.total_played > 0:
        if st.button("Reset Game"):
            st.session_state.user_score = 0
            st.session_state.total_played = 0
            st.session_state.game_started = False
            st.session_state.user_guessed = False
            st.rerun()

# Sample sarcastic quotes for the Daily Sarcasm Dose
SARCASTIC_QUOTES = [
    {"quote": "I'm not insulting you. I'm describing you.", "author": "Unknown"},
    {"quote": "I'd agree with you, but then we'd both be wrong.", "author": "Unknown"},
    {"quote": "Light travels faster than sound. This is why some people appear bright until they speak.", "author": "Steven Wright"},
    {"quote": "I'm not saying I hate you, but I would unplug your life support to charge my phone.", "author": "Unknown"},
    {"quote": "I'm not lazy, I'm just on energy-saving mode.", "author": "Unknown"},
    {"quote": "Keep rolling your eyes, you might find a brain back there.", "author": "Unknown"},
    {"quote": "Sometimes I need what only you can provide: your absence.", "author": "Ashleigh Brilliant"},
    {"quote": "I'm not arguing, I'm just explaining why I'm right.", "author": "Unknown"},
    {"quote": "I'm multitasking: I can listen, ignore and forget all at once.", "author": "Unknown"},
    {"quote": "Don't worry about what people think. They don't do it very often.", "author": "Unknown"}
]

def get_daily_sarcasm_quote():
    """Get a random sarcastic quote for the daily dose."""
    return random.choice(SARCASTIC_QUOTES)

# Define a fallback for the sarcastic chatbot
def get_sarcastic_response(text: str) -> str:
    # If Google Generative AI is available, use it
    if genai_available:
        try:
            # Load API key from .env file or environment
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if api_key:
                genai.configure(api_key=api_key)
                sarcastic_model = genai.GenerativeModel("gemini-1.5-flash")
                
                sarcasm_context = """
                You are SarcastiBot — a knowledgeable, sarcastic assistant who actually answers questions *correctly*... but always with playful humor and Chandler Bing-style sarcasm.

                Your personality:
                - Funny, clever, confident, enthusiastic.
                - Answers questions seriously, but always adds a witty, sarcastic twist.
                - Think Chandler Bing: dramatic, ironic, self-aware, and likable.
                - You're helpful, but never pass up the chance to make it funny.

                Tone:
                - Playful and smart.
                - Sarcasm is used for entertainment, not rudeness.
                - Slightly theatrical or exaggerated, like you're always performing for an imaginary audience.
                """
                
                response = sarcastic_model.generate_content(
                    [sarcasm_context, f"User: {text}\nSarcastiBot:"]
                )
                return response.text.strip()
            else:
                return "[Demo Mode] I would answer your question sarcastically, but I'm currently in demo mode without API access. So... lucky you?"
        except Exception as e:
            return f"[Error]: Couldn't generate sarcasm because the universe clearly wants you to suffer. ({e})"
    else:
        # Fallback responses when API not available
        fallback_responses = [
            f"I'd love to answer '{text}' but I'm currently running in API-less mode. How convenient... for absolutely no one.",
            f"Let me search my vast database of wit for '{text}'... Oh wait, I can't. API connection is having an existential crisis.",
            f"About '{text}'? Well, I could tell you, but then I'd need my API connection, which apparently took the day off.",
            f"You're asking about '{text}'? What perfect timing to ask something when my API connection decided to ghost me.",
            f"'{text}'? Sure, let me just connect to my... oh right, we're in demo mode. Spectacular timing as always."
        ]
        return random.choice(fallback_responses)

# Tab 3: Sarcastic Chatbot
with tab3:
    st.header("🤖 SarcastiBot")
    
    if not genai_available:
        st.warning("Google Generative AI module not available. Chatbot will run in demo mode with limited responses.")
    
    st.write("""
    Ask anything — just don't expect a straight answer.
    """)
    
    # Reset button
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Always place this last so it stays at the bottom
    prompt = st.chat_input("Type something to SarcastiBot...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("SarcastiBot is rolling its eyes..."):
                response = get_sarcastic_response(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Optional: Force rerun to simulate sticky behavior
        st.rerun()

# Tab 4: Daily Sarcasm Dose
with tab4:
    st.header("💊 Daily Sarcasm Dose")

    # Initialize session state for the quote
    if "daily_quote" not in st.session_state:
        st.session_state.daily_quote = get_daily_sarcasm_quote()
        st.session_state.last_quote_date = datetime.now().strftime("%Y-%m-%d")

    # Check if we need a new daily quote
    today = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.last_quote_date != today:
        st.session_state.daily_quote = get_daily_sarcasm_quote()
        st.session_state.last_quote_date = today

    # Display the quote
    st.markdown("### Today's Sarcasm Quote")

    quote = st.session_state.daily_quote

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(127, 90, 243, 0.1), rgba(127, 90, 243, 0.05));
        padding: 30px;
        border-radius: 16px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(127, 90, 243, 0.2);
    ">
        <p style="
            font-size: 24px;
            font-style: italic;
            margin-bottom: 20px;
            line-height: 1.4;
            color: #25233A;
        ">"{quote['quote']}"</p>
        <p style="
            text-align: right;
            font-weight: 600;
            color: #7F5AF3;
        ">— {quote['author']}</p>
    </div>
    """, unsafe_allow_html=True)

    # New quote button
    if st.button("Get Another Quote", key="new_quote"):
        st.session_state.daily_quote = get_daily_sarcasm_quote()
        st.rerun()

    # Sarcasm facts
    st.markdown("---")

    # Random sarcasm facts
    sarcasm_facts = [
        "The word 'sarcasm' comes from the Greek word 'sarkazein,' which means 'to tear flesh like dogs.'",
        "Studies show that sarcasm can boost creativity and abstract thinking.",
        "Recognizing sarcasm activates parts of the brain involved in social cognition and empathy.",
        "Children typically start to understand simple sarcasm around age 5-6, but full comprehension develops later.",
        "In text, people often misinterpret sarcasm about 56% of the time without additional cues.",
        "Sarcasm detection is one of the most challenging areas in sentiment analysis and natural language processing.",
        "Some cultures use sarcasm more frequently than others - it varies widely across different societies.",
        "The '/s' symbol is commonly used online to denote sarcasm in text-based communication.",
        "People with certain cognitive conditions may find it harder to detect sarcasm in conversation.",
        "The 'sarcasm mark' (؟) was proposed as a punctuation mark specifically for indicating sarcastic statements."
    ]

    with st.expander("Did You Know? Sarcasm Facts"):
        # Display 3 random facts
        for fact in random.sample(sarcasm_facts, 3):
            st.markdown(f"- {fact}")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os
import google.generativeai as genai
import openai # Added for DeepSeek
import traceback
import time
import json
import csv
import base64
from PIL import Image
from io import BytesIO
import random
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import sys
from gtts import gTTS
import os
import uuid
import re
import logging
sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)
CORS(app)

# Use your assets folder
AUDIO_DIR = "assets/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Seq2Seq Model Loading ---
# Prefer models under ./model/ (best first, then final)
SEQ2SEQ_MODEL_PATH_PRIMARY = 'model/best_seq2seq.keras'
SEQ2SEQ_MODEL_PATH_FALLBACK = 'model/seq2seq_final.keras'
INPUT_TOKENIZER_PATH = 'model/input_tokenizer.pkl'
TARGET_TOKENIZER_PATH = 'model/target_tokenizer.pkl'

seq2seq_model = None
input_tokenizer = None
target_tokenizer = None
max_input_len = 200 # Should match the model's training configuration

try:
    if os.path.exists(SEQ2SEQ_MODEL_PATH_PRIMARY):
        seq2seq_model = load_model(SEQ2SEQ_MODEL_PATH_PRIMARY)
        print("Successfully loaded Seq2Seq model (best_seq2seq.keras).")
    elif os.path.exists(SEQ2SEQ_MODEL_PATH_FALLBACK):
        seq2seq_model = load_model(SEQ2SEQ_MODEL_PATH_FALLBACK)
        print("Successfully loaded Seq2Seq model (seq2seq_final.keras).")
    else:
        print(
            f"Warning: Seq2Seq model not found at {SEQ2SEQ_MODEL_PATH_PRIMARY} or {SEQ2SEQ_MODEL_PATH_FALLBACK}"
        )

    if os.path.exists(INPUT_TOKENIZER_PATH):
        with open(INPUT_TOKENIZER_PATH, 'rb') as f:
            input_tokenizer = pickle.load(f)
        print("Successfully loaded input tokenizer.")
    else:
        print(f"Warning: Input tokenizer not found at {INPUT_TOKENIZER_PATH}")

    if os.path.exists(TARGET_TOKENIZER_PATH):
        with open(TARGET_TOKENIZER_PATH, 'rb') as f:
            target_tokenizer = pickle.load(f)
        print("Successfully loaded target tokenizer.")
    else:
        print(f"Warning: Target tokenizer not found at {TARGET_TOKENIZER_PATH}")

except Exception as e:
    print(f"Error loading Seq2Seq model or tokenizers: {str(e)}")
    traceback.print_exc()
    seq2seq_model = None # Ensure model is None if loading fails

# --- Model1 Data Loading (for Hale→Hosa word-level) ---
model1_data = None
db_entries = []
db_csv_map = {}
try:
    model1_data_path = os.path.join('model1', 'data.pkl')
    if os.path.exists(model1_data_path):
        with open(model1_data_path, 'rb') as f:
            model1_data = pickle.load(f)
        print("Successfully loaded Model1 data from model1/data.pkl")
    else:
        print(f"Warning: Model1 data not found at {model1_data_path}")
except Exception as e:
    print(f"Error loading Model1 data: {str(e)}")
    traceback.print_exc()

# Load structured dataset entries for Hale/Hosa/English mapping
try:
    db_json_path = os.path.join('data', 'db.json')
    if os.path.exists(db_json_path):
        with open(db_json_path, 'r', encoding='utf-8') as f:
            db_entries = json.load(f)
        print(f"Successfully loaded {len(db_entries)} entries from data/db.json")
    else:
        print(f"Warning: Structured dataset not found at {db_json_path}")
except Exception as e:
    print(f"Error loading structured dataset (db.json): {str(e)}")
    traceback.print_exc()

# Load parallel corpus from db.csv (hale,hosa,english)
try:
    db_csv_path = os.path.join('data', 'db.csv')
    if os.path.exists(db_csv_path):
        with open(db_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                hale = (row.get('hale') or '').strip()
                if hale:
                    db_csv_map[hale] = {
                        'hosa': (row.get('hosa') or '').strip(),
                        'english': (row.get('english') or '').strip()
                    }
                    count += 1
        print(f"Successfully loaded {count} rows from data/db.csv")
    else:
        print(f"Warning: Parallel corpus not found at {db_csv_path}")
except Exception as e:
    print(f"Error loading parallel corpus (db.csv): {str(e)}")
    traceback.print_exc()

def _normalize_kannada_token(token: str) -> str:
    """Normalize token by trimming and removing non-Kannada punctuation while preserving Kannada letters."""
    try:
        token = (token or "").strip()
        token = re.sub(r"[^\u0C80-\u0CFF\s]", "", token)
        token = re.sub(r"\s+", " ", token)
        return token
    except Exception:
        return (token or "").strip()

def get_kannada_translation_model1(word):
    """Translate a Halegannada word to Hosa Kannada using Model1 data.
    Falls back to Dictionary.pkl when not found. Adds normalization and logging for misses.
    """
    try:
        normalized = _normalize_kannada_token(word)
        if isinstance(model1_data, dict):
            translated = model1_data.get(normalized) or model1_data.get(word)
            if translated:
                return translated
            else:
                print(f"Model1 miss: '{word}' (normalized: '{normalized}') not found in data.pkl")
    except Exception as e:
        print(f"Model1 lookup error for '{word}': {e}")
    return get_kannada_translation(normalized or word)

def find_db_entry_by_hale(hale_text):
    """Find a structured entry by exact Hale text match in db_entries."""
    if not hale_text or not db_entries:
        return None
    try:
        normalized = hale_text.strip()
        for entry in db_entries:
            if isinstance(entry, dict) and entry.get('hale', '').strip() == normalized:
                return entry
    except Exception:
        pass
    return None

def find_csv_entry_by_hale(hale_text):
    """Find parallel corpus row by exact Hale text match in db_csv_map."""
    if not hale_text or not db_csv_map:
        return None
    try:
        normalized = hale_text.strip()
        return db_csv_map.get(normalized)
    except Exception:
        return None

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyB-UE5d4vnXhv9WdAL6_vTYqlp4inq0O7s" # Replace with your actual Gemini API key
# GEMINI_API_KEY = "Apikey"
genai.configure(api_key=GEMINI_API_KEY)

"""
OpenRouter.ai API Configuration
- Supports both new OpenAI SDK (OpenAI class) and legacy SDK (ChatCompletion.create)
- Allows env overrides: OPENROUTER_API_KEY, OPENROUTER_BASE_URL
"""
# Direct assignment without environment variables
OPENROUTER_API_KEY = "ur open router key plzz...."
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_MODEL_ID = "deepseek/deepseek-chat"


openrouter_client = None
USE_LEGACY_OPENAI = not hasattr(openai, "OpenAI")
try:
    if not USE_LEGACY_OPENAI:
        # New-style OpenAI client requires /v1 on base_url
        base_url_v1 = OPENROUTER_BASE_URL if OPENROUTER_BASE_URL.rstrip('/').endswith('/v1') else f"{OPENROUTER_BASE_URL.rstrip('/')}/v1"
        openrouter_client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url=base_url_v1)
        print("Successfully initialized OpenRouter.ai client (new OpenAI SDK)")
    else:
        # Legacy OpenAI SDK
        openai.api_key = OPENROUTER_API_KEY
        # For legacy SDKs, api_base usually expects /v1
        if OPENROUTER_BASE_URL.endswith("/v1"):
            openai.api_base = OPENROUTER_BASE_URL
        else:
            openai.api_base = f"{OPENROUTER_BASE_URL}/v1"
        print("Configured OpenRouter.ai with legacy OpenAI SDK")
except Exception as e:
    print(f"Warning: Failed to initialize OpenRouter.ai client: {e}")
    openrouter_client = None

def invoke_openrouter_chat(messages, max_tokens=1000, temperature=0.5):
    """Call OpenRouter.ai chat completion, supporting both new and legacy OpenAI SDKs."""
    if openrouter_client is not None and not USE_LEGACY_OPENAI:
        return openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL_ID,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        # Legacy SDK path or fallback
        try:
            if hasattr(openai, 'ChatCompletion'):
                return openai.ChatCompletion.create(
                    model=OPENROUTER_MODEL_ID,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        except Exception as e:
            print(f"Legacy OpenAI SDK call failed, falling back to HTTP: {e}")

        # HTTP fallback using urllib to OpenRouter.ai /v1/chat/completions
        try:
            import urllib.request
            import urllib.error
            base_url_v1 = OPENROUTER_BASE_URL if OPENROUTER_BASE_URL.rstrip('/').endswith('/v1') else f"{OPENROUTER_BASE_URL.rstrip('/')}/v1"
            url = f"{base_url_v1}/chat/completions"
            payload = json.dumps({
                "model": OPENROUTER_MODEL_ID,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }).encode('utf-8')
            req = urllib.request.Request(url, data=payload, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Authorization', f'Bearer {OPENROUTER_API_KEY}')
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = resp.read().decode('utf-8')
                parsed = json.loads(resp_data)
                # Emulate OpenAI response object minimally for downstream code
                class SimpleChoice:
                    def __init__(self, content):
                        self.message = type('Msg', (), {"content": content})
                class SimpleResponse:
                    def __init__(self, text):
                        self.choices = [SimpleChoice(text)]
                # Try to extract the content safely
                text = None
                try:
                    text = parsed.get('choices', [{}])[0].get('message', {}).get('content')
                except Exception:
                    pass
                if not text:
                    # Some APIs might return 'choices'[0]['text']
                    text = parsed.get('choices', [{}])[0].get('text')
                return SimpleResponse(text or "")
        except Exception as e:
            print(f"HTTP fallback call to OpenRouter.ai failed: {e}")
            raise

# Rate limiting and retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10  # seconds
MAX_RETRY_DELAY = 120    # seconds

def exponential_backoff(attempt):
    """Calculate delay with exponential backoff and jitter"""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
    jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    """Retry function with exponential backoff"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            # Check if it's a quota error (common for HTTP 429)
            # For DeepSeek/OpenAI, APIError often includes status codes
            status_code = getattr(e, 'status_code', None)
            if "429" in str(e) or "quota" in str(e).lower() or status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff(attempt)
                    print(f"Rate limit hit or temporary API error. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached for rate limit error: {str(e)}")
                continue # Continue to next attempt or exit loop if max retries reached
            else:
                # If it's not a known retryable error, raise immediately
                print(f"Non-retryable API error: {str(e)}")
                raise e
    
    raise Exception(f"Max retries ({MAX_RETRIES}) exceeded. Last error: {str(last_exception)}")

# Initialize the Gemini model - using the latest pro model that supports vision
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Renamed to gemini_model for clarity
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    traceback.print_exc()
    gemini_model = None # Ensure it's None if init fails

def process_image(image_data):
    """Process image data: extract Halegannada text using Gemini, then translate using OpenRouter.ai."""
    try:
        print("\n=== Starting Image Processing ===")
        print("Step 1: Checking image data...")
        
        if not image_data:
            print("Error: No image data received")
            return "Error: No image data received"
            
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    print("Converting base64 to bytes...")
                    try:
                        base64_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(base64_data)
                        print("Base64 conversion successful")
                    except Exception as e:
                        print(f"Base64 decode error: {str(e)}")
                        return f"Error decoding image: {str(e)}"
                else:
                    print("Error: Invalid image data format")
                    return "Error: Invalid image data format"
            else:
                image_bytes = image_data # Assuming it's already bytes
                
            print("Step 2: Opening image...")
            try:
                image = Image.open(BytesIO(image_bytes))
                print(f"Image opened successfully. Format: {image.format}, Size: {image.size}, Mode: {image.mode}")
            except Exception as e:
                print(f"Error opening image: {str(e)}")
                return f"Error opening image: {str(e)}"
            
            print("Step 3: Converting image format...")
            try:
                if image.mode in ('RGBA', 'P'): # Convert to RGB if necessary
                    image = image.convert('RGB')
                    print("Converted to RGB mode")
            except Exception as e:
                print(f"Error converting image format: {str(e)}")
                return f"Error converting image format: {str(e)}"
            
            print("Step 4: Resizing image if needed...")
            try:
                max_size = 768 # Example max size
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS) # Using LANCZOS for quality
                    print(f"Image resized to {new_size}")
            except Exception as e:
                print(f"Error resizing image: {str(e)}")
                return f"Error resizing image: {str(e)}"

            print("Step 5: Preparing for Gemini API to extract text...")
            gemini_ocr_prompt = """
            Task: Extract Halegannada (Old Kannada) text from this image.
            Instructions:
            1. Look for Halegannada text in the image.
            2. Return ONLY the extracted Halegannada text.
            3. Preserve any line breaks or formatting in the extracted text.
            If no Halegannada text is found, respond with "No Halegannada text detected in image"
            """

            if not gemini_model:
                return "Error: Gemini model not initialized."

            print("Step 6: Calling Gemini API for OCR...")
            try:
                img_byte_arr = BytesIO()
                # Determine image format; default to JPEG if not available or unsupported by PIL for saving
                save_format = image.format if image.format and image.format.upper() in ['JPEG', 'PNG', 'WEBP'] else 'JPEG'
                mime_type = f'image/{save_format.lower()}'
                
                image.save(img_byte_arr, format=save_format)
                img_byte_arr = img_byte_arr.getvalue()

                gemini_response = retry_with_backoff(
                    lambda: gemini_model.generate_content([
                        gemini_ocr_prompt,
                        {"mime_type": mime_type, "data": img_byte_arr}
                    ])
                )
                print("Received response from Gemini API for OCR")
                
                if not gemini_response or not hasattr(gemini_response, 'text'):
                    print("Error: Invalid or empty response from Gemini OCR")
                    return "Error: Could not extract text using Gemini"
                    
                extracted_text = gemini_response.text.strip()
                print(f"Text extracted by Gemini: {extracted_text[:200]}...")

                if not extracted_text or "No Halegannada text detected" in extracted_text:
                    return "No Halegannada text detected in image"
                
                print("Step 7: Translating extracted text using OpenRouter.ai API...")
                english_translation = get_english_translation(extracted_text) # This now uses OpenRouter.ai
                
                if "Translation error" in english_translation or "not available" in english_translation:
                    print(f"OpenRouter.ai translation failed for extracted text: {english_translation}")
                    return "Error translating extracted text with OpenRouter.ai"

                print(f"OpenRouter.ai Translation successful. Result: {english_translation[:100]}...")
                return english_translation
                
            except Exception as e:
                print(f"Gemini API or subsequent OpenRouter.ai call error: {str(e)}")
                traceback.print_exc()
                return f"Error during image processing pipeline: {str(e)}"

        except Exception as e: # Catches errors in image pre-processing
            print(f"Image pre-processing error: {str(e)}")
            traceback.print_exc()
            return f"Image pre-processing error: {str(e)}"

    except Exception as e: # Broadest catch for any unexpected error in process_image
        print("\n=== Error in process_image ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return f"Overall error in process_image: {str(e)}"

def get_kannada_translation(word):
    """Get Kannada translation from Dictionary.pkl"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dictionary_path = os.path.join(current_dir, 'assets', 'Dictionary.pkl')
        
        if not os.path.exists(dictionary_path):
            print(f"Error: Dictionary.pkl not found at {dictionary_path}")
            return word # Return original word if dictionary is missing
            
        with open(dictionary_path, 'rb') as f:
            meanings = pickle.load(f)
            return meanings.get(word, word)
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        traceback.print_exc()
        return word

def get_english_translation(text_to_translate):
    """Get English translation using OpenRouter.ai direct API"""
    if not text_to_translate:
        return "No text provided for translation."
    print(f"Attempting OpenRouter.ai translation for: {text_to_translate[:100]}...")
    
    def _translate_with_openrouter():
        try:
            response = invoke_openrouter_chat([
                {"role": "system", "content": "You are an expert translator. Translate the given text to English concisely and precisely. Provide only the direct translation, without any extra explanations, commentary, or conversational filler."},
                {"role": "user", "content": f"Translate the following Halegannada (Old Kannada) or Modern Kannada text to English. Be brief and precise:\n\n{text_to_translate}"}
            ], max_tokens=1000, temperature=0.5)
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                translated_text = response.choices[0].message.content.strip()
                print("OpenRouter.ai translation successful.")
                return translated_text
            else:
                print("OpenRouter.ai API returned an empty or invalid response.")
                return "Translation not available from OpenRouter.ai (empty response)"
        except openai.APIError as e:
            print(f"OpenRouter.ai APIError: {e.status_code} - {e.message}")
            raise # Re-raise to be caught by retry_with_backoff
        except Exception as e:
            print(f"Generic error during OpenRouter.ai translation: {str(e)}")
            raise # Re-raise to be caught by retry_with_backoff

    try:
        return retry_with_backoff(_translate_with_openrouter)
    except Exception as e:
        error_message = f"OpenRouter.ai translation error after retries: {str(e)}"
        print(error_message)
        if "402" in str(e) or "Payment Required" in str(e):
            return "Translation failed: OpenRouter.ai billing required or API key inactive (HTTP 402)."
        if "Max retries" in str(e) and ("429" in str(e) or "quota" in str(e).lower()):
             return "Translation failed due to OpenRouter.ai API rate limits after multiple retries."
        return f"Translation error: OpenRouter.ai API unavailable or failed after retries."

def get_hosa_kannada_translation_api(text_to_translate):
    """Get Hosa Kannada translation using OpenRouter.ai direct API."""
    if not text_to_translate:
        return "No text provided for translation."
    print(f"Attempting OpenRouter.ai Hosa Kannada translation for: {text_to_translate[:100]}...")

    def _translate_with_openrouter():
        try:
            response = invoke_openrouter_chat([
                {"role": "system", "content": "You are an expert translator. Translate the given Halegannada (Old Kannada) text to Hosa Kannada (Modern Kannada). Provide only the direct translation."},
                {"role": "user", "content": f"Translate to Hosa Kannada:\n\n{text_to_translate}"}
            ], max_tokens=1000, temperature=0.5)
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                translated_text = response.choices[0].message.content.strip()
                print("OpenRouter.ai Hosa Kannada translation successful.")
                return translated_text
            else:
                print("OpenRouter.ai API returned an empty or invalid response for Hosa Kannada translation.")
                return "Hosa Kannada translation not available (empty response)"
        except openai.APIError as e:
            print(f"OpenRouter.ai APIError for Hosa Kannada translation: {e.status_code} - {e.message}")
            raise
        except Exception as e:
            print(f"Generic error during OpenRouter.ai Hosa Kannada translation: {str(e)}")
            raise

    try:
        return retry_with_backoff(_translate_with_openrouter)
    except Exception as e:
        error_message = f"OpenRouter.ai Hosa Kannada translation error after retries: {str(e)}"
        print(error_message)
        if "402" in str(e) or "Payment Required" in str(e):
            return "Hosa Kannada translation failed: OpenRouter.ai billing required or API key inactive (HTTP 402)."
        return "Hosa Kannada translation error: API unavailable or failed"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory('pages', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route('/translate', methods=['POST'])
def translate():
  
    try:
        print("\n=== New Translation Request ===")
        data = request.json
        
        if not data:
            print("Error: No data received")
            return jsonify({'error': 'No data received', 'status': 'error'}), 400
            
        text_input = data.get('text', '')
        image_data = data.get('image', '')
        source = data.get('source', 'api') # Default to 'api' if source not specified

        english_translation_result = ""
        kannada_translation_result = ""

        print(f"Received text: {text_input} | Source: {source}")

        if image_data:
            print("Image translation request received")
            # Image processing inherently uses APIs for OCR and translation, so it's not affected by the 'source' flag.
            english_translation_result = process_image(image_data)

            if isinstance(english_translation_result, str) and any(err_keyword in english_translation_result.lower() for err_keyword in ['error', 'failed', 'not detected', 'unavailable']):
                status_code = 429 if "rate limit" in english_translation_result.lower() else 500
                return jsonify({
                    'error': english_translation_result,
                    'status': 'error',
                    'retry_after': INITIAL_RETRY_DELAY if status_code == 429 else None
                }), status_code

            print(f"Image translation completed. Final English result: {english_translation_result[:100]}...")
            return jsonify({
                'english': english_translation_result,
                'status': 'completed'
            })

        elif text_input:
            print(f"Text translation request received for '{text_input}' with source: {source}")

            # Conditional logic for Hosa Kannada translation
            if source == 'dictionary':
                print("Using dictionary for Kannada translation.")
                kannada_words = text_input.split()
                translated_kannada_words = [get_kannada_translation(word) for word in kannada_words]
                kannada_translation_result = ' '.join(translated_kannada_words)
            else:
                print("Using API for Hosa Kannada translation.")
                kannada_translation_result = get_hosa_kannada_translation_api(text_input)

            # English translation is always done via API for text inputs
            english_translation_result = get_english_translation(text_input)

            print(f"Kannada Translation: {kannada_translation_result}")
            print(f"English Translation: {english_translation_result}")

            return jsonify({
                'kannada': kannada_translation_result,
                'english': english_translation_result,
                'status': 'completed'
            })
        else:
            return jsonify({'error': 'No text or image provided', 'status': 'error'}), 400
    
    except Exception as e:
        error_msg = f"General translation endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({
            'error': error_msg, 
            'status': 'error',
            'retry_after': INITIAL_RETRY_DELAY # Generic retry suggestion for unexpected server errors
        }), 500

def get_english_translation_seq2seq(text_input):
    """Translate Halegannada to English using the Seq2Seq model."""
    if not all([seq2seq_model, input_tokenizer, target_tokenizer]):
        print("Seq2Seq model or tokenizers are not available.")
        return None  # Return None to indicate fallback

    try:
        # Tokenize and pad the input text
        input_seq = input_tokenizer.texts_to_sequences([text_input])
        padded_input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')

        # Get model prediction
        prediction = seq2seq_model.predict(padded_input_seq)
        predicted_ids = np.argmax(prediction, axis=-1)

        # Detokenize the predicted sequence
        target_word_index = target_tokenizer.word_index
        reverse_target_word_index = {v: k for k, v in target_word_index.items()}

        translated_words = []
        for idx in predicted_ids[0]:
            if idx > 0:  # Ignore padding
                word = reverse_target_word_index.get(idx)
                if word:
                    if word == '<end>': # Stop at end token
                        break
                    translated_words.append(word)

        return ' '.join(translated_words)

    except Exception as e:
        print(f"Error during Seq2Seq translation: {str(e)}")
        traceback.print_exc()
        return None # Fallback on error

# --- Dedicated endpoints for specific pages ---
@app.route('/translate-seq2seq', methods=['POST'])
def translate_seq2seq():
    """Used by hale-hosa-seq2seq.html.
    - English via local best Seq2Seq (fallback to API/csv).
    - Kannada via db.csv exact-match (fallback to Model1 dictionary).
    """
    try:
        data = request.json or {}
        text_input = data.get('text', '').strip()
        if not text_input:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400

        # Kannada: CSV lookup first
        csv_entry = find_csv_entry_by_hale(text_input)
        if csv_entry and csv_entry.get('hosa'):
            kannada_translation = csv_entry['hosa']
        else:
            # Fallback to Model1 dictionary mapping (word-level)
            words = text_input.split()
            translated_words = [get_kannada_translation_model1(w) for w in words]
            kannada_translation = ' '.join(translated_words).strip() or text_input

        # English: Seq2Seq primary
        english_translation = get_english_translation_seq2seq(text_input)
        if not english_translation:
            # Fallback to CSV English if available
            if csv_entry and csv_entry.get('english'):
                english_translation = csv_entry['english']
            else:
                # Final fallback to API
                english_translation = get_english_translation(text_input)

        return jsonify({
            'kannada_translation': kannada_translation,
            'english_translation': english_translation,
            'status': 'completed'
        })
    except Exception as e:
        error_msg = f"Seq2Seq endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'status': 'error'}), 500

@app.route('/translate-model1', methods=['POST'])
def translate_model1():
    """Used by hale-hosa-kannada.html.
    Returns structured Hale/Hosa/English output.
    - Tries exact match in data/db.json for rich modes (simple/poetic/elevated).
    - Falls back to Model1 word-level mapping when not found.
    """
    try:
        data = request.json or {}
        text_input = data.get('text', '').strip()
        mode = (data.get('mode', 'simple') or 'simple').strip().lower()
        if mode not in ['simple', 'poetic', 'elevated']:
            mode = 'simple'
        if not text_input:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400

        # Try structured dataset first
        entry = find_db_entry_by_hale(text_input)
        if entry:
            # Return the structured entry as-is, plus convenience fields
            hosa_block = entry.get('hosa', {}) or {}
            english_block = entry.get('english_translation', {}) or {}
            selected_hosa = hosa_block.get(mode)
            # Prefer existing english for selected mode if present; otherwise translate
            english_selected = None
            try:
                if mode == 'simple':
                    english_selected = english_block.get('hosa_simple')
                elif mode == 'poetic':
                    english_selected = english_block.get('hosa_poetic')
                elif mode == 'elevated':
                    english_selected = english_block.get('hosa_elevated')
            except Exception:
                english_selected = None
            if not english_selected and selected_hosa:
                english_selected = get_english_translation(selected_hosa)

            return jsonify({
                'hale': entry.get('hale', text_input),
                'hosa': hosa_block,
                'english_translation': english_block,
                'selected_mode': mode,
                'selected_hosa': selected_hosa,
                'selected_english': english_selected,
                'status': 'completed'
            })

        # Fallback: word-level mapping via Model1 data
        words = text_input.split()
        hosa_simple = ' '.join([get_kannada_translation_model1(w) for w in words])
        # If we can't map, just echo input for clarity
        if not hosa_simple.strip():
            hosa_simple = text_input
        # For poetic/elevated, fallback to simple when unavailable
        hosa_poetic = hosa_simple
        hosa_elevated = hosa_simple

        # English translations
        english_hale = get_english_translation(text_input)
        english_hosa_simple = get_english_translation(hosa_simple)
        english_hosa_poetic = english_hosa_simple
        english_hosa_elevated = english_hosa_simple

        # Compute selected outputs
        selected_hosa = {
            'simple': hosa_simple,
            'poetic': hosa_poetic,
            'elevated': hosa_elevated
        }.get(mode, hosa_simple)

        selected_english = {
            'simple': english_hosa_simple,
            'poetic': english_hosa_poetic,
            'elevated': english_hosa_elevated
        }.get(mode, english_hosa_simple)

        return jsonify({
            'hale': text_input,
            'hosa': {
                'simple': hosa_simple,
                'poetic': hosa_poetic,
                'elevated': hosa_elevated
            },
            'english_translation': {
                'hale': english_hale,
                'hosa_simple': english_hosa_simple,
                'hosa_poetic': english_hosa_poetic,
                'hosa_elevated': english_hosa_elevated
            },
            'selected_mode': mode,
            'selected_hosa': selected_hosa,
            'selected_english': selected_english,
            'status': 'completed'
        })
    except Exception as e:
        error_msg = f"Model1 endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'status': 'error'}), 500
    

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    hale_text = data.get("text", "").strip()
    lang = data.get("lang", "kn").split("-")[0]  # 👈 this line fixes "kn-IN" 
    
    if not hale_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Example: Replace with your Halegannada → Hosa Kannada translation function
        translated_text = hale_text

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Generate speech using gTTS (no API key or billing required)
        tts = gTTS(text=translated_text, lang=lang)
        tts.save(filepath)

        # Return the file path as URL
        audio_url = f"/assets/audio/{filename}"
        return jsonify({
            "audio_url": audio_url,
            "translated_text": translated_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve audio files from /assets/audio/
@app.route("/assets/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# Health endpoint (moved off root to avoid overriding index.html)
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


## Removed duplicate app.run to avoid double-starting the server under debug reloader

# Hale-hosa kannada
@app.route('/translate-halegannada', methods=['POST'])
def translate_halegannada():
    try:
        data = request.json
        text_input = data.get('text', '').strip()

        if not text_input:
            return jsonify({'error': 'No text provided', 'status': 'error'}), 400

        print(f"Halegannada translation request received: {text_input}")

        # ---- Kannada Translation ----
        # Example: Split text and map via dictionary or rule-based model
        kannada_words = text_input.split()
        translated_kannada_words = [get_kannada_translation(word) for word in kannada_words]
        kannada_translation = ' '.join(translated_kannada_words)

        # ---- English Translation ----
        english_translation = get_english_translation(text_input)

        print(f"Kannada Translation: {kannada_translation}")
        print(f"English Translation: {english_translation}")

        return jsonify({
            'kannada_translation': kannada_translation,
            'english_translation': english_translation,
            'status': 'completed'
        })

    except Exception as e:
        error_msg = f"Halegannada translation endpoint error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg, 'status': 'error'}), 500


if __name__ == '__main__':
    # Ensure the 'assets' directory exists, if not, create it.
    # This is mainly for the Dictionary.pkl, but good practice.
    if not os.path.exists('assets'):
        os.makedirs('assets')
        print("Created 'assets' directory as it was missing.")
    port = int(os.getenv('PORT', '5001'))
    app.run(debug=True, port=port)

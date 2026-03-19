import os
import re
import random
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------
# MODEL LOADING (loaded once at startup)
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # explicit float32 to avoid quantization support errors
is_render = os.getenv("RENDER") == "true"

print(f"Using device: {device} with dtype: {dtype} (On Render: {is_render})")

processor = None
model = None
feature_extractor = None
tokenizer = None

try:
    if is_render:
        print("Loading lightweight model for Render Free Tier (ViT-GPT2)...")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    else:
        print("Loading BLIP model (Salesforce/blip-image-captioning-base)...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=dtype).to(device)
        
    if model:
        model.eval()
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# -------------------------------------------------------
# POST-PROCESSING: Content & Style Enhancement
# -------------------------------------------------------
def get_emojis_and_tags(text):
    """Retrieve highly relevant emojis and hashtags based on keywords."""
    text_lower = text.lower()
    emojis = []
    tags = []
    
    keyword_map = {
        'sunset': ('🌅', '#Sunset #GoldenHour #SunsetLover'),
        'beach': ('🏖️', '#BeachVibes #OceanLove #Wanderlust'),
        'monument': ('🏰', '#Historical #Architecture #TravelGoals'),
        'nature': ('🌿', '#NatureLover #ExploreMore #Outdoors'),
        'food': ('🍔', '#Foodie #Delicious #Yum'),
        'city': ('🌆', '#CityLife #UrbanVibes #Cityscape'),
        'building': ('🏢', '#Architecture #Urban'),
        'water': ('💧', '#Water #Fresh #Flow'),
        'mountain': ('⛰️', '#MountainView #Hiking #Adventure'),
        'dog': ('🐶', '#DogLover #Cute #Puppy'),
        'cat': ('🐱', '#CatLife #PetLove #Meow'),
        'flower': ('🌸', '#Bloom #Nature #Floral'),
        'sky': ('☁️', '#SkyPorn #Beautiful #Clouds'),
        'car': ('🚗', '#Cars #Drive #Roadtrip'),
        'people': ('👥', '#Community #Life #Moments'),
        'snow': ('❄️', '#Winter #Snow #Cold'),
        'smile': ('😊', '#Happy #Smiles #Joy'),
        'night': ('🌙', '#NightSky #Darkness #NightLife')
    }
    
    for kw, (emoji, tag_str) in keyword_map.items():
        if kw in text_lower:
            emojis.append(emoji)
            tags.extend(tag_str.split())
            
    # Default fallbacks if no strong keywords
    if not tags:
        tags = ['#Photography', '#Vibes', '#Moments']
    if not emojis:
        emojis = ['✨', '📸']
        
    # Shuffle and trim to max limits
    random.shuffle(tags)
    return "".join(list(dict.fromkeys(emojis))[:2]), " ".join(list(dict.fromkeys(tags))[:3])

def generate_social_caption(base_caption, style_idx):
    """Enhance the base caption into a premium social media post format."""
    # Clean up any bad grammar or robotic descriptions
    text = base_caption.strip()
    prefixes = ["a picture of", "an image of", "a photo of", "a photograph of", "an image showing", "a picture showing"]
    lower_text = text.lower()
    for prefix in prefixes:
        if lower_text.startswith(prefix):
            text = text[len(prefix):].strip()
            
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
        
    # High-quality Social media hooks depending on variation
    hooks = [
        ["Truly a sight to behold! 😍", "Can't get enough of this view.", "Such an incredible moment."],
        ["Capturing the pure magic of today ✨", "Lost in the beauty of this.", "A masterpiece of a moment 📸"],
        ["In awe of this stunning perspective.", "Words can't describe this vibe.", "Living for unforgettable moments like this!"]
    ]
    
    emojis, hashtags = get_emojis_and_tags(text)
    hook = random.choice(hooks[style_idx % len(hooks)])
    
    # Assembly
    return f"{hook} {text} {emojis}\n\n{hashtags}"

# -------------------------------------------------------
# MULTI-STRATEGY CAPTION GENERATION
# -------------------------------------------------------
def generate_captions(image):
    """Generate multiple engaging social media captions via rapid batch inference."""
    # Speed Optimization: Aggressive resize to 224x224
    image = image.resize((224, 224))
    
    # Conditional prompt to guide the model purely to descriptive core actions
    prompt = "a photography of"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
    
    with torch.no_grad():
        # num_beams=1 disables heavy beam-search for major speed improvements.
        # temperature=1.1 & top_p=0.95 increases human-like descriptive creativity.
        out = model.generate(
            **inputs,
            max_length=45,
            num_beams=1,
            temperature=1.1,
            repetition_penalty=1.3,
            num_return_sequences=3,
            do_sample=True,
            top_p=0.95
        )
        
    captions = []
    seen = set()
    for i in range(len(out)):
        raw = processor.decode(out[i], skip_special_tokens=True)
        # Strip injected prompt
        if raw.lower().startswith(prompt.lower()):
            raw = raw[len(prompt):].strip()
            
        social_caption = generate_social_caption(raw, i)
        
        # Deduplicate variations
        if social_caption not in seen:
            seen.add(social_caption)
            captions.append(social_caption)
            
    return captions if captions else ["Error generating captions"]

# -------------------------------------------------------
# UTILITY
# -------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------
# API ENDPOINT
# -------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not processor:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Upload JPG, PNG, or GIF.'}), 400
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open and convert
        image = Image.open(filepath).convert('RGB')
        
        # Generate and enhance
        captions = generate_captions(image)
        
        return jsonify({'captions': captions})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

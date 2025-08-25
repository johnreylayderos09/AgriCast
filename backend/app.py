#!/usr/bin/env python3
"""
AgriCast ML API - Fixed Production Version
Optimized for deployment with comprehensive debugging and fallbacks
"""

import os
import sys
import pickle
import requests
import pandas as pd
import numpy as np
import logging
import shutil
import warnings
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress sklearn version warnings for model compatibility
warnings.filterwarnings('ignore', message='.*unpickle estimator.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ===============================
# Logging Configuration
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# Flask App Setup
# ===============================
app = Flask(__name__)
CORS(app, origins="*")

# ===============================
# Configuration
# ===============================
MODEL_URL = "https://github.com/johnreylayderos09/AgriCast/releases/download/rfc/rfc.pkl"
MODEL_PATH = "rfc.pkl"
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
API_VERSION = "2.1.0"

# Global model variables
model = None
label_encoder = None
scaler = None
is_demo_mode = True  # Start in demo mode by default
initialized = False  # Ensure one-time initialization across runtimes

# ===============================
# Crop Data
# ===============================
CROP_EMOJIS = {
    'rice': '🌾', 'wheat': '🌾', 'barley': '🌾', 'jowar': '🌾', 'ragi': '🌾',
    'maize': '🌽', 'corn': '🌽',
    'tomato': '🍅', 'potato': '🥔', 'sweetpotato': '🍠', 'onion': '🧅',
    'garlic': '🧄', 'cabbage': '🥬', 'cauliflower': '🥦', 'cucumber': '🥒',
    'carrot': '🥕', 'pumpkin': '🎃', 'radish': '🥗', 'bittergourd': '🥒',
    'bottlegourd': '🥒', 'brinjal': '🍆',
    'banana': '🍌', 'mango': '🥭', 'papaya': '🥭', 'orange': '🍊',
    'pineapple': '🍍', 'grapes': '🍇', 'jackfruit': '🍈', 'drumstick': '🥦',
    'soybean': '🌱', 'soyabean': '🌱', 'moong': '🌱', 'horsegram': '🌱', 
    'blackgram': '🌱', 'beans': '🌱',
    'cotton': '🌿', 'jute': '🧵', 'rapeseed': '🌻', 'sunflower': '🌻',
    'turmeric': '🟡', 'coriander': '🌿', 'ladyfinger': '🌿',
    'blackpepper': '⚫', 'cardamom': '🟢'
}

FALLBACK_CROPS = [
    'rice', 'maize', 'moong', 'wheat', 'rapeseed', 'potato', 'jowar', 'onion',
    'sunflower', 'barley', 'cotton', 'sweetpotato', 'ragi', 'horsegram',
    'turmeric', 'banana', 'coriander', 'soybean', 'garlic', 'jute',
    'blackpepper', 'mango', 'tomato', 'papaya', 'brinjal', 'cardamom',
    'ladyfinger', 'orange', 'cabbage', 'pineapple', 'cauliflower',
    'cucumber', 'grapes', 'jackfruit', 'drumstick', 'bottlegourd',
    'radish', 'blackgram', 'bittergourd', 'pumpkin'
]

# ===============================
# Utility Functions
# ===============================
def get_disk_space():
    """Get available disk space in MB"""
    try:
        total, used, free = shutil.disk_usage("/")
        return {
            'total_gb': float(round(total / (1024**3), 2)),
            'free_gb': float(round(free / (1024**3), 2)),
            'used_gb': float(round(used / (1024**3), 2))
        }
    except Exception as e:
        logger.warning(f"Cannot get disk space: {e}")
        return None

def validate_prediction_input(data):
    """Enhanced input validation with range checking"""
    errors = []
    
    # Define valid ranges for agricultural parameters
    ranges = {
        'N': (0, 500, 'Nitrogen content (0-500 kg/ha)'),
        'P': (0, 150, 'Phosphorus content (0-150 kg/ha)'),
        'K': (0, 500, 'Potassium content (0-500 kg/ha)'),
        'pH': (3.5, 10.0, 'Soil pH (3.5-10.0)'),
        'temperature': (-10, 50, 'Temperature (-10°C to 50°C)'),
        'rainfall': (0, 5000, 'Annual rainfall (0-5000mm)')
    }
    
    for field, (min_val, max_val, desc) in ranges.items():
        if field not in data:
            errors.append(f'Missing required field: {field} ({desc})')
            continue
        
        try:
            value = float(data[field])
            if not (min_val <= value <= max_val):
                errors.append(f'{field} value {value} out of range. Expected: {desc}')
        except (ValueError, TypeError):
            errors.append(f'Invalid {field} value: must be a number')
    
    return errors

def get_suitability_level(probability):
    """Convert probability to suitability level"""
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.4:
        return 'Medium'
    else:
        return 'Low'

def normalize_crop_label(raw_label):
    """Convert a model label (string or index) to a human-readable crop name.

    Tries the following in order:
    1) If a `LabelEncoder` is available, use `inverse_transform` for integer labels
    2) If numeric and within range of `FALLBACK_CROPS`, use that index
    3) Otherwise, return the label as string
    """
    try:
        # Already a clear string crop name
        if isinstance(raw_label, str) and not raw_label.isdigit():
            return raw_label

        # Handle numpy scalar types and numbers represented as strings
        try:
            numeric_index = int(raw_label)
        except (ValueError, TypeError):
            # Not a pure numeric label; return as string
            return str(raw_label)

        # Prefer label encoder mapping when available
        if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
            try:
                decoded = label_encoder.inverse_transform([numeric_index])[0]
                return str(decoded)
            except Exception:
                pass

        # Fallback: index into FALLBACK_CROPS when index is valid
        if 0 <= numeric_index < len(FALLBACK_CROPS):
            return FALLBACK_CROPS[numeric_index]

        # Last resort
        return str(raw_label)
    except Exception:
        return str(raw_label)

def generate_farming_advice(input_data, predicted_crop):
    """Generate basic farming advice based on input parameters"""
    advice = []
    
    try:
        # pH advice
        ph = float(input_data.get('pH', 7))
        if ph < 6:
            advice.append("Consider adding lime to increase soil pH for better nutrient uptake")
        elif ph > 8:
            advice.append("Consider adding sulfur or organic matter to decrease soil pH")
        
        # Rainfall advice
        rainfall = float(input_data.get('rainfall', 1000))
        if rainfall < 500:
            advice.append("Low rainfall detected - ensure adequate irrigation system")
        elif rainfall > 2000:
            advice.append("High rainfall area - ensure proper drainage to prevent waterlogging")
        
        # Temperature advice
        temp = float(input_data.get('temperature', 25))
        if temp < 15:
            advice.append("Cool climate - consider cold-resistant crop varieties")
        elif temp > 35:
            advice.append("Hot climate - ensure adequate water supply and shade if possible")
        
        # Nutrient advice
        n_level = float(input_data.get('N', 100))
        p_level = float(input_data.get('P', 50))
        k_level = float(input_data.get('K', 100))
        
        if n_level < 50:
            advice.append("Low nitrogen levels - consider nitrogen-rich fertilizers or legume rotation")
        if p_level < 20:
            advice.append("Low phosphorus levels - add phosphate fertilizers for better root development")
        if k_level < 50:
            advice.append("Low potassium levels - add potash fertilizers for disease resistance")
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Error generating farming advice: {e}")
        advice = [f"Soil conditions appear suitable for {predicted_crop} cultivation"]
    
    return advice[:4] if advice else [f"Soil conditions appear suitable for {predicted_crop} cultivation"]

# ===============================
# Model Management Functions
# ===============================
def check_internet_connectivity():
    """Check if internet is available"""
    try:
        response = requests.get("https://httpbin.org/get", timeout=5)
        return response.status_code == 200
    except:
        return False

def download_model():
    """Download model with better error handling"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"📄 Model file exists ({file_size} bytes)")
        
        # Validate existing model
        if validate_existing_model():
            logger.info("✅ Existing model file is valid")
            return True
        else:
            logger.warning("⚠️ Existing model corrupted, attempting to re-download...")
            try:
                os.remove(MODEL_PATH)
            except:
                pass
    
    if not check_internet_connectivity():
        logger.error("❌ No internet connection available")
        return False
    
    logger.info("📥 Downloading model from GitHub...")
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"📊 Download size: {total_size} bytes ({total_size/1024/1024:.2f} MB)")
        
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress update every 1MB
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"📥 Download progress: {progress:.1f}%")
        
        logger.info(f"✅ Model downloaded successfully! ({downloaded} bytes)")
        
        # Validate downloaded file
        if validate_existing_model():
            logger.info("✅ Downloaded model validated")
            return True
        else:
            logger.error("❌ Downloaded model corrupted")
            try:
                os.remove(MODEL_PATH)
            except:
                pass
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ Download timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Connection error")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def validate_existing_model():
    """Validate that the model file is not corrupted"""
    try:
        with open(MODEL_PATH, "rb") as f:
            test_model = pickle.load(f)
        return hasattr(test_model, 'predict')
    except Exception as e:
        logger.warning(f"Model validation failed: {e}")
        return False

def create_fallback_preprocessing():
    """Create or load preprocessing objects"""
    global label_encoder, scaler
    
    try:
        # Try to load label encoder from file first
        if label_encoder is None:
            label_encoder_path = "label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                try:
                    logger.info("📂 Loading label encoder from file...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(label_encoder_path, "rb") as f:
                            label_encoder = pickle.load(f)
                    logger.info("✅ Label encoder loaded from file")
                except Exception as e:
                    logger.warning(f"Failed to load label encoder from file: {e}")
                    label_encoder = None
            
            # Fallback: create new label encoder
            if label_encoder is None:
                logger.info("🔧 Creating fallback label encoder...")
                label_encoder = LabelEncoder()
                label_encoder.fit(FALLBACK_CROPS)
        
        # Try to load scaler from file first
        if scaler is None:
            scaler_path = "scaler.pkl"
            if os.path.exists(scaler_path):
                try:
                    logger.info("📂 Loading scaler from file...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                    logger.info("✅ Scaler loaded from file")
                except Exception as e:
                    logger.warning(f"Failed to load scaler from file: {e}")
                    scaler = None
            
            # Fallback: create new scaler
            if scaler is None:
                logger.info("🔧 Creating fallback scaler...")
                scaler = StandardScaler()
                # Agricultural parameter ranges for scaling
                dummy_data = np.array([
                    [0, 0, 0, 3.5, -10, 0],          # Min values
                    [500, 150, 500, 10.0, 50, 5000] # Max values
                ])
                scaler.fit(dummy_data)
            
        return True
    except Exception as e:
        logger.error(f"Failed to create preprocessing objects: {e}")
        return False

def setup_demo_mode():
    """Setup demo mode with mock model"""
    global model, is_demo_mode
    
    logger.info("🎭 Setting up demo mode...")
    
    class MockModel:
        def __init__(self):
            self.classes_ = np.array(FALLBACK_CROPS)
        
        def predict(self, X):
            # Rule-based prediction for demo
            try:
                N, P, K, pH, temp, rainfall = X[0]
                
                # Simple decision tree logic
                if rainfall > 1200:
                    if temp > 25:
                        return ['rice']
                    else:
                        return ['wheat']
                elif temp > 30:
                    if pH > 7:
                        return ['cotton']
                    else:
                        return ['maize']
                elif pH < 6:
                    return ['potato']
                elif N > 100:
                    return ['wheat']
                elif rainfall < 600:
                    return ['barley']
                else:
                    return ['maize']
            except:
                return ['maize']  # Safe fallback
        
        def predict_proba(self, X):
            try:
                # Generate realistic probabilities
                n_classes = len(self.classes_)
                # Create a realistic distribution
                np.random.seed(42)  # For consistent results
                probs = np.random.dirichlet(np.ones(n_classes) * 0.1, size=1)
                # Boost the predicted class probability
                predicted_class = self.predict(X)[0]
                if predicted_class in self.classes_:
                    predicted_idx = list(self.classes_).index(predicted_class)
                    probs[0][predicted_idx] = max(0.4, probs[0][predicted_idx])
                    # Renormalize
                    probs = probs / probs.sum()
                return probs
            except:
                # Ultimate fallback
                probs = np.zeros((1, len(self.classes_)))
                probs[0][0] = 1.0  # Give 100% to first crop
                return probs
    
    try:
        model = MockModel()
        create_fallback_preprocessing()
        is_demo_mode = True
        
        # Test the demo model
        test_data = np.array([[100, 50, 200, 6.5, 25, 800]])
        test_pred = model.predict(test_data)
        logger.info(f"✅ Demo mode test successful: {test_pred[0]}")
        
        logger.info("✅ Demo mode activated!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to setup demo mode: {e}")
        return False

def load_model():
    """Load model with comprehensive fallbacks"""
    global model, label_encoder, scaler, is_demo_mode
    
    logger.info("🚀 Starting model loading...")
    
    try:
        # Try to download and load the real model first
        if download_model():
            logger.info("📂 Loading model...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings during model loading
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
            
            if not hasattr(model, 'predict'):
                raise ValueError("Invalid model: missing predict method")
            
            logger.info(f"✅ Model loaded: {type(model).__name__}")
            
            # Load preprocessing objects
            if not create_fallback_preprocessing():
                raise ValueError("Failed to create preprocessing objects")
            
            # Test model
            test_data = np.array([[100, 50, 200, 6.5, 25, 800]])
            if scaler:
                test_data = scaler.transform(test_data)
            test_pred = model.predict(test_data)
            logger.info(f"✅ Model test successful: {test_pred[0]}")
            
            is_demo_mode = False
            return True
        else:
            raise ValueError("Model download failed")
            
    except Exception as e:
        logger.warning(f"⚠️ Real model loading failed: {e}")
        logger.info("🔄 Switching to demo mode...")
        result = setup_demo_mode()
        
        # Debug logging for demo model
        if model is not None:
            logger.info(f"✅ Demo model created successfully: {type(model).__name__}")
            logger.info(f"✅ Demo model has predict method: {hasattr(model, 'predict')}")
            logger.info(f"✅ Demo model has classes: {hasattr(model, 'classes_')}")
            if hasattr(model, 'classes_'):
                logger.info(f"✅ Demo model classes count: {len(model.classes_)}")
        else:
            logger.error("❌ Demo model is None after setup")
        
        return result

# ===============================
# API Routes
# ===============================

@app.route("/", methods=['GET'])
def home():
    """Root endpoint with API information"""
    # Check if model is properly loaded (either real model or demo model)
    model_is_loaded = model is not None and hasattr(model, 'predict')
    
    return jsonify({
        "message": "🌾 AgriCast ML API",
        "version": API_VERSION,
        "status": "healthy",
        "model_status": "loaded" if model_is_loaded else "not_loaded",
        "model_loaded": model_is_loaded,
        "demo_mode": is_demo_mode,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/": "API information",
            "/health": "Comprehensive health check",
            "/predict": "Crop prediction (POST)",
            "/weather": "Weather data (GET with lat/lon)",
            "/geocoding": "Location services (GET)",
            "/docs": "API documentation"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "N": 90,
                "P": 42,
                "K": 43,
                "pH": 6.5,
                "temperature": 25.5,
                "rainfall": 202.9
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    model_is_loaded = model is not None and hasattr(model, 'predict')
    
    model_info = {}
    if model_is_loaded:
        model_info = {
            'type': type(model).__name__,
            'is_demo': is_demo_mode,
            'has_classes': hasattr(model, 'classes_'),
            'has_predict_proba': hasattr(model, 'predict_proba')
        }
        
        if hasattr(model, 'classes_'):
            # Convert numpy arrays/types to JSON-serializable Python types
            model_info.update({
                'total_classes': int(len(model.classes_)),
                'sample_classes': [str(cls) for cls in model.classes_[:10]]
            })
    
    return jsonify({
        'status': 'healthy',
        'message': '🌾 AgriCast ML API',
        'version': API_VERSION,
        'timestamp': datetime.now().isoformat(),
        'model': {
            'loaded': model_is_loaded,
            'demo_mode': is_demo_mode,
            'info': model_info
        },
        'preprocessing': {
            'label_encoder': {
                'loaded': label_encoder is not None,
                'classes': [str(cls) for cls in label_encoder.classes_] if label_encoder is not None and hasattr(label_encoder, 'classes_') else []
            },
            'scaler': {
                'loaded': scaler is not None,
                'fitted': scaler is not None and hasattr(scaler, 'scale_')
            }
        },
        'services': {
            'weather_api': OPENWEATHER_API_KEY is not None,
            'geocoding_api': OPENWEATHER_API_KEY is not None
        },
        'system': {
            'python_version': sys.version.split()[0],
            'working_directory': os.getcwd(),
            'disk_space': get_disk_space(),
            'environment': os.environ.get('RENDER', 'local')
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Crop prediction endpoint"""
    model_is_loaded = model is not None and hasattr(model, 'predict')
    
    if not model_is_loaded:
        return jsonify({
            'success': False,
            'error': 'ML model not available',
            'suggestion': 'Check /health endpoint for details'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False, 
                'error': 'No input data provided'
            }), 400
        
        # Handle different input formats
        if 'features' in data:
            # Array format: [N, P, K, pH, temperature, rainfall]
            features = data['features']
            if len(features) != 6:
                return jsonify({
                    'success': False,
                    'error': 'Features array must have exactly 6 values: [N, P, K, pH, temperature, rainfall]'
                }), 400
            
            input_params = dict(zip(['N', 'P', 'K', 'pH', 'temperature', 'rainfall'], features))
        else:
            # Named parameters format
            validation_errors = validate_prediction_input(data)
            if validation_errors:
                return jsonify({
                    'success': False,
                    'error': 'Input validation failed',
                    'details': validation_errors
                }), 400
            
            input_params = {k: data[k] for k in ['N', 'P', 'K', 'pH', 'temperature', 'rainfall']}
        
        # Prepare input for model
        features = [float(input_params[k]) for k in ['N', 'P', 'K', 'pH', 'temperature', 'rainfall']]
        input_data = np.array([features])
        
        # Scale if available
        if scaler is not None:
            try:
                input_data = scaler.transform(input_data)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}, using raw data")
        
        # Make prediction
        predicted_class = model.predict(input_data)[0]
        
        # Get probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_data)[0]
            except Exception as e:
                logger.warning(f"Probability prediction failed: {e}")
        
        # Prepare response
        response = prepare_prediction_response(
            predicted_class, probabilities, input_params
        )
        
        logger.info(f"Prediction: {predicted_class} (demo: {is_demo_mode})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

def prepare_prediction_response(predicted_class, probabilities, input_data):
    """Prepare comprehensive prediction response"""
    try:
        # Get class labels
        if hasattr(model, 'classes_'):
            class_labels = model.classes_
        else:
            class_labels = FALLBACK_CROPS
        
        # Prepare recommendations
        recommendations = []
        confidence = 85.0
        
        if probabilities is not None and len(probabilities) == len(class_labels):
            # Sort by probability
            crop_probs = list(zip(class_labels, probabilities))
            crop_probs.sort(key=lambda x: x[1], reverse=True)
            
            for crop, prob in crop_probs[:5]:  # Top 5
                crop_name = str(normalize_crop_label(crop)).lower()
                emoji = CROP_EMOJIS.get(crop_name, '🌱')
                
                recommendations.append({
                    'crop': crop_name.capitalize(),
                    'emoji': emoji,
                    'probability': round(float(prob) * 100, 1),
                    'suitability': get_suitability_level(float(prob))
                })
            
            confidence = round(float(np.max(probabilities)) * 100, 1)
        else:
            # Single prediction fallback
            crop_name = str(normalize_crop_label(predicted_class)).lower()
            emoji = CROP_EMOJIS.get(crop_name, '🌱')
            recommendations = [{
                'crop': crop_name.capitalize(),
                'emoji': emoji,
                'probability': 85.0,  # Default confidence
                'suitability': 'High'
            }]
        
        return {
            'success': True,
            'predicted_crop': str(normalize_crop_label(predicted_class)).capitalize(),
            'confidence': confidence,
            'recommendations': recommendations,
            'input_data': input_data,
            'farming_advice': generate_farming_advice(input_data, predicted_class),
            'model_info': {
                'demo_mode': is_demo_mode,
                'model_type': type(model).__name__
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error preparing prediction response: {e}")
        # Ultimate fallback
        return {
            'success': True,
            'predicted_crop': 'Maize',
            'confidence': 75.0,
            'recommendations': [{
                'crop': 'Maize',
                'emoji': '🌽',
                'probability': 75.0,
                'suitability': 'Medium'
            }],
            'input_data': input_data,
            'farming_advice': ['Consider soil testing for optimal results'],
            'model_info': {
                'demo_mode': True,
                'model_type': 'Fallback'
            },
            'timestamp': datetime.now().isoformat()
        }

@app.route('/weather', methods=['GET'])
def get_weather():
    """Weather data endpoint"""
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not lat or not lon:
        return jsonify({
            'success': False,
            'error': 'Missing latitude or longitude parameters'
        }), 400
    
    if not OPENWEATHER_API_KEY:
        return jsonify({
            'success': False,
            'error': 'Weather service not configured'
        }), 503
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        weather_data = response.json()
        
        # Extract useful information
        processed_data = {
            'temperature': weather_data.get('main', {}).get('temp'),
            'humidity': weather_data.get('main', {}).get('humidity'),
            'pressure': weather_data.get('main', {}).get('pressure'),
            'weather': weather_data.get('weather', [{}])[0].get('description'),
            'wind_speed': weather_data.get('wind', {}).get('speed'),
            'location': weather_data.get('name'),
            'country': weather_data.get('sys', {}).get('country')
        }
        
        return jsonify({
            'success': True,
            'data': weather_data,
            'processed': processed_data
        })
        
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': 'Weather service timeout'
        }), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch weather data'
        }), 500

@app.route('/geocoding', methods=['GET'])
def get_geocoding():
    """Geocoding endpoint"""
    q = request.args.get('q')
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    geocode_type = request.args.get('type', 'forward')
    limit = min(int(request.args.get('limit', '5')), 10)  # Max 10 results
    
    if not OPENWEATHER_API_KEY:
        return jsonify({
            'success': False,
            'error': 'Geocoding service not configured'
        }), 503
    
    try:
        base_url = "https://api.openweathermap.org/geo/1.0"
        
        if geocode_type == 'reverse' and lat and lon:
            url = f"{base_url}/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'limit': limit,
                'appid': OPENWEATHER_API_KEY
            }
        elif q:
            url = f"{base_url}/direct"
            params = {
                'q': q,
                'limit': limit,
                'appid': OPENWEATHER_API_KEY
            }
        else:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        geocoding_data = response.json()
        return jsonify({
            'success': True,
            'data': geocoding_data
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Geocoding API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch location data'
        }), 500

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        "title": "AgriCast ML API Documentation",
        "version": API_VERSION,
        "description": "Machine Learning API for crop prediction based on soil and climate parameters",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "API information and status"
            },
            "/health": {
                "method": "GET",
                "description": "Comprehensive health check with system information"
            },
            "/predict": {
                "method": "POST",
                "description": "Predict suitable crops based on soil and climate data",
                "parameters": {
                    "N": "Nitrogen content (0-500 kg/ha)",
                    "P": "Phosphorus content (0-150 kg/ha)", 
                    "K": "Potassium content (0-500 kg/ha)",
                    "pH": "Soil pH (3.5-10.0)",
                    "temperature": "Average temperature (-10 to 50°C)",
                    "rainfall": "Annual rainfall (0-5000mm)"
                },
                "example": {
                    "N": 90, "P": 42, "K": 43,
                    "pH": 6.502, "temperature": 20.879, "rainfall": 202.9
                }
            },
            "/weather": {
                "method": "GET", 
                "description": "Get current weather data",
                "parameters": {
                    "lat": "Latitude",
                    "lon": "Longitude"
                }
            },
            "/geocoding": {
                "method": "GET",
                "description": "Convert between location names and coordinates",
                "parameters": {
                    "q": "Location query (for forward geocoding)",
                    "lat": "Latitude (for reverse geocoding)",
                    "lon": "Longitude (for reverse geocoding)",
                    "type": "forward or reverse (default: forward)",
                    "limit": "Max results (default: 5, max: 10)"
                }
            }
        },
        "supported_crops": sorted(FALLBACK_CROPS),
        "response_format": {
            "prediction": {
                "success": True,
                "predicted_crop": "Rice",
                "confidence": 87.5,
                "recommendations": [
                    {
                        "crop": "Rice",
                        "emoji": "🌾", 
                        "probability": 87.5,
                        "suitability": "High"
                    }
                ],
                "farming_advice": ["Helpful farming tips"],
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    })

# ===============================
# Error Handlers
# ===============================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'suggestion': 'Check /docs for available endpoints'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'suggestion': 'Please try again or contact support'
    }), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        'success': False,
        'error': 'Service temporarily unavailable',
        'suggestion': 'Check /health for service status'
    }), 503

# ===============================
# Additional Routes for Deployment
# ===============================

# Health check for Render deployment
@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for health checks"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    }), 200

# CORS preflight handling
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Request logging middleware
@app.after_request
def after_request(response):
    # Log requests (avoid logging sensitive data)
    if not request.path.startswith('/ping'):
        logger.info(f"{request.method} {request.path} - {response.status_code}")
    
    # Add security headers
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    
    return response

# ===============================
# Startup and Main
# ===============================

def ensure_initialized_once():
    """Run app initialization exactly once per process."""
    global initialized
    if initialized:
        return True
    try:
        result = initialize_app()
        if result:
            initialized = True
        return result
    except Exception:
        # Do not mark initialized so a later retry can occur
        raise

def initialize_app():
    """Initialize the application with proper error handling"""
    logger.info("🚀 Initializing AgriCast ML API...")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check environment
    if os.environ.get('RENDER'):
        logger.info("🌐 Running on Render platform")
    else:
        logger.info("💻 Running locally")
    
    # Check required environment variables
    if OPENWEATHER_API_KEY:
        logger.info("✅ Weather API key configured")
    else:
        logger.warning("⚠️  Weather API key not configured - weather services will be unavailable")
    
    # Initialize model - ensure we always have a working model
    try:
        if load_model():
            # Verify the model is actually working
            if model is not None and hasattr(model, 'predict'):
                if is_demo_mode:
                    logger.info("✅ API initialized in DEMO mode")
                else:
                    logger.info("✅ API initialized with ML model")
                return True
            else:
                logger.warning("⚠️  Model loaded but not functional, switching to demo mode")
                return setup_demo_mode()
        else:
            logger.error("❌ Failed to initialize model")
            return setup_demo_mode()
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        logger.info("🔄 Attempting emergency demo mode setup...")
        return setup_demo_mode()

# Initialize at import time for WSGI servers (e.g., Gunicorn on Render)
# Flask 3.x removed before_first_request; import-time init ensures readiness
try:
    ensure_initialized_once()
except Exception as e:
    logger.error(f"Initialization during import failed: {e}")

if __name__ == "__main__":
    # Initialize the application
    if not ensure_initialized_once():
        logger.error("❌ Application initialization failed")
        # Try one more time with just demo mode
        logger.info("🆘 Emergency startup - demo mode only...")
        if not setup_demo_mode():
            logger.error("❌ Complete initialization failure")
            sys.exit(1)
    
    # Get configuration from environment
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🎭 Demo mode: {is_demo_mode}")
    
    try:
        # Run the Flask app
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True  # Enable threading for better performance
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

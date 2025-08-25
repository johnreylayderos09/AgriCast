// api/predict.js - Vercel serverless function for crop prediction
// Uses OpenAI GPT for generative AI predictions when ML model is unavailable

const CROP_DATABASE = {
  rice:       { ph: [5.5, 6.5], temp: [20, 30], rainfall: [1000, 2000], N: [100, 200], P: [50, 100], K: [100, 200], emoji: '🌾' },
  wheat:      { ph: [6.0, 7.5], temp: [15, 25], rainfall: [400, 800], N: [120, 180], P: [40, 80], K: [80, 150], emoji: '🌾' },
  barley:     { ph: [6.0, 7.5], temp: [12, 25], rainfall: [300, 700], N: [100, 150], P: [40, 70], K: [70, 120], emoji: '🌾' },
  jowar:      { ph: [5.0, 8.0], temp: [25, 35], rainfall: [400, 1000], N: [80, 150], P: [40, 70], K: [60, 120], emoji: '🌾' },
  ragi:       { ph: [4.5, 7.5], temp: [20, 30], rainfall: [700, 1200], N: [60, 120], P: [40, 60], K: [50, 100], emoji: '🌾' },
  maize:      { ph: [6.0, 7.0], temp: [20, 30], rainfall: [600, 1200], N: [150, 250], P: [60, 100], K: [120, 200], emoji: '🌽' },
  corn:       { ph: [6.0, 7.0], temp: [20, 30], rainfall: [600, 1200], N: [150, 250], P: [60, 100], K: [120, 200], emoji: '🌽' },
  tomato:     { ph: [6.0, 6.8], temp: [18, 26], rainfall: [400, 800], N: [100, 150], P: [80, 120], K: [150, 250], emoji: '🍅' },
  potato:     { ph: [5.0, 6.5], temp: [15, 25], rainfall: [500, 800], N: [120, 180], P: [60, 100], K: [200, 300], emoji: '🥔' },
  sweetpotato:{ ph: [5.5, 6.5], temp: [20, 30], rainfall: [750, 1500], N: [80, 120], P: [40, 70], K: [100, 200], emoji: '🍠' },
  onion:      { ph: [6.0, 7.0], temp: [15, 25], rainfall: [350, 600], N: [100, 150], P: [50, 80], K: [120, 180], emoji: '🧅' },
  garlic:     { ph: [6.0, 7.5], temp: [12, 24], rainfall: [300, 600], N: [80, 120], P: [50, 80], K: [100, 150], emoji: '🧄' },
  cabbage:    { ph: [6.0, 6.8], temp: [15, 20], rainfall: [400, 700], N: [120, 200], P: [80, 120], K: [150, 200], emoji: '🥬' },
  cauliflower:{ ph: [6.0, 7.0], temp: [15, 20], rainfall: [400, 700], N: [100, 150], P: [80, 120], K: [150, 200], emoji: '🥦' },
  cucumber:   { ph: [5.5, 7.0], temp: [18, 27], rainfall: [600, 1000], N: [80, 120], P: [60, 100], K: [100, 150], emoji: '🥒' },
  carrot:     { ph: [6.0, 7.0], temp: [16, 21], rainfall: [400, 600], N: [80, 120], P: [60, 100], K: [100, 150], emoji: '🥕' },
  pumpkin:    { ph: [5.5, 7.5], temp: [20, 27], rainfall: [500, 1000], N: [80, 120], P: [40, 70], K: [100, 150], emoji: '🎃' },
  radish:     { ph: [6.0, 7.5], temp: [10, 20], rainfall: [350, 600], N: [60, 100], P: [40, 60], K: [80, 120], emoji: '🥗' },
  bittergourd:{ ph: [5.5, 6.7], temp: [18, 30], rainfall: [800, 1500], N: [60, 100], P: [40, 70], K: [80, 120], emoji: '🥒' },
  bottlegourd:{ ph: [5.5, 6.7], temp: [20, 30], rainfall: [700, 1200], N: [60, 100], P: [40, 70], K: [80, 120], emoji: '🥒' },
  brinjal:    { ph: [5.5, 6.6], temp: [20, 30], rainfall: [600, 1000], N: [100, 150], P: [50, 80], K: [120, 180], emoji: '🍆' },
  banana:     { ph: [5.5, 7.0], temp: [26, 30], rainfall: [1200, 2500], N: [200, 300], P: [50, 100], K: [300, 600], emoji: '🍌' },
  mango:      { ph: [5.5, 7.5], temp: [24, 30], rainfall: [750, 2500], N: [100, 200], P: [50, 100], K: [100, 300], emoji: '🥭' },
  papaya:     { ph: [6.0, 7.0], temp: [22, 30], rainfall: [1500, 2500], N: [150, 250], P: [60, 100], K: [200, 300], emoji: '🥭' },
  orange:     { ph: [5.5, 7.0], temp: [20, 30], rainfall: [1000, 1500], N: [80, 120], P: [40, 80], K: [100, 150], emoji: '🍊' },
  pineapple:  { ph: [4.5, 6.5], temp: [20, 30], rainfall: [1000, 1500], N: [120, 180], P: [40, 80], K: [150, 200], emoji: '🍍' },
  grapes:     { ph: [5.5, 7.0], temp: [15, 30], rainfall: [600, 900], N: [60, 120], P: [40, 80], K: [80, 150], emoji: '🍇' },
  jackfruit:  { ph: [6.0, 7.5], temp: [24, 30], rainfall: [1500, 2500], N: [80, 150], P: [40, 70], K: [80, 150], emoji: '🍈' },
  drumstick:  { ph: [6.0, 7.0], temp: [25, 35], rainfall: [250, 1500], N: [60, 100], P: [40, 70], K: [60, 120], emoji: '🥦' },
  soybean:    { ph: [6.0, 7.0], temp: [20, 30], rainfall: [450, 700], N: [60, 100], P: [40, 80], K: [80, 120], emoji: '🌱' },
  soyabean:   { ph: [6.0, 7.0], temp: [20, 30], rainfall: [450, 700], N: [60, 100], P: [40, 80], K: [80, 120], emoji: '🌱' },
  moong:      { ph: [6.0, 7.5], temp: [25, 35], rainfall: [600, 1000], N: [40, 80], P: [30, 60], K: [40, 80], emoji: '🌱' },
  horsegram:  { ph: [5.5, 7.0], temp: [20, 30], rainfall: [400, 700], N: [30, 60], P: [20, 50], K: [40, 70], emoji: '🌱' },
  blackgram:  { ph: [6.0, 7.5], temp: [20, 30], rainfall: [600, 1000], N: [40, 80], P: [30, 60], K: [40, 80], emoji: '🌱' },
  beans:      { ph: [6.0, 7.5], temp: [15, 25], rainfall: [600, 1000], N: [60, 100], P: [40, 70], K: [80, 120], emoji: '🌱' },
  cotton:     { ph: [5.8, 8.0], temp: [21, 30], rainfall: [500, 1200], N: [120, 200], P: [50, 90], K: [100, 180], emoji: '🌿' },
  jute:       { ph: [6.0, 7.5], temp: [24, 37], rainfall: [1500, 2500], N: [80, 120], P: [40, 70], K: [80, 150], emoji: '🧵' },
  rapeseed:   { ph: [5.5, 7.5], temp: [10, 25], rainfall: [400, 800], N: [80, 120], P: [40, 70], K: [60, 100], emoji: '🌻' },
  sunflower:  { ph: [6.0, 7.5], temp: [20, 25], rainfall: [400, 650], N: [100, 150], P: [60, 100], K: [120, 180], emoji: '🌻' },
  turmeric:   { ph: [5.0, 7.5], temp: [20, 30], rainfall: [1000, 2000], N: [150, 200], P: [60, 100], K: [120, 200], emoji: '🟡' },
  coriander:  { ph: [6.0, 7.0], temp: [15, 25], rainfall: [500, 700], N: [40, 80], P: [30, 60], K: [40, 70], emoji: '🌿' },
  ladyfinger: { ph: [6.0, 7.0], temp: [20, 30], rainfall: [600, 1000], N: [60, 100], P: [40, 70], K: [60, 120], emoji: '🌿' },
  blackpepper:{ ph: [5.5, 6.5], temp: [23, 32], rainfall: [2000, 3000], N: [80, 120], P: [40, 70], K: [80, 120], emoji: '⚫' },
  cardamom:   { ph: [5.0, 6.8], temp: [15, 30], rainfall: [1500, 2500], N: [100, 150], P: [60, 100], K: [100, 150], emoji: '🟢' }
};

const CROP_EMOJIS = {
  rice: '🌾', wheat: '🌾', barley: '🌾', jowar: '🌾', ragi: '🌾',
  maize: '🌽', corn: '🌽',
  tomato: '🍅', potato: '🥔', sweetpotato: '🍠', onion: '🧅',
  garlic: '🧄', cabbage: '🥬', cauliflower: '🥦', cucumber: '🥒',
  carrot: '🥕', pumpkin: '🎃', radish: '🥗', bittergourd: '🥒',
  bottlegourd: '🥒', brinjal: '🍆',
  banana: '🍌', mango: '🥭', papaya: '🥭', orange: '🍊',
  pineapple: '🍍', grapes: '🍇', jackfruit: '🍈', drumstick: '🥦',
  soybean: '🌱', soyabean: '🌱', moong: '🌱', horsegram: '🌱', 
  blackgram: '🌱', beans: '🌱',
  cotton: '🌿', jute: '🧵', rapeseed: '🌻', sunflower: '🌻',
  turmeric: '🟡', coriander: '🌿', ladyfinger: '🌿',
  blackpepper: '⚫', cardamom: '🟢',
  default: '🌱'
};

function isInRange(value, [min, max]) {
  return value >= min && value <= max;
}

function validateInput(data) {
  const errors = [];
  const ranges = {
    N: [0, 500, 'Nitrogen (0-500 kg/ha)'],
    P: [0, 150, 'Phosphorus (0-150 kg/ha)'],
    K: [0, 500, 'Potassium (0-500 kg/ha)'],
    pH: [3.5, 10.0, 'pH (3.5-10.0)'],
    temperature: [-10, 50, 'Temperature (-10°C to 50°C)'],
    rainfall: [0, 5000, 'Rainfall (0-5000mm)']
  };

  for (const [field, [min, max, desc]] of Object.entries(ranges)) {
    if (!(field in data)) {
      errors.push(`Missing required field: ${field} (${desc})`);
      continue;
    }
    
    const value = parseFloat(data[field]);
    if (isNaN(value) || value < min || value > max) {
      errors.push(`Invalid ${field}: ${data[field]}. Expected: ${desc}`);
    }
  }

  return errors;
}

function calculateSoilHealth(inputs) {
  let health = 0;
  
  // pH health (ideal: 6.0-7.0)
  if (inputs.pH >= 6.0 && inputs.pH <= 7.0) health += 25;
  else if (inputs.pH >= 5.5 && inputs.pH <= 7.5) health += 20;
  else health += 10;
  
  // Nutrient levels
  health += Math.min(25, (inputs.N / 200) * 25);
  health += Math.min(25, (inputs.P / 100) * 25); 
  health += Math.min(25, (inputs.K / 300) * 25);
  
  return Math.round(health);
}

function generateFarmingAdvice(inputs, predictedCrop) {
  const advice = [];
  
  try {
    const { pH, rainfall, temperature, N, P, K } = inputs;
    
    if (pH < 6) {
      advice.push("Consider adding lime to increase soil pH for better nutrient uptake");
    } else if (pH > 8) {
      advice.push("Consider adding sulfur or organic matter to decrease soil pH");
    }
    
    if (rainfall < 500) {
      advice.push("Low rainfall detected - ensure adequate irrigation system");
    } else if (rainfall > 2000) {
      advice.push("High rainfall area - ensure proper drainage to prevent waterlogging");
    }
    
    if (temperature < 15) {
      advice.push("Cool climate - consider cold-resistant crop varieties");
    } else if (temperature > 35) {
      advice.push("Hot climate - ensure adequate water supply and shade if possible");
    }
    
    if (N < 50) {
      advice.push("Low nitrogen levels - consider nitrogen-rich fertilizers or legume rotation");
    }
    if (P < 20) {
      advice.push("Low phosphorus levels - add phosphate fertilizers for better root development");
    }
    if (K < 50) {
      advice.push("Low potassium levels - add potash fertilizers for disease resistance");
    }

    // Crop-specific advice
    if (predictedCrop.toLowerCase() === 'rice' && rainfall < 1000) {
      advice.push("Rice requires high water availability - consider flood irrigation");
    }
    if (predictedCrop.toLowerCase() === 'wheat' && temperature > 25) {
      advice.push("Wheat prefers cooler temperatures - plant in cooler seasons");
    }
    
  } catch (error) {
    console.error('Error generating farming advice:', error);
  }
  
  return advice.length > 0 ? advice.slice(0, 4) : [`Soil conditions appear suitable for ${predictedCrop} cultivation`];
}

async function callOpenAI(inputs) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  
  if (!OPENAI_API_KEY) {
    throw new Error('OpenAI API key not configured');
  }

  const prompt = `You are an agricultural AI expert. Based on these soil and climate parameters, predict the top 5 most suitable crops and provide their suitability percentages.

Soil Parameters:
- Nitrogen (N): ${inputs.N} kg/ha
- Phosphorus (P): ${inputs.P} kg/ha  
- Potassium (K): ${inputs.K} kg/ha
- pH Level: ${inputs.pH}
- Temperature: ${inputs.temperature}°C
- Annual Rainfall: ${inputs.rainfall}mm

Consider the optimal growing conditions for different crops. Respond in this exact JSON format:
{
  "crops": [
    {"name": "crop_name", "suitability": percentage_number},
    {"name": "crop_name", "suitability": percentage_number}
  ],
  "primary_crop": "best_crop_name",
  "confidence": percentage_number,
  "reasoning": "brief_explanation"
}

Available crops to choose from: rice, wheat, maize, corn, tomato, potato, onion, cabbage, carrot, soybean, cotton, sunflower, banana, mango, barley, beans, peas, cucumber, lettuce, spinach`;

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are an expert agricultural consultant with deep knowledge of crop requirements and soil science.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 500,
        temperature: 0.3
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content;
    
    if (!content) {
      throw new Error('No response from OpenAI');
    }

    // Parse JSON response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Invalid JSON response from OpenAI');
    }

    return JSON.parse(jsonMatch[0]);
  } catch (error) {
    console.error('OpenAI API call failed:', error);
    throw error;
  }
}

function ruleBasedPrediction(inputs) {
  const cropScores = [];
  
  Object.entries(CROP_DATABASE).forEach(([cropName, requirements]) => {
    let score = 0;
    let factors = 0;
    
    // pH compatibility (30% weight)
    if (isInRange(inputs.pH, requirements.ph)) {
      score += 30;
    } else {
      const phDiff = Math.min(
        Math.abs(inputs.pH - requirements.ph[0]),
        Math.abs(inputs.pH - requirements.ph[1])
      );
      score += Math.max(0, 30 - (phDiff * 5));
    }
    factors++;
    
    // Temperature compatibility (25% weight)  
    if (isInRange(inputs.temperature, requirements.temp)) {
      score += 25;
    } else {
      const tempDiff = Math.min(
        Math.abs(inputs.temperature - requirements.temp[0]),
        Math.abs(inputs.temperature - requirements.temp[1])
      );
      score += Math.max(0, 25 - (tempDiff * 1));
    }
    factors++;
    
    // Rainfall compatibility (25% weight)
    if (isInRange(inputs.rainfall, requirements.rainfall)) {
      score += 25;
    } else {
      const rainDiff = Math.min(
        Math.abs(inputs.rainfall - requirements.rainfall[0]),
        Math.abs(inputs.rainfall - requirements.rainfall[1])
      ) / 100;
      score += Math.max(0, 25 - rainDiff);
    }
    factors++;
    
    // Nutrient compatibility (20% weight total)
    let nutrientScore = 0;
    if (isInRange(inputs.N, requirements.N)) nutrientScore += 7;
    if (isInRange(inputs.P, requirements.P)) nutrientScore += 7;
    if (isInRange(inputs.K, requirements.K)) nutrientScore += 6;
    score += nutrientScore;
    
    cropScores.push({
      name: cropName,
      suitability: Math.min(100, Math.max(0, Math.round(score))),
      emoji: requirements.emoji
    });
  });

  // Sort by suitability score
  cropScores.sort((a, b) => b.suitability - a.suitability);
  
  return {
    crops: cropScores.slice(0, 5),
    primary_crop: cropScores[0]?.name || 'maize',
    confidence: cropScores[0]?.suitability || 50,
    reasoning: 'Prediction based on soil and climate parameter matching'
  };
}

async function tryMLBackend(inputs) {
  const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'https://agricast-backend.onrender.com';
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);
    
    const response = await fetch(`${ML_BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputs),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`ML Backend error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.success && data.recommendations) {
      return {
        crops: data.recommendations.map(rec => ({
          name: rec.crop.toLowerCase(),
          suitability: rec.probability,
          emoji: rec.emoji
        })),
        primary_crop: data.predicted_crop.toLowerCase(),
        confidence: data.confidence,
        reasoning: 'ML model prediction from backend service'
      };
    } else {
      throw new Error('Invalid ML backend response');
    }
  } catch (error) {
    console.error('ML Backend failed:', error);
    throw error;
  }
}

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({
      success: false,
      error: 'Method not allowed. Use POST.'
    });
  }

  try {
    const inputs = req.body;
    
    // Validate input
    const validationErrors = validateInput(inputs);
    if (validationErrors.length > 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid input parameters',
        details: validationErrors
      });
    }

    const processedInputs = {
      N: parseFloat(inputs.N),
      P: parseFloat(inputs.P),
      K: parseFloat(inputs.K),
      pH: parseFloat(inputs.pH),
      temperature: parseFloat(inputs.temperature),
      rainfall: parseFloat(inputs.rainfall)
    };

    let predictionResult;
    let predictionMethod = 'unknown';

    // Try prediction methods in order of preference
    try {
      // 1. Try ML Backend first (if available)
      predictionResult = await tryMLBackend(processedInputs);
      predictionMethod = 'ml_backend';
    } catch (mlError) {
      console.log('ML Backend unavailable, trying OpenAI...');
      
      try {
        // 2. Try OpenAI GPT
        predictionResult = await callOpenAI(processedInputs);
        predictionMethod = 'openai_gpt';
      } catch (aiError) {
        console.log('OpenAI unavailable, using rule-based prediction...');
        
        // 3. Fallback to rule-based system
        predictionResult = ruleBasedPrediction(processedInputs);
        predictionMethod = 'rule_based';
      }
    }

    // Format response
    const recommendations = predictionResult.crops.map(crop => ({
      crop: crop.name.charAt(0).toUpperCase() + crop.name.slice(1),
      emoji: CROP_EMOJIS[crop.name.toLowerCase()] || CROP_EMOJIS.default,
      probability: crop.suitability,
      suitability: crop.suitability >= 70 ? 'High' : crop.suitability >= 40 ? 'Medium' : 'Low'
    }));

    const response = {
      success: true,
      predicted_crop: predictionResult.primary_crop.charAt(0).toUpperCase() + predictionResult.primary_crop.slice(1),
      confidence: predictionResult.confidence,
      recommendations: recommendations,
      input_data: processedInputs,
      soil_health: calculateSoilHealth(processedInputs),
      farming_advice: generateFarmingAdvice(processedInputs, predictionResult.primary_crop),
      prediction_method: predictionMethod,
      reasoning: predictionResult.reasoning || '',
      timestamp: new Date().toISOString()
    };

    console.log(`Prediction completed using ${predictionMethod}: ${predictionResult.primary_crop}`);
    
    return res.status(200).json(response);

  } catch (error) {
    console.error('Prediction endpoint error:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Prediction failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

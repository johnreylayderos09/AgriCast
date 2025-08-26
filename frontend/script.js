// Enhanced AgriCast Configuration: Render ML + Vercel Generative AI + Weather APIs
const CONFIG = {
    // Render backend URL - for ML predictions
    ML_API_BASE_URL: "https://agricast-backend.onrender.com",
    
    // Vercel API functions - for generative AI and weather/geocoding
    // Uses relative paths so it works with any Vercel domain
    VERCEL_API_BASE_URL: "/api",
    
    // Request timeout (increased for cold starts)
    REQUEST_TIMEOUT: 35000,
    
    // Prediction strategy priority
    PREDICTION_PRIORITY: ['ml_model', 'generative_ai', 'rule_based']
};

// Enhanced utility function for API requests with better error handling
async function makeApiRequest(url, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    };

    const finalOptions = { ...defaultOptions, ...options };
    
    try {
        console.log(`Making request to: ${url}`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);
        
        const response = await fetch(url, {
            ...finalOptions,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        console.log(`Response status: ${response.status} for ${url}`);
        
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorData.message || errorMessage;
                } else {
                    const textError = await response.text();
                    errorMessage = textError || errorMessage;
                }
            } catch (parseError) {
                console.warn('Could not parse error response:', parseError);
            }
            throw new Error(errorMessage);
        }
        
        // Handle different response types
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            console.log(`Response data:`, data);
            return data;
        } else {
            const text = await response.text();
            console.log(`Response text:`, text);
            return { success: true, message: text };
        }
        
    } catch (error) {
        console.error(`Request failed for ${url}:`, error);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - service may be starting up');
        }
        throw error;
    }
}

// Test both ML and AI systems with improved detection
async function testSystemConnections() {
    const results = {
        mlModel: false,
        generativeAI: false,
        weather: false
    };
    
    showClimateStatus('🔌 Testing system connections...', 'loading');
    
    // Test ML Model (Render) - with multiple fallback attempts
    try {
        console.log('Testing ML Model health at:', `${CONFIG.ML_API_BASE_URL}/health`);
        const mlResponse = await makeApiRequest(`${CONFIG.ML_API_BASE_URL}/health`);
        
        // Flexible health check - accept various success indicators
        if (mlResponse && (
            mlResponse.status === 'healthy' || 
            mlResponse.status === 'OK' || 
            mlResponse.success === true ||
            mlResponse.message === 'ML model is ready' ||
            (typeof mlResponse === 'string' && (mlResponse.includes('healthy') || mlResponse.includes('ready')))
        )) {
            results.mlModel = true;
            console.log('✅ ML Model health check passed:', mlResponse);
        } else {
            console.log('ML Model health check returned unexpected format:', mlResponse);
            // Try a simple prediction test
            const testPrediction = await makeApiRequest(`${CONFIG.ML_API_BASE_URL}/predict`, {
                method: 'POST',
                body: JSON.stringify({ N: 120, P: 80, K: 200, pH: 6.5, temperature: 25, rainfall: 1200 })
            });
            
            if (testPrediction && (testPrediction.success || testPrediction.recommendations)) {
                results.mlModel = true;
                console.log('✅ ML Model prediction test passed:', testPrediction);
            }
        }
    } catch (error) {
        console.warn('⚠️ ML Model unavailable:', error.message);
        
        // Try root endpoint as backup
        try {
            const rootResponse = await makeApiRequest(`${CONFIG.ML_API_BASE_URL}/`);
            if (rootResponse) {
                results.mlModel = true;
                console.log('✅ ML Model root endpoint accessible');
            }
        } catch (rootError) {
            console.warn('⚠️ ML Model completely unavailable');
        }
    }
    
    // Test Generative AI (Vercel)
    try {
        console.log('Testing Generative AI health at:', `${CONFIG.VERCEL_API_BASE_URL}/health`);
        const aiResponse = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/health`);
        
        if (aiResponse && (
            aiResponse.status === 'healthy' || 
            aiResponse.status === 'degraded' ||
            aiResponse.success === true ||
            aiResponse.services
        )) {
            results.generativeAI = true;
            console.log('✅ Generative AI connected:', aiResponse);
        }
    } catch (error) {
        console.warn('⚠️ Generative AI unavailable:', error.message);
    }
    
    // Test Weather API (Vercel) - with better error handling
    try {
        console.log('Testing Weather API...');
        const weatherTest = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/geocoding?q=Manila`);
        
        if (weatherTest && (
            (weatherTest.success && weatherTest.data && weatherTest.data.length > 0) ||
            (Array.isArray(weatherTest) && weatherTest.length > 0)
        )) {
            results.weather = true;
            console.log('✅ Weather API connected');
        }
    } catch (error) {
        console.warn('⚠️ Weather API unavailable:', error.message);
    }
    
    // Display comprehensive connection status
    const connectedServices = Object.values(results).filter(Boolean).length;
    const totalServices = Object.keys(results).length;
    
    const serviceNames = {
        mlModel: 'ML Model (Render)',
        generativeAI: 'AI Assistant (Vercel)', 
        weather: 'Weather API (Vercel)'
    };
    
    const connectedList = Object.entries(results)
        .filter(([key, connected]) => connected)
        .map(([key]) => serviceNames[key]);
    
    const disconnectedList = Object.entries(results)
        .filter(([key, connected]) => !connected)
        .map(([key]) => serviceNames[key]);
    
    if (connectedServices === totalServices) {
        showClimateStatus('✅ All systems connected and ready!', 'success');
        console.log('🎉 Connected services:', connectedList);
    } else if (connectedServices > 0) {
        showClimateStatus(`⚠️ ${connectedServices}/${totalServices} systems connected`, 'warning');
        console.log('✅ Connected services:', connectedList);
        console.log('❌ Disconnected services:', disconnectedList);
    } else {
        showClimateStatus('❌ No systems available - using offline mode', 'error');
        console.log('❌ All services disconnected:', Object.values(serviceNames));
    }
    
    return results;
}

// Enhanced prediction function with multiple AI approaches
async function runPrediction() {
    // Get input values with validation
    const inputs = {
        N: parseFloat(document.getElementById('nitrogen').value) || 120,
        P: parseFloat(document.getElementById('phosphorus').value) || 80,
        K: parseFloat(document.getElementById('potassium').value) || 200,
        pH: parseFloat(document.getElementById('ph').value) || 6.5,
        temperature: parseFloat(document.getElementById('temperature').value) || 25,
        rainfall: parseFloat(document.getElementById('rainfall').value) || 1200
    };

    console.log('🚀 Starting multi-system prediction with inputs:', inputs);

    // Show loading state
    document.getElementById('initial-state').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    // Test system availability
    const systemStatus = await testSystemConnections();
    
    let predictionResult = null;
    let usedMethod = 'unknown';
    
    // Strategy 1: Try ML Model first (Render backend)
    if (systemStatus.mlModel) {
        try {
            showClimateStatus('🔬 Running ML model prediction...', 'loading');
            console.log('Attempting ML model prediction...');
            
            const mlResponse = await makeApiRequest(`${CONFIG.ML_API_BASE_URL}/predict`, {
                method: 'POST',
                body: JSON.stringify(inputs)
            });

            console.log('ML Model Response:', mlResponse);

            if (mlResponse && (mlResponse.success || mlResponse.recommendations)) {
                predictionResult = {
                    score: mlResponse.confidence || 75,
                    soilHealth: calculateSoilHealthFromInputs(inputs),
                    recommendedCrops: (mlResponse.recommendations || []).map(crop => ({
                        emoji: crop.emoji || '🌱',
                        name: crop.crop || crop.name || 'Unknown',
                        compatibility: Math.round(crop.probability || crop.compatibility || 0)
                    })),
                    farmingAdvice: mlResponse.farming_advice || mlResponse.advice || [],
                    reasoning: mlResponse.reasoning || 'Based on trained machine learning model analysis'
                };
                
                usedMethod = 'ml_model';
                showClimateStatus('✅ ML model prediction completed!', 'success');
            }
        } catch (error) {
            console.warn('ML Model prediction failed:', error);
            showClimateStatus('⚠️ ML model failed, trying AI assistant...', 'warning');
        }
    }
    
    // Strategy 2: Try Generative AI (Vercel function)
    if (!predictionResult && systemStatus.generativeAI) {
        try {
            showClimateStatus('🤖 Running AI assistant prediction...', 'loading');
            console.log('Attempting generative AI prediction...');
            
            const aiResponse = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/predict`, {
                method: 'POST',
                body: JSON.stringify(inputs)
            });

            console.log('Generative AI Response:', aiResponse);

            if (aiResponse && (aiResponse.success || aiResponse.recommendations)) {
                predictionResult = {
                    score: Math.round(aiResponse.confidence || 80),
                    soilHealth: calculateSoilHealthFromInputs(inputs),
                    recommendedCrops: (aiResponse.recommendations || []).map(crop => ({
                        emoji: crop.emoji || '🌱',
                        name: crop.crop || crop.name || 'Unknown',
                        compatibility: Math.round(crop.probability || crop.compatibility || 0)
                    })),
                    farmingAdvice: aiResponse.farming_advice || aiResponse.advice || [],
                    reasoning: aiResponse.reasoning || 'Generated by AI assistant based on agricultural knowledge'
                };
                
                usedMethod = 'generative_ai';
                showClimateStatus('✅ AI assistant prediction completed!', 'success');
            }
        } catch (error) {
            console.warn('Generative AI prediction failed:', error);
            showClimateStatus('⚠️ AI assistant failed, using rule-based system...', 'warning');
        }
    }
    
    // Strategy 3: Fallback to rule-based system
    if (!predictionResult) {
        try {
            showClimateStatus('📊 Using rule-based prediction system...', 'loading');
            console.log('Using rule-based fallback system...');
            
            predictionResult = await ruleBasedFallback(inputs);
            usedMethod = 'rule_based';
            showClimateStatus('✅ Rule-based prediction completed!', 'success');
        } catch (error) {
            console.error('All prediction methods failed:', error);
            showError('All prediction systems are currently unavailable. Please try again later.');
            return;
        }
    }

    // Display results with method indicator
    if (predictionResult) {
        displayResults(predictionResult, inputs, usedMethod);
    }

    // Hide loading and show results
    setTimeout(() => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';
        
        const chartSection = document.getElementById('chart-section');
        if (chartSection) chartSection.style.display = 'block';
    }, 500);
}

// Enhanced rule-based fallback system
async function ruleBasedFallback(inputs) {
    const cropDatabase = {
        rice: { 
            ph: [5.5, 6.5], temp: [20, 30], rainfall: [1000, 2000], 
            N: [100, 200], P: [50, 100], K: [100, 200], 
            emoji: '🌾',
            advice: ['Plant in flooded fields during rainy season', 'Ensure proper drainage during harvest', 'Apply nitrogen in 3 split doses']
        },
        wheat: { 
            ph: [6.0, 7.5], temp: [15, 25], rainfall: [400, 800], 
            N: [120, 180], P: [40, 80], K: [80, 150], 
            emoji: '🌾',
            advice: ['Plant in well-drained soil', 'Apply nitrogen fertilizer in split doses', 'Avoid waterlogging']
        },
        maize: { 
            ph: [6.0, 7.0], temp: [20, 30], rainfall: [600, 1200], 
            N: [150, 250], P: [60, 100], K: [120, 200], 
            emoji: '🌽',
            advice: ['Ensure adequate spacing (75cm between rows)', 'Apply potash at flowering stage', 'Regular weeding is essential']
        },
        tomato: { 
            ph: [6.0, 6.8], temp: [18, 26], rainfall: [400, 800], 
            N: [100, 150], P: [80, 120], K: [150, 250], 
            emoji: '🍅',
            advice: ['Use drip irrigation to prevent leaf diseases', 'Support plants with stakes', 'Apply mulch around plants']
        },
        potato: { 
            ph: [5.0, 6.5], temp: [15, 25], rainfall: [500, 800], 
            N: [120, 180], P: [60, 100], K: [200, 300], 
            emoji: '🥔',
            advice: ['Plant in raised beds or ridges', 'Hill up soil around plants as they grow', 'Harvest before rainy season']
        },
        soybean: {
            ph: [6.0, 7.0], temp: [20, 30], rainfall: [450, 700],
            N: [50, 100], P: [40, 80], K: [100, 200],
            emoji: '🫘',
            advice: ['Inoculate seeds with rhizobia bacteria', 'Avoid waterlogging conditions', 'Harvest when pods turn brown']
        },
        onion: {
            ph: [6.0, 7.0], temp: [15, 25], rainfall: [350, 600],
            N: [100, 150], P: [50, 80], K: [100, 150],
            emoji: '🧅',
            advice: ['Plant in well-drained sandy loam soil', 'Avoid excessive nitrogen in later stages', 'Cure bulbs properly after harvest']
        },
        cabbage: {
            ph: [6.0, 6.8], temp: [15, 20], rainfall: [600, 1000],
            N: [150, 200], P: [60, 100], K: [150, 200],
            emoji: '🥬',
            advice: ['Plant in cool season', 'Ensure consistent moisture', 'Apply balanced fertilizer regularly']
        }
    };

    const cropScores = [];
    const farmingAdvice = [];
    
    Object.entries(cropDatabase).forEach(([cropName, requirements]) => {
        let score = 0;
        let matchCount = 0;
        
        // Calculate compatibility score with weighted factors
        if (isInRange(inputs.pH, requirements.ph)) { 
            score += 25; matchCount++; 
        } else {
            score += Math.max(5, 25 - Math.abs(inputs.pH - (requirements.ph[0] + requirements.ph[1]) / 2) * 5);
        }
        
        if (isInRange(inputs.temperature, requirements.temp)) { 
            score += 25; matchCount++; 
        } else {
            score += Math.max(5, 25 - Math.abs(inputs.temperature - (requirements.temp[0] + requirements.temp[1]) / 2) * 2);
        }
        
        if (isInRange(inputs.rainfall, requirements.rainfall)) { 
            score += 20; matchCount++; 
        } else {
            score += Math.max(3, 20 - Math.abs(inputs.rainfall - (requirements.rainfall[0] + requirements.rainfall[1]) / 2) * 0.01);
        }
        
        if (isInRange(inputs.N, requirements.N)) { 
            score += 15; matchCount++; 
        } else {
            score += Math.max(3, 15 - Math.abs(inputs.N - (requirements.N[0] + requirements.N[1]) / 2) * 0.1);
        }
        
        if (isInRange(inputs.P, requirements.P)) { 
            score += 10; matchCount++; 
        } else {
            score += Math.max(2, 10 - Math.abs(inputs.P - (requirements.P[0] + requirements.P[1]) / 2) * 0.2);
        }
        
        if (isInRange(inputs.K, requirements.K)) { 
            score += 5; matchCount++; 
        } else {
            score += Math.max(1, 5 - Math.abs(inputs.K - (requirements.K[0] + requirements.K[1]) / 2) * 0.05);
        }
        
        cropScores.push({
            crop: cropName,
            confidence: Math.min(100, Math.round(score)),
            emoji: requirements.emoji,
            matchCount: matchCount
        });
        
        // Add farming advice for well-matching crops
        if (score > 60) {
            farmingAdvice.push(...requirements.advice);
        }
    });

    cropScores.sort((a, b) => b.confidence - a.confidence);
    
    // Generate general advice based on soil conditions
    const generalAdvice = [];
    if (inputs.pH < 6.0) generalAdvice.push('Apply lime to raise soil pH for better nutrient availability');
    if (inputs.pH > 7.5) generalAdvice.push('Add organic matter or sulfur to lower soil pH');
    if (inputs.N < 100) generalAdvice.push('Increase nitrogen fertilization for better plant growth');
    if (inputs.P < 50) generalAdvice.push('Apply phosphorus fertilizer to improve root development');
    if (inputs.K < 150) generalAdvice.push('Add potassium fertilizer for better disease resistance');
    if (inputs.temperature > 30) generalAdvice.push('Consider shade nets or cooling methods for sensitive crops');
    if (inputs.rainfall > 1500) generalAdvice.push('Ensure proper drainage to prevent waterlogging');
    
    return {
        score: cropScores[0]?.confidence || 50,
        soilHealth: calculateSoilHealthFromInputs(inputs),
        recommendedCrops: cropScores.slice(0, 6).map(crop => ({
            emoji: crop.emoji,
            name: crop.crop.charAt(0).toUpperCase() + crop.crop.slice(1),
            compatibility: crop.confidence
        })),
        farmingAdvice: [...new Set([...farmingAdvice, ...generalAdvice])].slice(0, 8),
        reasoning: 'Analysis based on optimal growing conditions and soil requirements for each crop type'
    };
}

// Calculate soil health from inputs with better algorithm
function calculateSoilHealthFromInputs(inputs) {
    let health = 0;
    
    // pH health (ideal: 6.0-7.0) - weighted 30%
    if (inputs.pH >= 6.0 && inputs.pH <= 7.0) {
        health += 30;
    } else if (inputs.pH >= 5.5 && inputs.pH <= 7.5) {
        health += 25;
    } else if (inputs.pH >= 5.0 && inputs.pH <= 8.0) {
        health += 15;
    } else {
        health += 5;
    }
    
    // Nutrient levels with realistic scoring - weighted 70%
    // Nitrogen (ideal: 120-180)
    const nScore = inputs.N >= 120 && inputs.N <= 180 ? 25 : 
                   inputs.N >= 80 && inputs.N <= 220 ? 20 : 
                   inputs.N >= 40 && inputs.N <= 260 ? 15 : 10;
    health += nScore;
    
    // Phosphorus (ideal: 50-100)  
    const pScore = inputs.P >= 50 && inputs.P <= 100 ? 20 :
                   inputs.P >= 30 && inputs.P <= 120 ? 15 :
                   inputs.P >= 20 && inputs.P <= 140 ? 10 : 5;
    health += pScore;
    
    // Potassium (ideal: 150-250)
    const kScore = inputs.K >= 150 && inputs.K <= 250 ? 25 :
                   inputs.K >= 100 && inputs.K <= 300 ? 20 :
                   inputs.K >= 50 && inputs.K <= 350 ? 15 : 10;
    health += kScore;
    
    return Math.round(Math.min(100, Math.max(0, health)));
}

function isInRange(value, [min, max]) {
    return value >= min && value <= max;
}

// Enhanced display results function with method indicator
function displayResults(results, inputs, usedMethod) {
    // Update prediction score
    const scoreElement = document.getElementById('score');
    if (scoreElement) {
        scoreElement.textContent = `${results.score}%`;
    }
    
    // Show prediction method used
    const methodIndicators = {
        'ml_model': { emoji: '🔬', name: 'ML Model', color: '#4CAF50' },
        'generative_ai': { emoji: '🤖', name: 'AI Assistant', color: '#2196F3' },
        'rule_based': { emoji: '📊', name: 'Rule-Based', color: '#FF9800' }
    };
    
    const method = methodIndicators[usedMethod] || methodIndicators['rule_based'];
    
    // Add/update method indicator
    const scoreContainer = scoreElement ? scoreElement.parentElement : null;
    if (scoreContainer) {
        let methodDiv = document.getElementById('method-indicator');
        if (!methodDiv) {
            methodDiv = document.createElement('div');
            methodDiv.id = 'method-indicator';
            scoreContainer.appendChild(methodDiv);
        }
        
        methodDiv.style.cssText = `
            margin-top: 10px; 
            padding: 8px 12px; 
            background: ${method.color}20; 
            border: 1px solid ${method.color}; 
            border-radius: 20px; 
            font-size: 0.85rem; 
            color: ${method.color};
            text-align: center;
            font-weight: 500;
        `;
        methodDiv.innerHTML = `${method.emoji} Predicted using ${method.name}`;
    }
    
    // Update soil health
    const soilHealthBar = document.getElementById('soil-health');
    const soilHealthText = document.getElementById('soil-health-text');
    
    if (soilHealthBar && soilHealthText) {
        soilHealthBar.style.width = `${results.soilHealth}%`;
        
        if (results.soilHealth >= 70) {
            soilHealthBar.className = 'indicator-fill status-good';
            soilHealthText.textContent = `Good (${results.soilHealth}%)`;
        } else if (results.soilHealth >= 40) {
            soilHealthBar.className = 'indicator-fill status-moderate';
            soilHealthText.textContent = `Moderate (${results.soilHealth}%)`;
        } else {
            soilHealthBar.className = 'indicator-fill status-poor';
            soilHealthText.textContent = `Poor (${results.soilHealth}%)`;
        }
    }
    
    // Update crop recommendations
    const cropContainer = document.getElementById('crop-recommendations');
    if (cropContainer && results.recommendedCrops) {
        cropContainer.innerHTML = results.recommendedCrops.map(crop => `
            <div class="crop-item" style="
                display: flex; 
                align-items: center; 
                padding: 12px; 
                margin: 8px 0; 
                background: #f8f9fa; 
                border-radius: 8px;
                border-left: 4px solid ${crop.compatibility >= 70 ? '#4CAF50' : crop.compatibility >= 50 ? '#FF9800' : '#f44336'};
            ">
                <div class="crop-emoji" style="font-size: 2rem; margin-right: 12px;">${crop.emoji}</div>
                <div style="flex: 1;">
                    <div style="font-weight: bold; font-size: 1.1rem;">${crop.name}</div>
                    <div style="color: #666; font-size: 0.9rem;">Compatibility: ${crop.compatibility}%</div>
                </div>
                <div style="
                    background: ${crop.compatibility >= 70 ? '#4CAF50' : crop.compatibility >= 50 ? '#FF9800' : '#f44336'};
                    color: white;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    font-weight: bold;
                ">
                    ${crop.compatibility >= 70 ? 'High' : crop.compatibility >= 50 ? 'Medium' : 'Low'}
                </div>
            </div>
        `).join('');
    }
    
    // Display farming advice
    if (results.farmingAdvice && results.farmingAdvice.length > 0) {
        const adviceSection = document.getElementById('farming-advice-section');
        const adviceContainer = document.getElementById('farming-advice');
        if (adviceContainer && adviceSection) {
            adviceSection.style.display = 'block';
            adviceContainer.innerHTML = results.farmingAdvice.map((advice, index) => `
                <div class="advice-item" style="
                    padding: 8px 12px;
                    margin: 6px 0;
                    background: #e8f5e8;
                    border-left: 3px solid #4CAF50;
                    border-radius: 4px;
                    font-size: 0.9rem;
                ">
                    <strong>${index + 1}.</strong> ${advice}
                </div>
            `).join('');
        }
    }
    
    // Display prediction reasoning
    if (results.reasoning) {
        const reasoningSection = document.getElementById('reasoning-section');
        const reasoningContainer = document.getElementById('prediction-reasoning');
        if (reasoningContainer && reasoningSection) {
            reasoningSection.style.display = 'block';
            reasoningContainer.style.cssText = `
                padding: 12px;
                background: #f0f8ff;
                border: 1px solid #2196F3;
                border-radius: 6px;
                color: #1565c0;
                font-style: italic;
                margin: 10px 0;
            `;
            reasoningContainer.textContent = results.reasoning;
        }
    }
    
    // Update soil parameter bars
    updateSoilParameterBars(inputs);
}

function updateSoilParameterBars(inputs) {
    const parameters = [
        { id: 'ph', value: inputs.pH, min: 0, max: 14, optimal: [6.0, 7.0], unit: '' },
        { id: 'n', value: inputs.N, min: 0, max: 300, optimal: [100, 200], unit: 'ppm' },
        { id: 'p', value: inputs.P, min: 0, max: 150, optimal: [50, 100], unit: 'ppm' },
        { id: 'k', value: inputs.K, min: 0, max: 400, optimal: [100, 300], unit: 'ppm' }
    ];
    
    parameters.forEach(param => {
        const bar = document.getElementById(`${param.id}-bar`);
        const status = document.getElementById(`${param.id}-status`);
        const valueSpan = document.getElementById(`${param.id}-value`);
        
        if (bar && status) {
            const percentage = ((param.value - param.min) / (param.max - param.min)) * 100;
            bar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
            
            const isOptimal = param.value >= param.optimal[0] && param.value <= param.optimal[1];
            const isModerate = param.value >= param.optimal[0] * 0.7 && param.value <= param.optimal[1] * 1.3;
            
            if (isOptimal) {
                bar.className = 'indicator-fill status-good';
                status.textContent = 'Optimal';
                status.style.color = '#4CAF50';
            } else if (isModerate) {
                bar.className = 'indicator-fill status-moderate';
                status.textContent = 'Moderate';
                status.style.color = '#FF9800';
            } else {
                bar.className = 'indicator-fill status-poor';
                status.textContent = 'Poor';
                status.style.color = '#f44336';
            }
            
            if (valueSpan) {
                valueSpan.textContent = `${param.value}${param.unit}`;
            }
        }
    });
}

// Weather and location functions using Vercel API functions
async function getCurrentLocation() {
    const coordsDisplay = document.getElementById('coordinates');
    const locationInput = document.getElementById('location-input');
    
    if (!navigator.geolocation) {
        alert('Geolocation is not supported by this browser.');
        return;
    }
    
    coordsDisplay.textContent = 'Getting location...';
    
    navigator.geolocation.getCurrentPosition(
        async (position) => {
            const lat = position.coords.latitude.toFixed(4);
            const lon = position.coords.longitude.toFixed(4);
            
            coordsDisplay.textContent = `${lat}, ${lon}`;
            
            // Reverse geocoding using Vercel function
            try {
                const response = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/geocoding?lat=${lat}&lon=${lon}&type=reverse`);
                
                if (response.success && response.data && response.data.length > 0) {
                    const location = response.data[0];
                    locationInput.value = `${location.name}, ${location.country}`;
                } else if (response.length > 0) {
                    const location = response[0];
                    locationInput.value = `${location.name}, ${location.country}`;
                }
            } catch (error) {
                console.error('Reverse geocoding failed:', error);
            }
        },
        (error) => {
            coordsDisplay.textContent = 'Location access denied';
            console.error('Geolocation error:', error);
        }
    );
}

async function fetchClimateData() {
    const locationInput = document.getElementById('location-input');
    const tempInput = document.getElementById('temperature');
    const rainfallInput = document.getElementById('rainfall');
    const coordsDisplay = document.getElementById('coordinates');
    
    const location = locationInput.value.trim();
    if (!location) {
        showClimateStatus('Please enter a location first', 'error');
        return;
    }
    
    try {
        showClimateStatus('🔍 Looking up location coordinates...', 'loading');
        
        // Get coordinates from location name using Vercel function
        const geocodeResponse = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/geocoding?q=${encodeURIComponent(location)}`);
        
        let locationData;
        if (geocodeResponse.success && geocodeResponse.data && geocodeResponse.data.length > 0) {
            locationData = geocodeResponse.data[0];
        } else if (geocodeResponse.length > 0) {
            locationData = geocodeResponse[0];
        } else {
            throw new Error('Location not found');
        }
        
        const { lat, lon, name, country } = locationData;
        coordsDisplay.textContent = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
        locationInput.value = `${name}, ${country}`;
        
        showClimateStatus('🌡️ Fetching weather data...', 'loading');
        
        // Get weather data using Vercel function
        const weatherResponse = await makeApiRequest(`${CONFIG.VERCEL_API_BASE_URL}/weather?lat=${lat}&lon=${lon}`);
        
        let weatherData;
        if (weatherResponse.success && weatherResponse.data) {
            weatherData = weatherResponse.data;
        } else {
            weatherData = weatherResponse;
        }
        
        // Update weather display
        updateWeatherDisplay(weatherData, `${name}, ${country}`);
        
        // Update input fields
        if (tempInput && weatherData.main && weatherData.main.temp) {
            tempInput.value = weatherData.main.temp.toFixed(1);
        }
        
        // Estimate annual rainfall (enhanced estimation based on climate data)
        if (rainfallInput && weatherData.main) {
            const humidity = weatherData.main.humidity || 50;
            const temp = weatherData.main.temp || 25;
            const weatherCondition = weatherData.weather && weatherData.weather[0] ? weatherData.weather[0].main : 'Clear';
            
            // More sophisticated rainfall estimation
            let baseRainfall = humidity * 12; // Base on humidity
            
            // Adjust for temperature (cooler = more rain potential)
            baseRainfall += Math.max(0, 30 - temp) * 25;
            
            // Adjust for current weather condition
            const weatherMultiplier = {
                'Rain': 1.8,
                'Drizzle': 1.5,
                'Thunderstorm': 2.0,
                'Snow': 1.3,
                'Clouds': 1.2,
                'Clear': 0.8,
                'Mist': 1.1,
                'Fog': 1.1
            };
            
            baseRainfall *= (weatherMultiplier[weatherCondition] || 1.0);
            
            // Add some randomness and ensure reasonable range
            const estimatedAnnualRainfall = Math.round(
                baseRainfall + Math.random() * 200 + 300
            );
            
            // Clamp to realistic values (200-3000mm annually)
            rainfallInput.value = Math.min(3000, Math.max(200, estimatedAnnualRainfall));
        }
        
        showClimateStatus('✅ Climate data retrieved successfully!', 'success');
        
    } catch (error) {
        showClimateStatus(`❌ Error: ${error.message}`, 'error');
        console.error('Climate data fetch error:', error);
    }
}

function updateWeatherDisplay(weatherData, locationName) {
    const weatherSummary = document.getElementById('weather-summary');
    const weatherIcon = document.getElementById('weather-icon');
    const weatherMain = document.getElementById('weather-main');
    const weatherDesc = document.getElementById('weather-desc');
    const currentLocation = document.getElementById('current-location');
    const feelsLike = document.getElementById('feels-like');
    const humidity = document.getElementById('humidity');
    
    if (weatherSummary && weatherData && weatherData.weather && weatherData.main) {
        weatherSummary.style.display = 'block';
        
        // Weather icons mapping
        const iconMap = {
            '01d': '☀️', '01n': '🌙', '02d': '⛅', '02n': '☁️',
            '03d': '☁️', '03n': '☁️', '04d': '☁️', '04n': '☁️',
            '09d': '🌧️', '09n': '🌧️', '10d': '🌦️', '10n': '🌦️',
            '11d': '⛈️', '11n': '⛈️', '13d': '❄️', '13n': '❄️',
            '50d': '🌫️', '50n': '🌫️'
        };
        
        if (weatherIcon) {
            weatherIcon.textContent = iconMap[weatherData.weather[0].icon] || '🌤️';
        }
        if (weatherMain) {
            weatherMain.textContent = weatherData.weather[0].main || 'Unknown';
        }
        if (weatherDesc) {
            weatherDesc.textContent = weatherData.weather[0].description || 'Weather data';
        }
        if (currentLocation) {
            currentLocation.textContent = locationName;
        }
        if (feelsLike) {
            feelsLike.textContent = weatherData.main.feels_like ? weatherData.main.feels_like.toFixed(1) : 'N/A';
        }
        if (humidity) {
            humidity.textContent = weatherData.main.humidity || 'N/A';
        }
    }
}

function showClimateStatus(message, type) {
    const statusDiv = document.getElementById('climate-status');
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = `climate-status ${type}`;
        
        // Auto-hide success messages after 3 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (statusDiv.textContent === message) {
                    statusDiv.textContent = '';
                    statusDiv.className = 'climate-status';
                }
            }, 3000);
        }
    }
}

function showError(message) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
    const cropContainer = document.getElementById('crop-recommendations');
    if (cropContainer) {
        cropContainer.innerHTML = `
            <div style="
                text-align: center; 
                padding: 30px 20px; 
                color: #d32f2f;
                background: #ffebee;
                border: 2px solid #ffcdd2;
                border-radius: 12px;
                margin: 20px 0;
            ">
                <div style="font-size: 3rem; margin-bottom: 15px;">❌</div>
                <h3 style="margin: 10px 0; color: #c62828;">Prediction Failed</h3>
                <p style="margin: 15px 0; font-size: 1.1rem;"><strong>Error:</strong> ${message}</p>
                <p style="margin: 10px 0; color: #666;">All prediction systems are currently unavailable.</p>
                <div style="margin-top: 20px;">
                    <button onclick="runPrediction()" style="
                        margin: 5px; 
                        padding: 12px 24px; 
                        background: #4CAF50; 
                        color: white; 
                        border: none; 
                        border-radius: 6px; 
                        cursor: pointer;
                        font-size: 1rem;
                        font-weight: bold;
                        transition: background 0.3s;
                    " onmouseover="this.style.background='#45a049'" onmouseout="this.style.background='#4CAF50'">
                        🔄 Try Again
                    </button>
                    <button onclick="location.reload()" style="
                        margin: 5px; 
                        padding: 12px 24px; 
                        background: #2196F3; 
                        color: white; 
                        border: none; 
                        border-radius: 6px; 
                        cursor: pointer;
                        font-size: 1rem;
                        font-weight: bold;
                        transition: background 0.3s;
                    " onmouseover="this.style.background='#1976d2'" onmouseout="this.style.background='#2196F3'">
                        🔄 Reload Page
                    </button>
                </div>
            </div>
        `;
    }
}

// Enhanced initialization with better error handling
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced AgriCast initialized');
    console.log('Current URL:', window.location.href);
    console.log('ML Backend URL:', CONFIG.ML_API_BASE_URL);
    console.log('Vercel API URL:', CONFIG.VERCEL_API_BASE_URL);
    console.log('Prediction Priority:', CONFIG.PREDICTION_PRIORITY);
    
    // Test all system connections on load
    setTimeout(() => {
        testSystemConnections().catch(error => {
            console.error('System test failed:', error);
            showClimateStatus('❌ System initialization failed', 'error');
        });
    }, 1000);
    
    // Add event listeners for form inputs if they exist
    const inputs = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'temperature', 'rainfall'];
    inputs.forEach(inputId => {
        const element = document.getElementById(inputId);
        if (element) {
            element.addEventListener('input', function() {
                // Clear previous results when inputs change
                const methodIndicator = document.getElementById('method-indicator');
                if (methodIndicator) {
                    methodIndicator.style.opacity = '0.5';
                }
            });
        }
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Ctrl + Enter to run prediction
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            runPrediction();
        }
        
        // Ctrl + L to get current location
        if (event.ctrlKey && event.key.toLowerCase() === 'l') {
            event.preventDefault();
            getCurrentLocation();
        }
    });
    
    console.log('✅ AgriCast initialization complete');
});

// Add global error handler for unhandled promises
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showClimateStatus('❌ An unexpected error occurred', 'error');
    event.preventDefault();
});

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        makeApiRequest,
        testSystemConnections,
        runPrediction,
        ruleBasedFallback,
        calculateSoilHealthFromInputs,
        getCurrentLocation,
        fetchClimateData
    };
}

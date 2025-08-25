export default async function handler(req, res) {
  // Enable CORS for all origins (adjust in production)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { q, lat, lon, type, limit = 5 } = req.query;
  
  const API_KEY = process.env.OPENWEATHER_API_KEY;
  
  if (!API_KEY) {
    return res.status(500).json({ error: 'API key not configured' });
  }
  
  try {
    let url;
    
    if (type === 'reverse' && lat && lon) {
      // Reverse geocoding
      url = `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=${limit}&appid=${API_KEY}`;
    } else if (q) {
      // Forward geocoding
      url = `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(q)}&limit=${limit}&appid=${API_KEY}`;
    } else {
      return res.status(400).json({ error: 'Missing required parameters' });
    }
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`OpenWeatherMap Geocoding API error: ${response.status}`);
    }
    
    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('Geocoding API error:', error);
    res.status(500).json({ error: 'Failed to fetch location data' });
  }
}

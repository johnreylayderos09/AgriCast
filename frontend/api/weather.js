// api/weather.js - Vercel serverless function for current weather data
export default async function handler(req, res) {
  // Enable CORS for all origins
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Cache-Control', 'no-store');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({
      success: false,
      error: 'Method not allowed. Use GET.'
    });
  }

  const { q, lat, lon, units = 'metric', lang } = req.query;

  const API_KEY = process.env.OPENWEATHER_API_KEY;

  if (!API_KEY) {
    return res.status(503).json({
      success: false,
      error: 'Weather service not configured'
    });
  }

  // Validate parameters: require either q OR lat+lon
  const hasCoords = typeof lat !== 'undefined' && typeof lon !== 'undefined' && lat !== '' && lon !== '';
  const hasQuery = typeof q === 'string' && q.trim().length > 0;

  if (!hasCoords && !hasQuery) {
    return res.status(400).json({
      success: false,
      error: 'Missing required parameters. Provide either "q" (city) or both "lat" and "lon"'
    });
  }

  try {
    let url;
    const base = 'https://api.openweathermap.org/data/2.5/weather';

    if (hasCoords) {
      const latNum = Number(lat);
      const lonNum = Number(lon);
      if (!Number.isFinite(latNum) || !Number.isFinite(lonNum)) {
        return res.status(400).json({ success: false, error: 'Invalid coordinates' });
      }
      const params = new URLSearchParams({ lat: String(latNum), lon: String(lonNum), units, appid: API_KEY });
      if (lang) params.set('lang', String(lang));
      url = `${base}?${params.toString()}`;
    } else {
      const cleanQuery = q.trim();
      const params = new URLSearchParams({ q: cleanQuery, units, appid: API_KEY });
      if (lang) params.set('lang', String(lang));
      url = `${base}?${params.toString()}`;
    }

    console.log('Weather request URL:', url.replace(API_KEY, 'API_KEY'));

    const response = await fetch(url, {
      method: 'GET',
      headers: { 'User-Agent': 'AgriCast-Weather/1.0' }
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('OpenWeatherMap Weather API error:', response.status, errorText);
      if (response.status === 401) {
        return res.status(503).json({ success: false, error: 'Weather service authentication failed' });
      } else if (response.status === 404) {
        return res.status(404).json({ success: false, error: 'Location not found' });
      } else if (response.status === 400) {
        return res.status(400).json({ success: false, error: 'Invalid weather request parameters' });
      }
      throw new Error(`OpenWeatherMap Weather API error: ${response.status}`);
    }

    const data = await response.json();
    return res.status(200).json({ success: true, data, timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Weather API error:', error);
    return res.status(500).json({ success: false, error: 'Failed to fetch weather data', message: error.message });
  }
}

// api/health.js - Updated Vercel health check endpoint
export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    const timestamp = new Date().toISOString();
    
    // Check environment variables
    const services = {
      weather_api: process.env.OPENWEATHER_API_KEY ? 'configured' : 'not_configured',
      openai_api: process.env.OPENAI_API_KEY ? 'configured' : 'not_configured',
      ml_backend: process.env.ML_BACKEND_URL ? 'configured' : 'using_default'
    };

    // Test ML backend connectivity (with timeout)
    let mlBackendStatus = 'unknown';
    const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'https://agricast-backend.onrender.com';
    
    try {
      const controller = new AbortController();
      const timeoutMs = 3000; // keep very short to avoid Vercel cold-start timeouts
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const mlResponse = await fetch(`${ML_BACKEND_URL}/health`, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'AgriCast-Health-Check/1.0'
        }
      }).catch(err => {
        // Guard against fetch hanging
        throw err;
      });

      clearTimeout(timeoutId);

      if (mlResponse && mlResponse.ok) {
        let mlData;
        try {
          mlData = await mlResponse.json();
        } catch (_) {
          mlData = { status: 'unknown' };
        }
        mlBackendStatus = mlData.status === 'healthy' ? 'connected' : 'connected_with_issues';
      } else if (mlResponse) {
        mlBackendStatus = `http_${mlResponse.status}`;
      } else {
        mlBackendStatus = 'unreachable';
      }
    } catch (error) {
      mlBackendStatus = error.name === 'AbortError' ? 'timeout' : 'unreachable';
    }

    // System information (safe values only)
    const systemInfo = {
      platform: 'vercel',
      node_version: process.version,
      memory_usage: {
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024), // MB
        heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) // MB
      },
      uptime: Math.round(process.uptime()) // seconds
    };

    // API capabilities
    const capabilities = {
      prediction_methods: [
        mlBackendStatus === 'connected' ? 'ml_backend (available)' : 'ml_backend (unavailable)',
        services.openai_api === 'configured' ? 'openai_gpt (available)' : 'openai_gpt (unavailable)',
        'rule_based (always available)'
      ],
      weather_services: services.weather_api === 'configured' ? 'available' : 'unavailable',
      geocoding_services: services.weather_api === 'configured' ? 'available' : 'unavailable'
    };

    // Determine overall health status
    let overallStatus = 'healthy';
    let statusMessage = 'All systems operational';
    
    const availableSystems = [
      mlBackendStatus === 'connected',
      services.openai_api === 'configured',
      services.weather_api === 'configured'
    ].filter(Boolean).length;
    
    if (availableSystems === 0) {
      overallStatus = 'degraded';
      statusMessage = 'Limited functionality - only rule-based predictions available';
    } else if (availableSystems < 3) {
      overallStatus = 'degraded';
      statusMessage = 'Partial functionality - some services unavailable';
    }

    // Build health response
    const healthData = {
      status: overallStatus,
      message: statusMessage,
      version: '2.1.0',
      timestamp: timestamp,
      success: true,
      services: services,
      ml_backend: {
        url: ML_BACKEND_URL,
        status: mlBackendStatus,
        last_check: timestamp
      },
      system: systemInfo,
      capabilities: capabilities,
      endpoints: {
        '/api/health': 'Health check and system status',
        '/api/predict': 'Crop prediction with multiple AI methods',
        '/api/weather': 'Weather data (requires OpenWeather API key)',
        '/api/geocoding': 'Location services (requires OpenWeather API key)'
      },
      diagnostics: {
        total_available_systems: availableSystems,
        total_systems: 3,
        critical_services: {
          prediction: mlBackendStatus === 'connected' || services.openai_api === 'configured' ? 'available' : 'limited',
          weather: services.weather_api === 'configured' ? 'available' : 'unavailable'
        }
      }
    };

    // Return appropriate status code
    // Always return quickly with 200 for health checks
    return res.status(200).json(healthData);

  } catch (error) {
    console.error('Health check error:', error);
    
    return res.status(200).json({ // Return 200 even for errors to help with debugging
      status: 'error',
      message: 'Health check encountered an error',
      error: error.message,
      timestamp: new Date().toISOString(),
      success: false,
      version: '2.1.0'
    });
  }
}

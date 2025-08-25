import { Configuration, OpenAIApi } from "openai";

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY, // Vercel environment variable
});
const openai = new OpenAIApi(configuration);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed, use POST' });
  }

  try {
    const { pH, N, P, K, temperature, rainfall } = req.body;

    const prompt = `
You are an agricultural expert AI. Provide a detailed reasoning on soil and climate:

Soil pH: ${pH}
Nitrogen (N): ${N} ppm
Phosphorus (P): ${P} ppm
Potassium (K): ${K} ppm
Temperature: ${temperature}°C
Annual Rainfall: ${rainfall} mm

Explain how these parameters impact crop suitability, nutrient availability, and give farming advice.
`;

    const completion = await openai.createChatCompletion({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 500,
      temperature: 0.7,
    });

    const reasoning = completion.data.choices[0].message.content.trim();

    res.status(200).json({ reasoning });

  } catch (error) {
    console.error("OpenAI API error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

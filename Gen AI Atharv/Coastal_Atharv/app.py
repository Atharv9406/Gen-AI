from flask import Flask, request, jsonify, send_from_directory
import os
import requests

# 1️⃣ Create the Flask app object BEFORE any @app.route decorators
app = Flask(__name__, static_folder='.', static_url_path='')

# 2️⃣ Serve frontend HTML
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 3️⃣ API route for AI generation
@app.route('/api/generate', methods=['POST'])
def generate_strategy():
    data = request.json

    # Updated prompt for short, plain-text answers
    prompt = f"""
You are a coastal engineer advisor.
Given the following site info, propose 3 coastal protection strategies.
Provide each strategy in a short, plain-text format. Do not use any Markdown symbols or headings.
Include only:
- strategy name
- 1-2 line summary
- main actions
- pros/cons
- rough cost (Low/Medium/High)
- environmental considerations
- monitoring suggestions

Site: {data.get('site')}
Region: {data.get('region')}
Shoreline: {data.get('shoreline')}
Issue: {data.get('issue')}
Budget: {data.get('budget')}
Preference: {data.get('preference')}
Notes: {data.get('notes')}
"""

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
        "search": True,
        "web": True,
        "reasoning_effort": "high"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract AI text
        choices = data.get("choices", [])
        ai_text = ""
        if len(choices) > 0:
            ai_text = choices[0].get("message", {}).get("content", "")
        if not ai_text:
            ai_text = "No content returned from AI."

        return jsonify({"strategy": ai_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

# 4️⃣ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
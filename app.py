
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import requests
import os
import json
import re
from dotenv import load_dotenv

# ------------------------------------------
# 1. INITIALIZATION
# ------------------------------------------

load_dotenv()

app = Flask(__name__)

FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

FACT_CHECK_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)

GEMINI_MODEL = "gemini-2.5-flash"

# Create Gemini client
client = None

if GEMINI_KEY:
    client = genai.Client(
        api_key=GEMINI_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta"
        )
    )


# ------------------------------------------
# 2. GOOGLE FACT CHECK API
# ------------------------------------------

def get_fact_checks(news):

    if not FACT_CHECK_KEY:
        return [], "Google Fact Check API key is missing."

    params = {
        "query": news,
        "key": FACT_CHECK_KEY,
        "languageCode": "en",
        "pageSize": 10
    }

    try:

        response = requests.get(
            FACT_CHECK_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        reviews = []

        for claim in data.get("claims", []):

            for review in claim.get("claimReview", []):

                reviews.append({
                    "claim": claim.get("text", ""),
                    "rating": review.get(
                        "textualRating", "Unknown"
                    ),
                    "publisher": review.get(
                        "publisher", {}
                    ).get("name", "Unknown"),
                    "title": review.get(
                        "title", "Fact Check"
                    ),
                    "url": review.get("url", ""),
                    "date": review.get("reviewDate", "")
                })

        return reviews, None

    except requests.exceptions.RequestException as error:

        print("FACT CHECK API ERROR:", repr(error))

        return [], "Google Fact Check search was unavailable."


# ------------------------------------------
# 3. EXTRACT GOOGLE SEARCH SOURCES
# ------------------------------------------

def extract_sources(response):

    sources = []

    try:

        for candidate in response.candidates or []:

            metadata = getattr(
                candidate,
                "grounding_metadata",
                None
            )

            if not metadata:
                continue

            for chunk in getattr(
                metadata,
                "grounding_chunks",
                []
            ):

                web = getattr(chunk, "web", None)

                if not web:
                    continue

                url = getattr(web, "uri", "")
                title = getattr(web, "title", "Source")

                if url and url.startswith("https://"):

                    if not any(
                        item["url"] == url
                        for item in sources
                    ):

                        sources.append({
                            "title": title,
                            "url": url
                        })

    except Exception as error:

        print("SOURCE EXTRACTION ERROR:", repr(error))

    return sources[:8]


# ------------------------------------------
# 4. GEMINI AI NEWS VERIFICATION
# ------------------------------------------

def verify_with_gemini(news, fact_checks):

    if client is None:
        raise ValueError(
            "Gemini API key is missing in Render Environment."
        )

    fact_check_data = json.dumps(
        fact_checks,
        ensure_ascii=False
    )

    prompt = f"""
You are a professional news fact-checking assistant.

Verify the exact factual statement provided below.

NEWS STATEMENT:
{news}

EXISTING GOOGLE FACT CHECK RESULTS:
{fact_check_data}

YOUR TASK:

1. Investigate the exact news statement.
2. Use Google Search grounding to find reliable evidence.
3. Prefer official government sources and established
   news organizations.
4. Compare the claim with the evidence.
5. Do not confuse a related claim with the exact claim.
6. Do not blindly trust an unrelated fact-check result.
7. Never invent evidence or source links.

VERDICT RULES:

TRUE:
Reliable evidence supports the central claim.

FAKE:
Reliable evidence clearly contradicts the central claim.

UNVERIFIED:
Evidence is insufficient, ambiguous or contradictory.

Return only a valid JSON object:

{{
  "verdict": "TRUE",
  "confidence": 95,
  "explanation": "A short explanation based on evidence.",
  "checked_claim": "The exact claim that was checked"
}}

The verdict must be TRUE, FAKE or UNVERIFIED.
Confidence must be a number between 0 and 100.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ],
            temperature=0.2
        )
    )

    response_text = response.text or ""

    # Extract JSON response
    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "Gemini did not return a valid JSON response."
        )

    result = json.loads(match.group())

    verdict = str(
        result.get("verdict", "UNVERIFIED")
    ).upper().strip()

    if verdict not in ["TRUE", "FAKE", "UNVERIFIED"]:
        verdict = "UNVERIFIED"

    try:
        confidence = float(
            result.get("confidence", 0)
        )
        confidence = max(0, min(100, confidence))

    except (ValueError, TypeError):
        confidence = 0

    explanation = str(
        result.get(
            "explanation",
            "No explanation available."
        )
    )

    checked_claim = str(
        result.get("checked_claim", "")
    )

    sources = extract_sources(response)

    # Require search evidence for a definite result
    if not sources:
        verdict = "UNVERIFIED"
        explanation += (
            " No Google Search grounding sources were returned. "
            "The claim requires manual verification."
        )

    # Avoid definite results with low confidence
    if confidence < 70:
        verdict = "UNVERIFIED"

    return {
        "status": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "checked_claim": checked_claim,
        "sources": sources
    }


# ------------------------------------------
# 5. HOME PAGE
# ------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ------------------------------------------
# 6. CHECK NEWS ROUTE
# ------------------------------------------

@app.route("/api/check", methods=["POST"])
def check_news():

    data = request.get_json(silent=True) or {}

    news = data.get("news", "").strip()

    if len(news) < 10:

        return jsonify({
            "error": "Please enter a valid news statement."
        }), 400

    if len(news) > 3000:

        return jsonify({
            "error": "Maximum 3000 characters allowed."
        }), 400

    if not GEMINI_KEY:

        return jsonify({
            "error": "Gemini API key is missing."
        }), 500

    try:

        # First: Google Fact Check API
        reviews, warning = get_fact_checks(news)

        print("\n========== FACT CHECK RESULTS ==========")
        print("NEWS:", news)
        print("REVIEWS FOUND:", len(reviews))

        for review in reviews:
            print("CLAIM:", review["claim"])
            print("RATING:", review["rating"])
            print("PUBLISHER:", review["publisher"])

        # Second: Gemini AI + Google Search
        ai_result = verify_with_gemini(
            news,
            reviews
        )

        print("\n========== GEMINI VERIFICATION ==========")
        print("VERDICT:", ai_result["status"])
        print("CONFIDENCE:", ai_result["confidence"])
        print("EXPLANATION:", ai_result["explanation"])
        print("SOURCES:", ai_result["sources"])

        return jsonify({

            "status": ai_result["status"],

            "message": ai_result["explanation"],

            "confidence": ai_result["confidence"],

            "checked_claim": ai_result["checked_claim"],

            "reviews": reviews,

            "sources": ai_result["sources"],

            "fact_check_warning": warning,

            "verification_method": (
                "Google Fact Check API + Gemini AI + Google Search"
            )

        }), 200

    except ValueError as error:

        print("VERIFICATION ERROR:", repr(error))

        return jsonify({
            "error": str(error)
        }), 500

    except Exception as error:

        print("VERIFICATION ERROR:", repr(error))

        return jsonify({
            "error": (
                "AI verification failed. "
                "Please check Render application logs."
            )
        }), 500


# ------------------------------------------
# 7. HEALTH CHECK
# ------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "gemini_configured": bool(GEMINI_KEY),
        "fact_check_configured": bool(FACT_CHECK_KEY)
    })


# ------------------------------------------
# 8. RUN APPLICATION
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

import requests
import os
import json
import re

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

GEMINI_MODEL = "gemini-3.8-flash"

client = None

if GEMINI_KEY:
    client = genai.Client(
        api_key=GEMINI_KEY
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
# 4. EXTRACT JSON RESPONSE
# ------------------------------------------

def extract_json(response_text):

    if not response_text:
        raise ValueError("Empty response received from Gemini.")

    # Remove Markdown code fences if present
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        response_text,
        flags=re.IGNORECASE
    ).strip()

    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        # Find the first valid JSON object
        decoder = json.JSONDecoder()

        start = cleaned.find("{")

        if start == -1:
            raise ValueError(
                "Gemini did not return a valid JSON response."
            )

        try:
            result, _ = decoder.raw_decode(cleaned[start:])
            return result

        except json.JSONDecodeError:
            raise ValueError(
                "Unable to extract valid JSON from Gemini response."
            )


# ------------------------------------------
# 5. GEMINI AI NEWS VERIFICATION
# ------------------------------------------

def verify_with_gemini(news, fact_checks):

    if client is None:
        raise ValueError(
            "Gemini API key is missing."
        )

    fact_check_data = json.dumps(
        fact_checks,
        ensure_ascii=False
    )

    prompt = f"""
You are a professional news fact-checking assistant.

Your task is to verify the exact factual statement.

NEWS STATEMENT:
{news}

EXISTING GOOGLE FACT CHECK RESULTS:
{fact_check_data}

INSTRUCTIONS:

1. Investigate the exact news statement.
2. Use Google Search grounding to find reliable evidence.
3. Prefer official government websites and established
   news organizations.
4. Compare the claim with available evidence.
5. Do not confuse a related claim with the exact claim.
6. Do not blindly trust unrelated fact-check results.
7. Never invent evidence or source links.
8. If the information is insufficient, return UNVERIFIED.

VERDICT RULES:

TRUE:
Reliable evidence supports the central claim.

FAKE:
Reliable evidence clearly contradicts the central claim.

UNVERIFIED:
Evidence is insufficient, ambiguous, or contradictory.

Return ONLY a valid JSON object in this format:

{{
    "verdict": "TRUE",
    "confidence": 95,
    "explanation": "A short explanation based on evidence.",
    "checked_claim": "The exact claim that was checked"
}}

The verdict must be TRUE, FAKE, or UNVERIFIED.
Confidence must be a number between 0 and 100.
Do not include Markdown or additional text.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ]
        )
    )

    response_text = response.text or ""

    print("\n========== GEMINI RAW RESPONSE ==========")
    print(response_text)

    # Extract JSON
    result = extract_json(response_text)

    # Validate verdict
    verdict = str(
        result.get("verdict", "UNVERIFIED")
    ).upper().strip()

    if verdict not in ["TRUE", "FAKE", "UNVERIFIED"]:
        verdict = "UNVERIFIED"

    # Validate confidence
    try:

        confidence = float(
            result.get("confidence", 0)
        )

        confidence = max(
            0,
            min(100, confidence)
        )

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

    # Extract Google Search sources
    sources = extract_sources(response)

    # Require search evidence for definite results
    if not sources:

        verdict = "UNVERIFIED"

        explanation += (
            " No Google Search grounding sources were returned. "
            "Manual verification is recommended."
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
# 6. HOME PAGE
# ------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ------------------------------------------
# 7. CHECK NEWS ROUTE
# ------------------------------------------

@app.route("/api/check", methods=["POST"])
def check_news():

    data = request.get_json(silent=True) or {}

    news = str(data.get("news", "")).strip()

    # Validate input
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

        error_text = str(error).lower()

        if "429" in error_text or "resource_exhausted" in error_text:

            return jsonify({
                "error": (
                    "Gemini API quota exceeded. "
                    "Please try again after the quota resets."
                )
            }), 429

        return jsonify({
            "error": (
                "AI verification failed. "
                "Please check Render application logs."
            )
        }), 500


# ------------------------------------------
# 8. HEALTH CHECK
# ------------------------------------------

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "gemini_configured": bool(GEMINI_KEY),

        "fact_check_configured": bool(FACT_CHECK_KEY)

    })


# ------------------------------------------
# 9. RUN APPLICATION
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

import requests
import os
import json
import re
import time
import hashlib
import threading

# ------------------------------------------
# 1. INITIALIZATION
# ------------------------------------------

load_dotenv()

app = Flask(__name__)

FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

FACT_CHECK_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)

client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# ------------------------------------------
# 2. CACHE
# ------------------------------------------

CACHE = {}
CACHE_TTL = 3600
MAX_CACHE_SIZE = 200
CACHE_LOCK = threading.Lock()


def get_cache(news):

    key = hashlib.sha256(
        news.lower().strip().encode("utf-8")
    ).hexdigest()

    with CACHE_LOCK:

        item = CACHE.get(key)

        if item and time.time() - item["time"] < CACHE_TTL:
            print("CACHE HIT")
            return item["data"]

        if item:
            del CACHE[key]

    return None


def save_cache(news, result):

    key = hashlib.sha256(
        news.lower().strip().encode("utf-8")
    ).hexdigest()

    with CACHE_LOCK:

        if len(CACHE) >= MAX_CACHE_SIZE:
            oldest = min(
                CACHE,
                key=lambda k: CACHE[k]["time"]
            )
            del CACHE[oldest]

        CACHE[key] = {
            "time": time.time(),
            "data": result
        }


# ------------------------------------------
# 3. GOOGLE FACT CHECK API
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
            timeout=15
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

        print("FACT CHECK ERROR:", repr(error))

        return [], "Google Fact Check search is unavailable."


# ------------------------------------------
# 4. EXTRACT GOOGLE SEARCH SOURCES
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

        print("SOURCE ERROR:", repr(error))

    return sources[:8]


# ------------------------------------------
# 5. EXTRACT JSON
# ------------------------------------------

def extract_json(response_text):

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        response_text,
        flags=re.IGNORECASE
    ).replace("```", "").strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        decoder = json.JSONDecoder()

        start = cleaned.find("{")

        if start == -1:
            raise ValueError("Invalid JSON response from Gemini.")

        result, _ = decoder.raw_decode(cleaned[start:])

        return result


# ------------------------------------------
# 6. GEMINI VERIFICATION
# ------------------------------------------

def verify_with_gemini(news, fact_checks):

    if client is None:
        raise ValueError("Gemini API key is missing.")

    fact_check_data = json.dumps(
        fact_checks,
        ensure_ascii=False
    )

    prompt = f"""
You are a professional news fact-checking assistant.

Verify the exact factual claim below.

NEWS:
{news}

EXISTING FACT CHECK RESULTS:
{fact_check_data}

INSTRUCTIONS:

1. Use Google Search grounding.
2. Prefer official and reliable sources.
3. Check the exact claim, not merely related topics.
4. Never invent evidence or links.
5. Distinguish facts from opinions.
6. If evidence is insufficient, return UNVERIFIED.

VERDICTS:

TRUE: Reliable evidence supports the claim.

FAKE: Reliable evidence contradicts the claim.

UNVERIFIED: Evidence is insufficient or ambiguous.

Return ONLY JSON:

{{
    "verdict": "UNVERIFIED",
    "confidence": 0,
    "explanation": "Explain the available evidence.",
    "checked_claim": "The exact claim checked"
}}

Confidence must be between 0 and 100.
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

            temperature=0.1,

            max_output_tokens=500
        )
    )

    response_text = response.text or ""

    print("GEMINI RESPONSE:", response_text)

    result = extract_json(response_text)

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

    if not sources:

        verdict = "UNVERIFIED"

        explanation += (
            " No Google Search grounding sources were returned. "
            "Manual verification is recommended."
        )

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
# 7. HOME PAGE
# ------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ------------------------------------------
# 8. CHECK NEWS
# ------------------------------------------

@app.route("/api/check", methods=["POST"])
def check_news():

    data = request.get_json(silent=True) or {}

    news = str(data.get("news", "")).strip()

    if len(news) < 10:

        return jsonify({
            "error": "Please enter a valid news statement."
        }), 400

    if len(news) > 3000:

        return jsonify({
            "error": "Maximum 3000 characters allowed."
        }), 400

    if not GEMINI_KEY or client is None:

        return jsonify({
            "error": "Gemini API key is missing."
        }), 500

    # Check cache first

    cached = get_cache(news)

    if cached:

        cached["cached"] = True

        return jsonify(cached), 200

    try:

        reviews, warning = get_fact_checks(news)

        print("NEWS:", news)
        print("FACT CHECK REVIEWS:", len(reviews))

        ai_result = verify_with_gemini(
            news,
            reviews
        )

        result = {

            "status": ai_result["status"],

            "message": ai_result["explanation"],

            "confidence": ai_result["confidence"],

            "checked_claim": ai_result["checked_claim"],

            "reviews": reviews,

            "sources": ai_result["sources"],

            "fact_check_warning": warning,

            "verification_method": (
                "Google Fact Check API + Gemini AI + Google Search"
            ),

            "cached": False
        }

        # Save successful verification

        save_cache(news, result)

        print("VERDICT:", ai_result["status"])

        return jsonify(result), 200

    except ValueError as error:

        print("VERIFICATION ERROR:", repr(error))

        return jsonify({
            "error": str(error)
        }), 500

    except Exception as error:

        print("GEMINI ERROR:", repr(error))

        error_text = str(error).lower()

        error_code = getattr(error, "code", None)

        if (
            error_code == 429
            or "429" in error_text
            or "resource_exhausted" in error_text
        ):

            return jsonify({
                "error": (
                    "Gemini API quota or rate limit exceeded. "
                    "Please check Google AI Studio Rate Limits "
                    "and try again when quota is available."
                )
            }), 429

        if error_code == 404 or "404" in error_text:

            return jsonify({
                "error": (
                    "Gemini model or API resource not found. "
                    "Check your model name and API configuration."
                )
            }), 404

        if error_code == 400 or "400" in error_text:

            return jsonify({
                "error": (
                    "Invalid Gemini API request. "
                    "Check the request configuration."
                )
            }), 400

        return jsonify({
            "error": (
                "AI verification failed. "
                "Please check Render application logs."
            )
        }), 500


# ------------------------------------------
# 9. HEALTH CHECK
# ------------------------------------------

@app.route("/health")
def health():

    return jsonify({

        "status": "running",

        "gemini_configured": bool(GEMINI_KEY),

        "fact_check_configured": bool(FACT_CHECK_KEY),

        "model": GEMINI_MODEL

    })


# ------------------------------------------
# 10. RUN APPLICATION
# ------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

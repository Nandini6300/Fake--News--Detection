
from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

# API keys
FACT_CHECK_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Google API URLs
FACT_CHECK_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)

GEMINI_MODEL = "gemini-2.5-flash"

# Create Gemini client
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


# ------------------------------------------------
# 1. SEARCH GOOGLE FACT CHECK API
# ------------------------------------------------

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

        claims = data.get("claims", [])

        reviews = []

        for claim in claims:

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

        print("FACT CHECK API ERROR:", error)

        return [], "Google Fact Check search was unavailable."


# ------------------------------------------------
# 2. GET GEMINI GOOGLE SEARCH SOURCES
# ------------------------------------------------

def extract_sources(response):

    sources = []

    try:

        for candidate in response.candidates:

            metadata = getattr(
                candidate, "grounding_metadata", None
            )

            if not metadata:
                continue

            chunks = getattr(
                metadata, "grounding_chunks", []
            )

            for chunk in chunks:

                web = getattr(chunk, "web", None)

                if not web:
                    continue

                url = getattr(web, "uri", "")

                title = getattr(
                    web, "title", "Source"
                )

                if url and url.startswith("https://"):

                    if not any(
                        item["url"] == url for item in sources
                    ):

                        sources.append({
                            "title": title,
                            "url": url
                        })

    except Exception as error:

        print("SOURCE EXTRACTION ERROR:", error)

    return sources[:8]


# ------------------------------------------------
# 3. GEMINI AI VERIFICATION
# ------------------------------------------------

def verify_with_gemini(news, fact_checks):

    if not client:
        raise ValueError(
            "Gemini API key is missing. Check your .env file."
        )

    # Pass Google Fact Check results to Gemini
    fact_check_data = json.dumps(
        fact_checks,
        ensure_ascii=False
    )

    prompt = f"""
You are a careful professional fact-checking assistant.

Your task is to verify the user's factual news statement.

USER'S NEWS:
{news}

GOOGLE FACT CHECK API RESULTS:
{fact_check_data}

INSTRUCTIONS:

1. Independently investigate the user's exact statement.
2. Use Google Search grounding to find relevant evidence.
3. Examine reliable sources such as official government
   websites, established news organizations and primary sources.
4. Examine the Google Fact Check results, but do not blindly
   trust their ratings.
5. A fact-check article about a different claim must NOT be
   used to classify the user's statement.
6. Compare the actual meaning of the claim, not just similar
   words or names.
7. Distinguish between a false claim and a claim that lacks
   sufficient evidence.
8. Do not invent facts, sources, URLs or quotations.

VERDICT RULES:

TRUE:
Reliable evidence supports the exact statement.

FAKE:
Reliable evidence clearly contradicts the exact statement.

UNVERIFIED:
Evidence is insufficient, contradictory, ambiguous or
the statement cannot be properly checked.

If the statement contains multiple claims, consider whether
the central factual claim is supported.

Return ONLY a valid JSON object in this format:

{{
    "verdict": "TRUE",
    "confidence": 95,
    "explanation": "Short explanation of the evidence.",
    "checked_claim": "The exact claim that was checked"
}}

The verdict must be exactly TRUE, FAKE or UNVERIFIED.
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

    # Extract JSON from Gemini response
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Gemini returned an invalid verification response."
        )

    result = json.loads(
        response_text[start:end + 1]
    )

    verdict = str(
        result.get("verdict", "UNVERIFIED")
    ).upper().strip()

    if verdict not in ["TRUE", "FAKE", "UNVERIFIED"]:
        verdict = "UNVERIFIED"

    confidence = result.get("confidence", 0)

    try:
        confidence = max(0, min(100, float(confidence)))
    except (ValueError, TypeError):
        confidence = 0

    explanation = str(
        result.get(
            "explanation",
            "Insufficient explanation was returned."
        )
    )

    checked_claim = str(
        result.get("checked_claim", "")
    )

    sources = extract_sources(response)

    # Without verifiable search sources, do not
    # present the AI answer as confirmed.
    if not sources:
        verdict = "UNVERIFIED"
        explanation = (
            explanation +
            " No verifiable Google Search citations "
            "were returned, so this result needs manual review."
        )

    # A very low confidence result should not be
    # presented as a definite answer.
    if confidence < 70:
        verdict = "UNVERIFIED"

    return {
        "status": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "checked_claim": checked_claim,
        "sources": sources
    }


# ------------------------------------------------
# 4. HOME PAGE
# ------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ------------------------------------------------
# 5. NEWS VERIFICATION API
# ------------------------------------------------

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
            "error": "News statement is too long. Maximum 3000 characters."
        }), 400

    if not GEMINI_KEY:

        return jsonify({
            "error": "Gemini API key is missing. Add GEMINI_API_KEY to .env."
        }), 500

    try:

        # First: Search existing fact-check articles
        reviews, fact_check_warning = get_fact_checks(news)

        print("\n========== FACT CHECK RESULTS ==========")
        print("NEWS:", news)
        print("REVIEWS FOUND:", len(reviews))

        for review in reviews:
            print("CLAIM:", review["claim"])
            print("RATING:", review["rating"])
            print("PUBLISHER:", review["publisher"])

        # Second: Independently analyse using Gemini
        ai_result = verify_with_gemini(news, reviews)

        print("\n========== GEMINI AI RESULT ==========")
        print("VERDICT:", ai_result["status"])
        print("CONFIDENCE:", ai_result["confidence"])
        print("EXPLANATION:", ai_result["explanation"])
        print("SOURCES:", ai_result["sources"])

        # Return both API results to the website
        return jsonify({

            "status": ai_result["status"],

            "message": ai_result["explanation"],

            "confidence": ai_result["confidence"],

            "checked_claim": ai_result["checked_claim"],

            "reviews": reviews,

            "sources": ai_result["sources"],

            "fact_check_warning": fact_check_warning,

            "verification_method": (
                "Google Fact Check API + Gemini AI + Google Search"
            )

        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        print("VERIFICATION ERROR:", repr(error))

        return jsonify({
            "error": (
                "AI verification failed. Check your Gemini API key, "
                "API quota and terminal error."
            )
        }), 500


# ------------------------------------------------
# 6. RUN FLASK APPLICATION
# ------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

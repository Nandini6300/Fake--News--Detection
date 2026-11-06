# 📰 Fake News Detection Using Google Fact Check API and Gemini AI

## 📌 Project Overview

Fake News Detection is an AI-powered web application designed to identify whether a given news statement is **TRUE, FAKE, or UNVERIFIED**.

The project combines Google Fact Check API and Google Gemini AI with Google Search grounding to analyse news statements and provide evidence-based verification.

The main objective is to help users identify misleading information and promote awareness about fake news.

## 🎯 Objectives

- Identify potentially fake and misleading news.
- Verify news statements using Google Fact Check API.
- Analyse news using Gemini AI.
- Retrieve supporting evidence through Google Search.
- Display verification results with explanations and source links.

## 🚀 Features

- Simple and user-friendly web interface.
- News verification using AI.
- Google Fact Check API integration.
- Gemini AI integration.
- Google Search grounding.
- TRUE, FAKE and UNVERIFIED classifications.
- Confidence score.
- Verification explanation.
- Supporting source links.
- Online deployment.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| HTML | Webpage structure |
| CSS | Webpage styling |
| JavaScript | Frontend interactions |
| Google Fact Check API | Existing fact-check retrieval |
| Google Gemini AI | AI-based news analysis |
| Google Search grounding | Evidence discovery |
| Render | Cloud deployment |
| GitHub | Source code management |

## ⚙️ Project Workflow

1. User enters a news statement.
2. Flask receives the submitted news.
3. Google Fact Check API searches existing fact-checking articles.
4. Gemini AI analyses the news statement.
5. Google Search grounding retrieves relevant sources.
6. The application evaluates the available evidence.
7. The result is displayed as TRUE, FAKE or UNVERIFIED.

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Fake-News-Detection
```

### 3. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project directory:

```env
GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Never upload your `.env` file or expose your API keys publicly.

### 5. Run the application

```bash
py app.py
```

### 6. Open in browser

```text
http://127.0.0.1:5000/
```

## 🧪 Sample Test Cases

| News Statement | Expected Result |
|---|---|
| India became independent in 1947. | TRUE |
| Donald Trump won the 2024 US presidential election. | TRUE |
| India became independent in 1950. | FAKE |
| The Sun revolves around the Earth. | FAKE |

Actual results depend on the available evidence and API responses. Claims without sufficient evidence should be marked UNVERIFIED.

## 🌐 Deployment

The application is deployed using Render.

**Live Website:** https://fake-news-detection-iqdg.onrender.com/

## 🔮 Future Enhancements

- Support for multiple languages.
- News article URL verification.
- Image and video misinformation detection.
- Improved evidence ranking.
- More fact-checking data sources.
- Enhanced user interface.

## 👩‍💻 Project Conclusion

This project demonstrates how artificial intelligence and fact-checking technologies can work together to analyse news statements and help users identify misinformation.

It provides an interactive platform for evidence-based news verification and promotes digital media literacy.

## 📜 License

This project is developed for educational and academic purposes.

---

**Project Title:** Fake News Detection Using Google Fact Check API and Gemini AI

**Domain:** Artificial Intelligence and Web Development

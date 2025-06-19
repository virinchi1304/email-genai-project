# Email Classifier & Rewriter (Groq + FastAPI)

This is a simple GenAI project built using **FastAPI** and **Groq’s LLM** to:
- Classify emails into categories: `Work`, `Personal`, `Finance`, or `Spam`
- Rewrite emails in a selected tone: `Professional`, `Friendly`, `Apologetic`, or `Casual`

A basic HTML UI is included to interact with the API.

---

## Features

- FastAPI backend
- Groq LLM integration (LLaMA3)
- Prompt templates stored in files
- Simple front-end with HTML + JavaScript (no frameworks)
- Uses `.env` for API keys

---

## Setup Instructions

### 1. Clone the project
```bash
git clone https://github.com/virinchi1304/email-genai-project.git
cd email-genai-project

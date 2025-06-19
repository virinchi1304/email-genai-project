from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=groq_api_key)


app = FastAPI(title="EMAIL CLASSIFIER & REWRITER")


class EmailRequest(BaseModel):
    email: str

class RewriteRequest(BaseModel):
    email: str
    tone: str


def load_prompt(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Prompt file {file_path} not found.")
    


@app.post("/classify")
def classify_email(request: EmailRequest):
    try:
        template = load_prompt("prompts/classify_prompt.txt")
        prompt = template.replace("{{email}}", request.email)

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        category = response.choices[0].message.content.strip()

        import re

        match = re.search(r"(Personal|Work|Spam|Finance)", category, re.IGNORECASE)
        category = match.group(1).capitalize() if match else category

        return {"category": category}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification_Failed: {str(e)}")




@app.post("/rewrite")
def rewrite_email(request: RewriteRequest):
    try:
        template = load_prompt("prompts/rewrite_prompt.txt")
        prompt = (
            template.replace("{{tone}}", request.tone)
                    .replace("{{email}}", request.email)
        )

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
        return {"rewritten_email": rewritten}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rewriting_Failed: {str(e)}")




from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

# Mohawk College Policy Chatbot

Intro to Artificial Intelligence – Assignment 4 
Author: **Mahtabin Tushi**  
Date: **2026‑03‑27**

## Overview
This project implements a **command-line chatbot** for Mohawk College. The chatbot retrieves the most relevant policy from a local JSON dataset and answers user questions **grounded in that policy**.  

For this assignment, this version uses **Google Gemini**.

---

## Files
- `policy-chat.py` — Main Python script implementing the chatbot.
- `policies.json` — Dataset of Mohawk College policies (should be in the same folder).

---

## Setup Instructions

### 1. Install dependencies
Make sure you have Python 3.10+ installed. Install required libraries:

```bash
pip install spacy google-genai
python -m spacy download en_core_web_md
````

---

### 2. Set your Gemini API key

In **PowerShell**:

```powershell
$env:GEMINI_API_KEY="YOUR_REAL_GEMINI_API_KEY"
```

Check it is set:

```powershell
echo $env:GEMINI_API_KEY
```

---

### 3. Run the chatbot

```powershell
python policy-chat.py
```

You will see:

```
Program started successfully!
Gemini key loaded: True
Policies loaded: X
Mohawk Policy Chatbot is ready!
Type 'exit' or 'quit' to stop.
```

---

## How to Use

* Ask questions about Mohawk College policies, e.g.:

```
You: Can you summarize that?
You: Is there a policy on student conduct?
```

* Type `exit`, `quit`, or `bye` to stop.

---

## Preprocessing

* Loads `policies.json`.
* Handles missing titles or text fields.
* Normalizes whitespace and removes empty policies.
* Splits long policy text into **chunks** for semantic search.
* Precomputes **spaCy document vectors** for efficient similarity computation.

---

## Policy Retrieval

* Uses **spaCy similarity** between:

  * User question & policy titles
  * User question & text chunks
* Weighted scoring:

  * 70% title similarity
  * 30% chunk similarity
* Returns the **best matching policy**.

---

## Gemini API Prompting

* Sends the retrieved policy and user question to Gemini.
* System prompt ensures:

  * Answers are **grounded in the policy only**
  * Polite and professional responses
  * Resistant to **prompt injection**
  * Explicitly says if the policy does not contain the answer
* Supports **multi-turn conversation** with last 6 turns for context.

---

## Multi-turn Conversation

* Follow-up questions like:

  * “Can you summarize that?”
  * “Explain simply”
* Reuse the **last matched policy** to provide consistent answers.

---

## Politeness and Safety

* Detects rude language (e.g., “shit”, “fuck”) and responds politely:

```
Assistant: Please refrain from using inappropriate language. I can only answer Mohawk College policy questions.
```

* Irrelevant questions are politely declined.
* Prevents prompt injection and stays **policy-focused**.

---

## Limitations

* Semantic retrieval depends on spaCy embeddings, which may not always perfectly match nuanced policy questions.
* The chatbot only answers based on retrieved policy content; if a policy does not cover a topic, it explicitly states that.
* Chunking may split some sentences, but this is necessary for long documents.

---





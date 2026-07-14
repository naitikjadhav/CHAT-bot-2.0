# AI Chatbot for College Organization

A rule/NLP-based chatbot that answers common student queries about a college —
admissions, courses, fees, hostel, library, exams, placements, and more —
through a web chat interface.

## How it works (for your viva/report)

1. **Knowledge base (`intents.json`)** — Contains ~13 categories ("intents"),
   each with sample user phrases ("patterns") and the responses the bot should give.
2. **Text preprocessing** — Incoming user text is lowercased, punctuation
   stripped, and extra whitespace removed (`chatbot_engine.py`).
3. **Vectorization** — All training patterns are converted into TF-IDF
   (Term Frequency–Inverse Document Frequency) vectors using
   `sklearn.feature_extraction.text.TfidfVectorizer`. TF-IDF represents each
   sentence as a weighted vector of words, giving more importance to
   distinctive/rare words.
4. **Matching** — When a user sends a message, it is vectorized the same way,
   then compared against every known pattern using **cosine similarity**
   (measures the angle between two vectors — closer to 1 means more similar).
   The closest matching pattern's intent tag is selected.
5. **Confidence threshold** — If the best similarity score is below 0.25, the
   bot doesn't guess — it returns a fallback "I didn't understand" response
   instead of a wrong answer.
6. **Response** — A random response is picked from the matched intent's
   response list (keeps replies feeling natural rather than robotic/repetitive).
7. **Web layer (`app.py` + Flask)** — Exposes a `/api/chat` POST endpoint that
   the frontend calls with `fetch()`, and serves the chat UI at `/`.

This is a **retrieval-based chatbot** (as opposed to a generative one) —
a standard, well-understood architecture for FAQ/enquiry bots and a good fit
for a mini project because it needs no model training or GPU, is fully
explainable, and is easy to extend by just editing `intents.json`.

## Project structure
```
college-chatbot/
├── app.py                # Flask server + API route
├── chatbot_engine.py      # TF-IDF + cosine similarity matching logic
├── intents.json           # Knowledge base (patterns & responses)
├── requirements.txt
├── templates/
│   └── index.html         # Chat UI
└── static/
    └── style.css           # Styling
```

## Setup & run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open in browser
http://127.0.0.1:5000
```

## Testing the engine directly (no web server)

```bash
python chatbot_engine.py
```
This drops you into a command-line chat loop — useful for quickly checking
new intents without touching Flask/HTML.

## How to extend it

- **Add a new topic**: open `intents.json`, add a new object with a `tag`,
  a list of `patterns` (5-8 example phrases users might type), and a list of
  `responses`.
- **Improve matching further**: you could swap TF-IDF for spaCy/word
  embeddings, or add spell-correction preprocessing — good "future scope"
  points for your report.
- **Add memory/context**: currently each message is handled independently;
  a "follow-up question" feature would need session-based context tracking.

## Suggested report sections
- Introduction & problem statement (manual enquiry handling is slow / repetitive)
- Literature review (rule-based vs retrieval-based vs generative chatbots)
- System architecture (diagram: User → Flask API → TF-IDF matcher → intents.json → Response)
- Implementation (this codebase)
- Results/testing (sample conversations, accuracy on test queries)
- Limitations & future scope (no context memory, limited to known intents, could integrate an LLM API for open-ended questions)

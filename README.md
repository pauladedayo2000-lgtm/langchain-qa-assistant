# LangChain Q&A Assistant

A command-line assistant that answers questions using LangChain and the OpenAI API. It remembers the last 3 exchanges.

## Setup

1. Clone the repo and enter the folder:
   ```bash
   git clone https://github.com/pauladedayo2000-lgtm/langchain-qa-assistant.git
   cd langchain-qa-assistant
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your OpenAI API key.

## Run

Interactive chat:
```bash
python app.py
```

Run the five test questions (saves output to `test_results.md`):
```bash
python app.py --test
```

Type `quit` to exit.

## Test evidence

See `test_results.md` for five questions and answers, plus an empty-input test.

## Prompt and chain

The prompt has three parts: a system message, the chat history, and the user's question.
The system message tells the model to answer in plain English, stay under 120 words, and admit when it is unsure.
The chain is `prompt | ChatOpenAI | StrOutputParser`. The prompt formats the input, `gpt-4o-mini` generates the answer, and the parser returns it as plain text.
History is a list of the last 6 messages (3 question and answer pairs), passed into the prompt each time.
Temperature is 0.3 to keep answers consistent.

## Errors, risks, and limitations

**Errors handled:**
- Empty questions get a reminder instead of an API call.
- Invalid API key, rate limit or no credit, timeout, and connection failure each return a clear message.
- Any other error returns a short generic message. The program does not crash.

**Risks:**
- The model can state wrong facts with confidence. Always verify important answers.
- Questions are sent to OpenAI, so do not enter private or sensitive data.
- Every question costs tokens. A leaked key could run up charges, so keep `.env` out of Git.

**Limitations:**
- History keeps only the last 3 exchanges. Older context is forgotten.
- History is stored in memory and is lost when the program closes.
- The model has a training cutoff and cannot answer questions about recent events.
- Command-line only, with no web interface.
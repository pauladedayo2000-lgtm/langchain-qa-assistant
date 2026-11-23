import os
import sys

import openai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

MAX_TURNS = 3  # keep only the last 3 question/answer pairs

TEST_QUESTIONS = [
    "What is LangChain used for?",
    "What is the difference between a list and a tuple in Python?",
    "What is the capital of Nigeria?",
    "How many people live there?",  # follow-up, tests conversation history
    "Give me three tips for writing a good CV.",
]


def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A assistant. "
                "Answer in plain English in 120 words or fewer. "
                "If you are not sure of an answer, say so. "
                "Never invent facts.",
            ),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, timeout=30, max_retries=2)
    return prompt | llm | StrOutputParser()


def ask(chain, history, question):
    question = question.strip()
    if not question:
        return "Please type a question."

    try:
        answer = chain.invoke({"history": history, "question": question})
    except openai.AuthenticationError:
        return "Error: your API key is invalid. Check your .env file."
    except openai.RateLimitError:
        return "Error: rate limit or no credit left on your OpenAI account."
    except openai.APITimeoutError:
        return "Error: the request timed out. Try again."
    except openai.APIConnectionError:
        return "Error: could not reach OpenAI. Check your internet connection."
    except Exception as e:
        return f"Unexpected error: {type(e).__name__}"

    # Save to history only after a successful answer
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    del history[:-MAX_TURNS * 2]  # trim old messages
    return answer


def run_tests(chain):
    history = []
    lines = ["# Test Results\n"]
    for i, q in enumerate(TEST_QUESTIONS, start=1):
        a = ask(chain, history, q)
        print(f"Q{i}: {q}\nA{i}: {a}\n")
        lines.append(f"## Question {i}\n**Q:** {q}\n\n**A:** {a}\n")

    empty_reply = ask(chain, history, "   ")
    print(f"Empty input test: {empty_reply}")
    lines.append(f"## Empty input test\n**Input:** (blank spaces)\n\n**Reply:** {empty_reply}\n")

    with open("test_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved to test_results.md")


def chat(chain):
    history = []
    print("Q&A Assistant. Type 'quit' to exit.\n")
    while True:
        question = input("You: ")
        if question.strip().lower() in {"quit", "exit"}:
            break
        print(f"Assistant: {ask(chain, history, question)}\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")

    chain = build_chain()
    if "--test" in sys.argv:
        run_tests(chain)
    else:
        chat(chain)
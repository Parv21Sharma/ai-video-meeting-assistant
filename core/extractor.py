import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found in .env file!")

        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key,
        )
    return _llm


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found."
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found."
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found."
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)


def extract_all(transcript: str) -> dict[str, str]:
    if not transcript or not transcript.strip():
        return {
            "action_items": "No action items found.",
            "key_decisions": "No key decisions found.",
            "open_questions": "No open questions found.",
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_actions = executor.submit(extract_action_items, transcript)
        fut_decisions = executor.submit(extract_key_decisions, transcript)
        fut_questions = executor.submit(extract_questions, transcript)

        return {
            "action_items": fut_actions.result(),
            "key_decisions": fut_decisions.result(),
            "open_questions": fut_questions.result(),
        }
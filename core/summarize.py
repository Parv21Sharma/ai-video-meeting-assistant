
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found in .env file!")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    print("\nInitializing Gemini summarization...", flush=True)

    llm = get_llm()

    # STEP 1: Split transcript
    chunks = split_transcript(transcript)

    print(f"Transcript divided into {len(chunks)} chunk(s).", flush=True)

    # STEP 2: MAP PHASE
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a meeting transcript concisely."
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(
            f"Summarizing chunk {i + 1}/{len(chunks)}...",
            flush=True
        )

        try:
            result = map_chain.invoke({"text": chunk})
            chunk_summaries.append(result)

            print(
                f"Chunk {i + 1}/{len(chunks)} completed!",
                flush=True
            )

        except Exception as e:
            print(f"Error summarizing chunk {i + 1}: {e}", flush=True)
            raise

    print("\nMap phase completed successfully!", flush=True)

    # STEP 3: Combine partial summaries
    combined = "\n\n".join(chunk_summaries)

    print(
        f"Combined summaries length: {len(combined)} characters",
        flush=True
    )

    # STEP 4: REDUCE PHASE
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. "
            "Combine these partial summaries into one final "
            "professional meeting summary in bullet points."
        ),
        ("human", "{text}")
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    print("\nSending combined summaries to Gemini...", flush=True)

    try:
        final_summary = combined_chain.invoke(combined)

        print("Final summary received successfully!", flush=True)

        return final_summary

    except Exception as e:
        print(f"Final summarization failed: {e}", flush=True)
        raise


def generate_title(transcript: str) -> str:
    print("\nGenerating meeting title...", flush=True)

    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short "
                "professional meeting title (max 8 words). "
                "Only return the title, nothing else."
            ),
            ("human", "{text}")
        ])
        | llm
        | StrOutputParser()
    )

    try:
        title = title_chain.invoke(transcript[:2000])

        print("Meeting title generated successfully!", flush=True)

        return title

    except Exception as e:
        print(f"Title generation failed: {e}", flush=True)
        raise

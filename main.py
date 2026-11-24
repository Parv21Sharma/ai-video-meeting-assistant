import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("starting AI Video Assistant")
    t_start = time.perf_counter()

    # 1. Audio download / conversion / chunking
    t0 = time.perf_counter()
    chunks = process_input(source)
    t1 = time.perf_counter()

    # 2. Whisper transcription / translation
    lang_normalized = language.strip().lower()
    is_english = lang_normalized == "english"

    # English -> transcribe directly ("en", translate=False)
    # Hindi / Hinglish -> auto-detect language (None) and translate to English (translate=True)
    lang_code = "en" if is_english else None
    should_translate = not is_english

    transcript = transcribe_all(
        chunks,
        language=lang_code,
        translate=should_translate,
    )
    t2 = time.perf_counter()
    print(f"raw transcription (first 300 characters): {transcript[:300]}")

    # 3. Gemini title, summary, and extractions in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_title = executor.submit(generate_title, transcript)
        fut_summary = executor.submit(summarize, transcript)
        fut_actions = executor.submit(extract_action_items, transcript)
        fut_decisions = executor.submit(extract_key_decisions, transcript)
        fut_questions = executor.submit(extract_questions, transcript)

        title = fut_title.result()
        summary = fut_summary.result()
        action_item = fut_actions.result()
        decisions = fut_decisions.result()
        questions = fut_questions.result()
    t3 = time.perf_counter()

    # 4. Vector store & RAG chain setup
    rag_chain = build_rag_chain(transcript)
    t4 = time.perf_counter()

    print("\n⏱️ --- Pipeline Latency Breakdown ---")
    print(f"Audio prep / chunking: {t1 - t0:.2f}s")
    print(f"Whisper transcription:   {t2 - t1:.2f}s")
    print(f"Gemini insights:       {t3 - t2:.2f}s")
    print(f"Vector store & RAG:    {t4 - t3:.2f}s")
    print(f"Total pipeline time:   {t4 - t_start:.2f}s")
    print("------------------------------------\n")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hindi/hinglish): ").strip().lower() or "english"
    if language not in ["english", "hindi", "hinglish"]:
        language = "english"

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
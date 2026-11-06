import datetime
import json
import os
from functools import lru_cache

import gradio as gr
import groq
from dotenv import load_dotenv
from ddgs import DDGS
from groq import Groq
from pypdf import PdfReader

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit(
        "GROQ_API_KEY not found. Put this line in a .env file next to this script:\n"
        "GROQ_API_KEY=your_key_here"
    )

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-120b"

REASONING_EFFORT = "low"

MAX_OUTPUT_TOKENS = 3000

MAX_HISTORY_MESSAGES = 10
HISTORY_TOKEN_BUDGET = 3000

TURN_TOKEN_BUDGET = 6000

MAX_TOOL_ROUNDS = 4

MAX_TOOL_RESULT_CHARS = 5000
MAX_PDF_PAGES_PER_READ = 3



CHAR_FIXES = {
    0x2011: "-",   # non-breaking hyphen
    0x2010: "-",   # hyphen
    0x00A0: " ",   # non-breaking space
    0x202F: " ",   # narrow non-breaking space
    0x2009: " ",   # thin space
    0x200B: None,  # zero-width space
    0x2060: None,  # word joiner
    0xFEFF: None,  # byte order mark
}


def clean_text(text):
    return text.translate(CHAR_FIXES)


LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]


def estimate_tokens(text):
    return len(text) // 4 + 1


def estimate_messages_tokens(messages):
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            total += estimate_tokens(call["function"]["arguments"])
    return total


def limit_text(text, limit=MAX_TOOL_RESULT_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...cut off to save tokens. Ask for less text at a time.]"


def content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, (list, tuple)):
        return "".join(content_to_text(piece) for piece in content)
    return ""


def history_to_messages(history):
    messages = []
    for turn in history:
        if isinstance(turn, dict):
            text = content_to_text(turn.get("content"))
            if text.strip():
                messages.append({"role": turn["role"], "content": text})
        else:
            user_text, bot_text = turn
            if content_to_text(user_text).strip():
                messages.append({"role": "user", "content": content_to_text(user_text)})
            if content_to_text(bot_text).strip():
                messages.append({"role": "assistant", "content": content_to_text(bot_text)})
    return messages


def trim_history(messages):
    messages = messages[-MAX_HISTORY_MESSAGES:]
    while len(messages) > 1 and sum(estimate_tokens(m["content"]) for m in messages) > HISTORY_TOKEN_BUDGET:
        messages = messages[1:]
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]
    return messages


def format_wait(seconds):
    if seconds < 120:
        return f"{seconds} seconds"
    if seconds < 7200:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def rate_limit_message(error):
    seconds = None
    try:
        seconds = int(float(error.response.headers.get("retry-after")))
    except Exception:
        pass
    if seconds is None:
        return "I hit Groq's usage limit. Wait a bit and send that again."
    return (
        f"I hit Groq's usage limit. Try again in about {format_wait(seconds)}. "
        "(A short wait means the per-minute limit; a long one means the daily limit.)"
    )



def web_search(query):
    if not query.strip():
        return "No search query was given."
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        snippet = (r.get("body") or "")[:300]
        lines.append(f"- {r.get('title')}\n  {snippet}\n  {r.get('href')}")
    return limit_text("\n".join(lines))


@lru_cache(maxsize=16)
def count_pages(path):
    return len(PdfReader(path).pages)


@lru_cache(maxsize=8)
def load_pdf_pages(path):
    pages = []
    for page in PdfReader(path).pages:
        try:
            pages.append(clean_text(page.extract_text() or ""))
        except Exception:
            pages.append("")
    return tuple(pages)


def get_pdf_paths(files):
    if not files:
        return {}
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths = {}
    for f in files:
        if isinstance(f, str):
            path = f
        elif isinstance(f, dict):
            path = f.get("path")
        else:
            path = getattr(f, "name", None)
        if path and path.lower().endswith(".pdf") and os.path.exists(path):
            paths[os.path.basename(path)] = path
    return paths


def find_pdf(pdf_paths, filename):
    if filename in pdf_paths:
        return pdf_paths[filename]
    if filename:
        for name, path in pdf_paths.items():
            if filename.lower() in name.lower():
                return path
    if len(pdf_paths) == 1:
        return next(iter(pdf_paths.values()))
    return None


def read_pdf_pages(pdf_paths, filename, start_page, end_page=None):
    path = find_pdf(pdf_paths, filename)
    if path is None:
        return f"No PDF called '{filename}'. Uploaded PDFs: {', '.join(pdf_paths)}"
    pages = load_pdf_pages(path)
    total = len(pages)

    start = int(start_page)
    wanted_end = int(end_page) if end_page else start
    if start < 1 or start > total:
        return f"Page {start} doesn't exist. This PDF has {total} pages."
    end = min(wanted_end, total, start + MAX_PDF_PAGES_PER_READ - 1)

    parts = []
    for n in range(start, end + 1):
        text = pages[n - 1].strip()
        if not text:
            text = "(No readable text on this page. It may be a scanned image.)"
        parts.append(f"--- Page {n} of {total} ---\n{text}")
    result = limit_text("\n\n".join(parts))

    if end < min(wanted_end, total):
        result += (
            f"\n[Returned pages {start}-{end} only (max {MAX_PDF_PAGES_PER_READ} per read). "
            f"Read again starting at page {end + 1} for more.]"
        )
    return result


def search_pdf(pdf_paths, filename, query):
    path = find_pdf(pdf_paths, filename)
    if path is None:
        return f"No PDF called '{filename}'. Uploaded PDFs: {', '.join(pdf_paths)}"
    words = query.lower().split()
    if not words:
        return "No search words were given."

    hits = []
    for number, text in enumerate(load_pdf_pages(path), start=1):
        lowered = text.lower()
        if all(word in lowered for word in words):
            spot = lowered.find(words[0])
            snippet = " ".join(text[max(0, spot - 100): spot + 150].split())
            hits.append(f"Page {number}: ...{snippet}...")
        if len(hits) >= 8:
            break

    if not hits:
        return (
            f"No page contains all of: {query}. "
            "(If the PDF is a scan, it has no searchable text.)"
        )
    return limit_text("\n".join(hits))


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo). Use for recent events, current facts, "
            "or anything you are not confident about. Do not use for things you already know well."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Short search query"}},
            "required": ["query"],
        },
    },
}

READ_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "read_pdf_pages",
        "description": (
            f"Read the text of pages from an uploaded PDF. Returns at most {MAX_PDF_PAGES_PER_READ} pages per call. "
            "Page numbers start at 1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the uploaded PDF"},
                "start_page": {"type": "integer", "description": "First page to read"},
                "end_page": {"type": "integer", "description": "Last page to read (optional)"},
            },
            "required": ["filename", "start_page"],
        },
    },
}

SEARCH_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "search_pdf",
        "description": (
            "Find which pages of an uploaded PDF mention some words. "
            "Use this first on long PDFs, then read the pages it points to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the uploaded PDF"},
                "query": {"type": "string", "description": "Words to look for"},
            },
            "required": ["filename", "query"],
        },
    },
}


def available_tools(pdf_paths):
    tools = [WEB_SEARCH_TOOL]
    if pdf_paths:
        tools += [READ_PDF_TOOL, SEARCH_PDF_TOOL]
    return tools


def run_tool(name, args, pdf_paths):
    try:
        if name == "web_search":
            return web_search(args.get("query", ""))
        if name == "read_pdf_pages":
            return read_pdf_pages(
                pdf_paths, args.get("filename", ""), args.get("start_page", 1), args.get("end_page")
            )
        if name == "search_pdf":
            return search_pdf(pdf_paths, args.get("filename", ""), args.get("query", ""))
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"


def describe_tool_call(name, args):
    if name == "web_search":
        return f"Searching the web for: {args.get('query', '')}..."
    if name == "read_pdf_pages":
        start = args.get("start_page", "?")
        end = args.get("end_page") or start
        return f"Reading {args.get('filename', 'PDF')} (page {start} to {end})..."
    if name == "search_pdf":
        return f"Searching {args.get('filename', 'PDF')} for: {args.get('query', '')}..."
    return f"Using tool: {name}..."


def parse_arguments(raw):
    try:
        args = json.loads(raw or "{}")
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}


BASE_PROMPT = r"""You are a friendly, helpful assistant. Keep your answers clear and concise.
Today's date is {today}.

Formatting: write in Markdown. For math, use $$ ... $$ for big equations and \( ... \) for inline math. Use plain hyphens (-) and normal spaces.

Tools: answer from your own knowledge whenever you can. Use web_search only for recent events, current facts, or things you are not sure about, and mention where the information came from."""


def build_system_prompt(pdf_paths):
    prompt = BASE_PROMPT.format(today=datetime.date.today().isoformat())
    if pdf_paths:
        lines = []
        for name, path in pdf_paths.items():
            try:
                lines.append(f"- {name} ({count_pages(path)} pages)")
            except Exception:
                lines.append(f"- {name} (could not be opened)")
        prompt += (
            "\n\nThe user has uploaded these PDFs:\n" + "\n".join(lines) +
            "\nUse search_pdf and read_pdf_pages to look inside them. "
            "Never guess what a PDF says. For long PDFs, search first and then read only the pages you need."
        )
    return prompt


def assistant_message_to_dict(message):
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
            for call in message.tool_calls
        ],
    }


def ask_groq(messages, tools):
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "extra_body": {"reasoning_effort": REASONING_EFFORT},
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def chat(message, history, files):
    pdf_paths = get_pdf_paths(files)

    messages = [{"role": "system", "content": build_system_prompt(pdf_paths)}]
    messages += trim_history(history_to_messages(history))
    messages.append({"role": "user", "content": message})

    for round_number in range(MAX_TOOL_ROUNDS + 1):
        tools_allowed = (
            round_number < MAX_TOOL_ROUNDS
            and estimate_messages_tokens(messages) < TURN_TOKEN_BUDGET
        )
        tools = available_tools(pdf_paths) if tools_allowed else None

        try:
            response = ask_groq(messages, tools)
        except groq.RateLimitError as e:
            yield rate_limit_message(e)
            return
        except Exception as e:
            yield f"Something went wrong: {e}"
            return

        choice = response.choices[0]
        reply = choice.message

        if not reply.tool_calls or not tools:
            text = clean_text(reply.content or "")
            if choice.finish_reason == "length":
                text += "\n\n*(That reply hit the length limit and got cut off. Say \"continue\" to keep going.)*"
            yield text if text.strip() else "I didn't get a reply back. Try asking again."
            return

        messages.append(assistant_message_to_dict(reply))
        for call in reply.tool_calls:
            args = parse_arguments(call.function.arguments)
            yield describe_tool_call(call.function.name, args)
            result = run_tool(call.function.name, args, pdf_paths)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})


demo = gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(latex_delimiters=LATEX_DELIMITERS),
    additional_inputs=[
        gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"]),
    ],
    additional_inputs_accordion=gr.Accordion("📎 Upload PDFs", open=False),
    title="Groq Chatbot",
    description="Ask anything. I can search the web and read PDFs you upload.",
)

if __name__ == "__main__":
    demo.launch()
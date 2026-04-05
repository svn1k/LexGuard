import os, json, re, time, asyncio, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app, origins="*")

OG_OK = False
llm_client = None
og = None
WORKING_MODEL = None
_loop = None
_ready = False
_init_done = False
_init_lock = threading.Lock()

MODEL_PRIORITY = [
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_5",
    "CLAUDE_SONNET_4_6",
    "GPT_5_MINI",
]

# ── Event loop ────────────────────────────────────────────────────────────────

def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

threading.Thread(target=_start_loop, daemon=True).start()

def _run(coro, timeout=120):
    deadline = time.time() + 10
    while _loop is None and time.time() < deadline:
        time.sleep(0.1)
    if _loop is None:
        raise RuntimeError("Event loop not ready")

    async def _with_timeout():
        return await asyncio.wait_for(coro, timeout=timeout)

    return asyncio.run_coroutine_threadsafe(_with_timeout(), _loop).result(timeout=timeout + 5)

# ── OG init ───────────────────────────────────────────────────────────────────

def _init_og():
    global OG_OK, llm_client, og, _ready, _init_done, WORKING_MODEL

    with _init_lock:
        if _init_done:
            print("OG already initialized, skipping")
            return
        _init_done = True

    try:
        import opengradient as _og
        import ssl, urllib3
        og = _og
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        private_key = os.environ.get("OG_PRIVATE_KEY", "")
        if not private_key:
            raise ValueError("OG_PRIVATE_KEY not set")

        print(f"OG_PRIVATE_KEY found: {private_key[:6]}...")
        llm_client = og.LLM(private_key=private_key)

        try:
            approval = llm_client.ensure_opg_approval(min_allowance=0.1)
            print(f"OPG approval: {approval}")
        except Exception as e:
            print(f"Approval warning (continuing): {e}")

        OG_OK = True
        print("OG connected — selecting model...")
        _pick_model()

    except Exception as e:
        import traceback
        print(f"OG init failed: {e}\n{traceback.format_exc()}")
    finally:
        _ready = True
        print(f"OG ready. OG_OK={OG_OK}, model={WORKING_MODEL}")


def _pick_model():
    global WORKING_MODEL
    if not OG_OK or llm_client is None:
        return
    for name in MODEL_PRIORITY:
        if not hasattr(og.TEE_LLM, name):
            print(f"  {name}: not found in og.TEE_LLM")
            continue
        model = getattr(og.TEE_LLM, name)
        try:
            print(f"  Trying {name}...")
            result = _run(llm_client.chat(
                model=model,
                messages=[{"role": "user", "content": "Say: OK"}],
                max_tokens=5,
                temperature=0.0,
            ), timeout=90)
            raw = _extract_raw(result)
            print(f"  {name} -> {repr(raw[:80])}")
            if raw and raw.strip():
                WORKING_MODEL = model
                print(f"✓ Model selected: {name}")
                return
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("WARNING: No working model found")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_raw(result):
    if not result:
        return ""
    for attr in ['chat_output', 'completion_output', 'content', 'text', 'output']:
        val = getattr(result, attr, None)
        if val:
            if isinstance(val, dict) and val.get('content'):
                return str(val['content'])
            if isinstance(val, str) and val.strip():
                return val
    for attr in dir(result):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            if isinstance(val, str) and val.strip() and len(val) > 2:
                return val
        except:
            pass
    return ""


def _parse_json(raw):
    if not raw or not raw.strip():
        return {"error": "Empty response"}
    m = re.search(r"<JSON>(.*?)</JSON>", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception as e:
            print(f"JSON parse error in <JSON> block: {e}")
    m = re.search(r'\{[\s\S]*?"risk_score"[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {"error": "Parse failed", "raw": raw[:300]}


def call_llm(messages, retries=2):
    global WORKING_MODEL

    # Ждём инициализации OG (макс 120 сек)
    deadline = time.time() + 120
    while not _ready and time.time() < deadline:
        time.sleep(1)

    if not OG_OK or llm_client is None:
        return {"error": "OpenGradient not available"}

    # Ждём выбора модели (макс 60 сек)
    deadline = time.time() + 60
    while WORKING_MODEL is None and time.time() < deadline:
        time.sleep(2)

    if WORKING_MODEL is None:
        return {"error": "No working LLM model found — OG testnet may be down"}

    last_error = ""
    for attempt in range(retries):
        try:
            print(f"LLM attempt {attempt+1} | model: {WORKING_MODEL}")
            result = _run(llm_client.chat(
                model=WORKING_MODEL,
                messages=messages,
                max_tokens=3000,
                temperature=0.3,
            ), timeout=120)
            raw = _extract_raw(result)
            print(f"Raw (200): {repr(raw[:200])}")
            if not raw.strip():
                last_error = "Empty response"
                time.sleep(2)
                continue
            parsed = _parse_json(raw)
            if "error" in parsed:
                last_error = parsed["error"]
                time.sleep(1)
                continue
            tx = getattr(result, "transaction_hash", None) or getattr(result, "payment_hash", None)
            if tx:
                parsed["proof"] = {
                    "transaction_hash": tx,
                    "explorer_url": f"https://explorer.opengradient.ai/tx/{tx}",
                }
            return parsed
        except (asyncio.TimeoutError, TimeoutError):
            last_error = "Model timeout"
            print(f"LLM timeout attempt {attempt+1}")
        except Exception as e:
            last_error = str(e)
            print(f"LLM error attempt {attempt+1}: {e}")
            time.sleep(3)

    return {"error": f"All attempts failed: {last_error}"}


SYSTEM_PROMPT = """You are an expert legal analyst. Analyze the provided legal document and reply ONLY with valid JSON inside <JSON>...</JSON> tags. No text outside the tags.

Return this exact structure:
<JSON>
{
  "document_type": "Employment Agreement",
  "risk_score": 62,
  "clause_count": 24,
  "summary": "2-3 sentence summary of the document and overall risk assessment.",
  "risks": [
    {
      "level": "high",
      "title": "Overly broad non-compete clause",
      "clause": "Section 8.2",
      "explanation": "Clear explanation of why this clause is risky.",
      "quote": "Exact or paraphrased text from the clause",
      "recommendation": "Specific actionable advice."
    }
  ],
  "recommendations": [
    "Consult a licensed attorney before signing",
    "Request removal of the non-compete clause"
  ]
}
</JSON>

Rules:
- risk_score: 0-100 (higher = more risky)
- risks: 3-8 issues, ordered high to medium to low
- level: exactly "high", "medium", or "low"
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "og": OG_OK,
        "ready": _ready,
        "model": str(WORKING_MODEL) if WORKING_MODEL else None,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    doc_text = (data.get("doc_text") or "").strip()
    pdf_base64 = data.get("pdf_base64")
    doc_type = (data.get("doc_type") or "Legal Document").strip()

    if not doc_text and not pdf_base64:
        return jsonify({"error": "doc_text or pdf_base64 is required"}), 400

    print(f"\nAnalyzing | type: '{doc_type}' | chars: {len(doc_text)}")

    user_content = []
    if pdf_base64:
        user_content.append({
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}
        })
    if doc_text:
        user_content.append({"type": "text", "text": f"LEGAL DOCUMENT:\n\n{doc_text}"})
    user_content.append({"type": "text", "text": f"Document type: {doc_type}\n\nAnalyze and return the JSON."})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]["text"]}
    ]

    return jsonify(call_llm(messages))

# ── Ping ──────────────────────────────────────────────────────────────────────

def _ping():
    time.sleep(120)
    import urllib.request
    while True:
        time.sleep(240)
        try:
            url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:10000")
            urllib.request.urlopen(f"{url}/health", timeout=10)
            print("Self-ping OK")
        except Exception as e:
            print(f"Self-ping failed: {e}")

# ── Boot ──────────────────────────────────────────────────────────────────────

threading.Thread(target=_init_og, daemon=True).start()
threading.Thread(target=_ping, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

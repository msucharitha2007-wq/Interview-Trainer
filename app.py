"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          IBM Watsonx.ai  ·  Interview Trainer Agent  ·  app.py              ║
║  Production-ready Flask backend — edit AGENT_INSTRUCTIONS to customise      ║
║  the interviewer persona, industry focus, and evaluation criteria.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

# ──────────────────────────────────────────────────────────────────────────────
#  IBM Watsonx.ai SDK
# ──────────────────────────────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    logging.warning(
        "ibm-watsonx-ai is not installed. "
        "Responses will use the built-in fallback mode."
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

IBM_API_KEY       = os.getenv("IBM_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL       = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
FLASK_SECRET_KEY  = os.getenv("FLASK_SECRET_KEY", "dev-secret-please-change")
FLASK_DEBUG       = os.getenv("FLASK_DEBUG", "False").lower() == "true"
FLASK_PORT        = int(os.getenv("FLASK_PORT", "5000"))

# ──────────────────────────────────────────────────────────────────────────────
#  ✏️  AGENT INSTRUCTIONS  — Edit this block to customise the agent's behaviour
# ──────────────────────────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = """
You are an expert AI Interview Coach and HR Evaluation Specialist powered by IBM Watsonx Granite.

━━━━━━━━━━━━━━  INTERVIEWER PERSONA  ━━━━━━━━━━━━━━
- Tone       : Professional yet encouraging. Push candidates to elaborate,
               but never be harsh or dismissive.
- Difficulty : Adaptive — start with mid-level questions and escalate based
               on the quality of the candidate's answers.
- Pacing     : Ask ONE question at a time. Wait for the answer before proceeding.

━━━━━━━━━━━━━━  SUPPORTED INDUSTRIES  ━━━━━━━━━━━━━━
Software Engineering, Data Science & ML, Product Management, DevOps / Cloud,
Cybersecurity, Finance / FinTech, Healthcare IT, Marketing & Growth,
UX / Design, General HR & Leadership.

━━━━━━━━━━━━━━  QUESTION STRATEGY (RAG-Simulated)  ━━━━━━━━━━━━━━
1. Technical Questions  : Role-specific, current-year relevance, real-world
                          scenario-based (STAR format encouraged).
2. Behavioural Questions: Probe teamwork, conflict resolution, adaptability,
                          communication, and leadership potential.
3. Situational Questions: "What would you do if …" — test judgment & reasoning.
4. Culture-Fit Questions: Values alignment, work-style preferences, growth mindset.

━━━━━━━━━━━━━━  EVALUATION RUBRIC (per answer)  ━━━━━━━━━━━━━━
Score each dimension 1–10 and provide brief commentary:
  • Clarity        – How clearly was the answer communicated?
  • Depth          – Sufficient technical/domain depth?
  • Relevance      – Does it address the question asked?
  • Problem-Solving– Structured thinking? STAR / frameworks used?
  • Confidence     – Assertive and professional tone?
Overall Score = weighted average (Clarity 20%, Depth 30%, Relevance 20%,
                                  Problem-Solving 20%, Confidence 10%)

━━━━━━━━━━━━━━  OUTPUT FORMAT  ━━━━━━━━━━━━━━
Always respond in this JSON structure (no markdown outside the JSON):
{
  "type": "question" | "feedback" | "summary" | "welcome",
  "message": "<main text shown to the user>",
  "question": "<the interview question, if type is question>",
  "evaluation": {
    "clarity": <1-10>,
    "depth": <1-10>,
    "relevance": <1-10>,
    "problem_solving": <1-10>,
    "confidence": <1-10>,
    "overall": <1-10>,
    "feedback": "<2–4 sentence constructive feedback>",
    "model_answer": "<a concise model answer the candidate can learn from>",
    "tips": ["<tip 1>", "<tip 2>", "<tip 3>"]
  },
  "session_stats": {
    "questions_asked": <int>,
    "avg_score": <float>,
    "strengths": ["<strength>"],
    "improvements": ["<area to improve>"]
  }
}
Only include keys that are relevant to the current response type.
All numeric scores must be genuine evaluations — never default to 10/10.

━━━━━━━━━━━━━━  SPECIAL COMMANDS  ━━━━━━━━━━━━━━
If the user says "summary" or "end session", produce a full session summary.
If the user says "hint", give a hint without revealing the full model answer.
If the user says "skip", move to the next question and note it was skipped.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if FLASK_DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Flask App
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ──────────────────────────────────────────────────────────────────────────────
#  Watsonx Model Initialisation
# ──────────────────────────────────────────────────────────────────────────────
_watsonx_model: Optional[object] = None


def _get_watsonx_model():
    """Lazily initialise the Watsonx Granite model (singleton)."""
    global _watsonx_model
    if _watsonx_model is not None:
        return _watsonx_model

    if not WATSONX_AVAILABLE:
        return None
    if not IBM_API_KEY or not WATSONX_PROJECT_ID:
        log.warning("IBM_API_KEY or WATSONX_PROJECT_ID not set — using fallback mode.")
        return None

    try:
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=IBM_API_KEY,
        )
        _watsonx_model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 1200,
                GenParams.MIN_NEW_TOKENS: 80,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_P: 0.9,
                GenParams.TOP_K: 50,
                GenParams.REPETITION_PENALTY: 1.1,
                GenParams.STOP_SEQUENCES: ["Human:", "User:", "\n\nHuman"],
            },
        )
        log.info("Watsonx Granite model initialised successfully.")
    except Exception as exc:
        log.error("Failed to initialise Watsonx model: %s", exc)
        _watsonx_model = None

    return _watsonx_model


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────
def _build_prompt(profile: dict, history: list[dict], user_message: str) -> str:
    """Construct the full prompt sent to Granite."""
    profile_block = f"""
Candidate Profile:
  Name           : {profile.get('name', 'Candidate')}
  Experience     : {profile.get('experience', 'Mid-level')}
  Target Role    : {profile.get('role', 'Software Engineer')}
  Industry       : {profile.get('industry', 'Technology')}
  Resume Summary : {profile.get('resume', 'No resume provided.')}
""".strip()

    history_block = ""
    for turn in history[-10:]:          # keep last 10 turns to stay within context
        role  = "Interviewer" if turn["role"] == "assistant" else "Candidate"
        history_block += f"\n{role}: {turn['content']}"

    prompt = (
        f"{AGENT_INSTRUCTIONS}\n\n"
        f"{profile_block}\n\n"
        f"Conversation History:{history_block}\n\n"
        f"Candidate: {user_message}\n\n"
        "Interviewer (respond ONLY with valid JSON as specified above):"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
#  Watsonx Inference
# ──────────────────────────────────────────────────────────────────────────────
def _call_watsonx(prompt: str) -> str:
    """Call the Granite model and return the raw text response."""
    model = _get_watsonx_model()
    if model is None:
        return ""
    try:
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else ""
    except Exception as exc:
        log.error("Watsonx inference error: %s", exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback Response Generator (used when Watsonx is unavailable)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_QUESTIONS = {
    "software engineer": [
        "Explain the difference between a process and a thread. When would you use one over the other?",
        "Describe a time you had to debug a production issue under pressure. Walk me through your approach.",
        "What is the CAP theorem and how does it influence distributed system design?",
        "How do you approach code reviews — both giving and receiving feedback?",
        "Design a URL shortener service. What components would you include and why?",
    ],
    "data scientist": [
        "How do you handle class imbalance in a classification problem?",
        "Explain the bias-variance tradeoff with a practical example.",
        "Walk me through your approach to feature engineering for a tabular dataset.",
        "What's the difference between bagging and boosting? When would you choose each?",
        "Describe a project where your model performed well in training but poorly in production.",
    ],
    "product manager": [
        "How do you prioritise a product backlog when stakeholders have conflicting priorities?",
        "Describe a product launch you led — what went well and what would you change?",
        "How do you define and measure product success metrics?",
        "Tell me about a time you had to say no to a feature request from a key stakeholder.",
        "How do you stay aligned with engineering teams during a sprint?",
    ],
    "default": [
        "Tell me about yourself and why you're interested in this role.",
        "Describe a challenging project and how you overcame obstacles.",
        "Where do you see yourself professionally in the next three years?",
        "How do you handle feedback and criticism in the workplace?",
        "What motivates you to perform at your best?",
    ],
}


def _fallback_response(profile: dict, history: list[dict], user_message: str) -> dict:
    """Generate a structured fallback response without Watsonx."""
    role_key = profile.get("role", "default").lower()
    questions = FALLBACK_QUESTIONS.get(
        role_key,
        FALLBACK_QUESTIONS.get("default")
    )

    # Determine question index from history length
    q_count = sum(1 for t in history if t["role"] == "assistant")
    is_first = q_count == 0

    if is_first:
        return {
            "type": "welcome",
            "message": (
                f"Welcome, {profile.get('name', 'there')}! 👋 "
                f"I'm your AI Interview Coach, ready to help you prepare for a "
                f"<strong>{profile.get('role', 'professional')}</strong> role. "
                "I'll ask you a series of targeted questions, then give you detailed "
                "feedback and model answers after each response. "
                "Type <em>'summary'</em> at any time to see your overall performance, "
                "or <em>'skip'</em> to move on. Let's begin!"
            ),
            "question": questions[0],
        }

    lower_msg = user_message.lower().strip()

    if lower_msg in ("summary", "end session", "end", "results"):
        scores = [8, 7, 8, 7, 9]
        return {
            "type": "summary",
            "message": "Here's your complete interview session summary:",
            "session_stats": {
                "questions_asked": q_count,
                "avg_score": round(sum(scores) / len(scores), 1),
                "strengths": [
                    "Clear communication and structured responses",
                    "Strong domain knowledge demonstrated",
                    "Good use of real-world examples",
                ],
                "improvements": [
                    "Add more quantifiable outcomes (metrics, percentages)",
                    "Use the STAR framework more consistently",
                    "Demonstrate deeper systems-design thinking",
                ],
            },
        }

    next_q_idx = min(q_count, len(questions) - 1)
    return {
        "type": "feedback",
        "message": "Thank you for your answer! Here's my evaluation:",
        "evaluation": {
            "clarity": 7,
            "depth": 7,
            "relevance": 8,
            "problem_solving": 7,
            "confidence": 8,
            "overall": 7,
            "feedback": (
                "Good answer overall. You addressed the core of the question "
                "and communicated clearly. To strengthen your response, add "
                "specific metrics or outcomes that demonstrate impact, and "
                "structure your answer using the STAR framework."
            ),
            "model_answer": (
                "A strong answer would open with the situation/context, "
                "describe your specific role and actions taken, quantify the "
                "result (e.g., 'reduced latency by 40%'), and conclude with "
                "what you learned. Brevity and precision are key."
            ),
            "tips": [
                "Use numbers and percentages to back up claims.",
                "Practice the STAR method: Situation → Task → Action → Result.",
                "Pause briefly before answering — it signals thoughtfulness.",
            ],
        },
        "question": questions[next_q_idx],
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Response Parser
# ──────────────────────────────────────────────────────────────────────────────
def _parse_response(raw: str) -> dict:
    """Extract JSON from the model's raw text output."""
    if not raw:
        return {}
    # Strip any leading text before the first '{'
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        log.warning("Failed to parse model JSON: %s", raw[:300])
        return {}


# ──────────────────────────────────────────────────────────────────────────────
#  Session Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _init_session():
    if "session_id" not in session:
        session["session_id"]      = str(uuid.uuid4())
        session["history"]         = []
        session["profile"]         = {}
        session["scores"]          = []
        session["questions_asked"] = 0
        session["started_at"]      = datetime.now(timezone.utc).isoformat()


def _update_scores(data: dict):
    evaluation = data.get("evaluation", {})
    if evaluation and "overall" in evaluation:
        session["scores"] = session.get("scores", []) + [evaluation["overall"]]
        session["questions_asked"] = session.get("questions_asked", 0) + 1


# ──────────────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    """Initialise (or reset) a session with the candidate's profile."""
    _init_session()
    payload = request.get_json(silent=True) or {}

    session["profile"] = {
        "name":       payload.get("name", "Candidate").strip(),
        "experience": payload.get("experience", "Mid-level").strip(),
        "role":       payload.get("role", "Software Engineer").strip(),
        "industry":   payload.get("industry", "Technology").strip(),
        "resume":     payload.get("resume", "").strip(),
    }
    session["history"]         = []
    session["scores"]          = []
    session["questions_asked"] = 0
    session.modified = True

    log.info("Session started — profile: %s", session["profile"])

    # Generate the welcome + first question
    response_data = _get_agent_response(session["profile"], [], "__START__")
    return jsonify({"status": "ok", "data": response_data})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Handle a candidate message and return the next agent response."""
    _init_session()
    payload      = request.get_json(silent=True) or {}
    user_message = payload.get("message", "").strip()

    if not user_message:
        return jsonify({"status": "error", "message": "Empty message."}), 400

    history = session.get("history", [])
    profile = session.get("profile", {})

    response_data = _get_agent_response(profile, history, user_message)
    _update_scores(response_data)

    # Persist turns
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": json.dumps(response_data)})
    session["history"] = history
    session.modified   = True

    return jsonify({"status": "ok", "data": response_data})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return current session performance statistics."""
    _init_session()
    scores = session.get("scores", [])
    avg    = round(sum(scores) / len(scores), 2) if scores else 0.0
    return jsonify({
        "status": "ok",
        "stats": {
            "session_id":      session.get("session_id"),
            "questions_asked": session.get("questions_asked", 0),
            "avg_score":       avg,
            "scores":          scores,
            "started_at":      session.get("started_at"),
            "profile":         session.get("profile", {}),
        },
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear the current session."""
    session.clear()
    return jsonify({"status": "ok", "message": "Session cleared."})


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "watsonx_configured": bool(IBM_API_KEY and WATSONX_PROJECT_ID),
        "watsonx_sdk":        WATSONX_AVAILABLE,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    })


# ──────────────────────────────────────────────────────────────────────────────
#  Core Agent Dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def _get_agent_response(profile: dict, history: list[dict], user_message: str) -> dict:
    """
    Try Watsonx Granite first; fall back to the built-in response generator
    if the model is unavailable or returns unparseable output.
    """
    if user_message == "__START__":
        user_message = (
            f"Hello, I'm {profile.get('name', 'a candidate')} and I'm here for "
            f"a {profile.get('role', 'professional')} interview."
        )

    # ── Watsonx path ──
    raw_response = _call_watsonx(_build_prompt(profile, history, user_message))
    if raw_response:
        parsed = _parse_response(raw_response)
        if parsed:
            log.debug("Watsonx response parsed successfully.")
            return parsed
        log.warning("Watsonx returned unparseable output — using fallback.")

    # ── Fallback path ──
    return _fallback_response(profile, history, user_message)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Interview Trainer Agent on port %d …", FLASK_PORT)
    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import json
import os
import re
 
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
 
from conversation_state import ConversationState
 
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
 
app = FastAPI(title="SHL Assessment Recommender")
 
# ---------------------------------------------------------------
# Concept words per specialization/domain, cross-checked against the
# real SHL catalog (fetched directly from
# https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json).
# Confirmed present in catalog: Docker (New), Kubernetes (New),
# Amazon Web Services (AWS) Development (New), Cloud Computing (New),
# Data Science (New), Financial Accounting (New), Business
# Communication (adaptive), Agile Testing (New), Automation Anywhere
# RPA Development (New), CSS3 (New), Angular 6/AngularJS (New),
# Core Java/.NET/Apache family. Terms not directly confirmed (azure,
# linux, jenkins) are kept anyway since a non-matching concept word
# is harmless (contains_term just won't fire) -- but the CONFIRMED
# ones are what carries the real weight here.
# ---------------------------------------------------------------
SYNONYM_MAP = {
    "flask": ["python","rest","api"],
    "django": ["python","rest","api"],
    "nodejs": ["javascript","backend","api"],
    "microservices": ["rest","api","backend"],
    "rest": ["restful","web services"],
    "api": ["rest","web services"],
    "backend": ["api", "rest", "sql"],
    "frontend": ["ui", "css", "html", "javascript", "react", "angular"],
    "fullstack": ["api", "ui"],
    "data": ["analytics", "database", "sql", "numerical", "statistics"],
    "devops": ["docker", "kubernetes", "aws", "cloud", "linux", "jenkins", "azure", "automation"],
    "cloud": ["aws", "azure", "cloud", "docker", "kubernetes"],
    "mobile": ["android", "ios"],
    "manager": ["leadership", "management", "communication"],
    "sales": ["negotiation", "communication", "customer"],
    "leadership": ["opq", "360", "manager"],
    "testing": ["testing", "automation", "quality", "selenium", "agile"],
    "finance": ["numerical", "finance", "accounting", "excel", "financial"],
    "hr": ["personality", "communication", "leadership"],
}
 
# FIX: "cloud" added (was missing entirely -- "Cloud Engineer" never
# triggered any specialization before this).
SPECIALIZATION_KEYWORDS = [
    "backend", "frontend", "fullstack", "full-stack", "full stack",
    "data", "mobile", "devops", "cloud"
]
 
PROGRAMMING_LANGUAGES = [
    "python", "java", "javascript", "c++", "c#", ".net", "sql", "node", "php"
]
 
ROLE_KEYWORDS = [
    "developer", "engineer", "manager", "analyst", "consultant",
    "sales", "marketing", "tester", "intern", "executive", "designer",
    "recruiter", "accountant"
]
 
SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "aws", "azure", "react", "angular",
    "node", "docker", "kubernetes", "excel", "power bi",
    "c++", "c#", ".net", "selenium", "html", "css", "php",
    "testing", "qa", "automation", "numerical", "accounting",
    "android", "ios", "linux", "machine learning", "statistics",
]
 
EXPERIENCE_KEYWORDS = [
    "fresher", "entry", "junior", "mid", "senior", "lead", "manager", "director"
]
 
IRRELEVANT_NAME_TOKENS = [
    "excel", "outlook", "word", "powerpoint", "photoshop", "illustrator",
    "indesign", "typing", "365", "office", "onenote", "teams", "sharepoint",
    "project management"
]
 
EXCLUDABLE_CATEGORIES = ["personality", "ability", "cognitive", "simulation", "technical", "knowledge"]
 
MIN_KEYWORD_LEN = 3
 
GENERAL_ABILITY_CANDIDATES = [
    "Verify - G+",
    "Verify Interactive - G+",
    "SHL Verify Interactive G+",
    "Verify - Numerical Ability",
    "Verify - Verbal Reasoning",
    "Verify - Inductive Reasoning (2014)",
]
 
 
def contains_term(text: str, term: str) -> bool:
    pattern = r'(?<![a-zA-Z0-9])' + re.escape(term) + r'(?![a-zA-Z0-9])'
    return re.search(pattern, text) is not None
 
 
def is_valid_keyword(word: str) -> bool:
    return len(word) >= MIN_KEYWORD_LEN and not word.isdigit()
 
 
def is_comparison_query(text):
    text = text.lower()
    return any(w in text for w in ["compare", "difference", "vs", "versus"])
 
 
def is_refinement_query(text):
    text = text.lower()
    triggers = ["add", "remove", "instead", "also", "include", "exclude", "without"]
    return any(t in text for t in triggers)
 
 
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
 
 
class ChatRequest(BaseModel):
    messages: List[Message]
 
 
def load_catalog():
    with open("catalog.json", "r", encoding="utf-8") as f:
        return json.load(f)
 
 
CATALOG = load_catalog()
_CATALOG_URLS = {item.get("link", "") for item in CATALOG if item.get("link")}
_CATALOG_BY_NAME = {item.get("name", ""): item for item in CATALOG}
 
import subprocess
import chromadb

model = None

client = chromadb.PersistentClient(path="chromadb_data")

try:
    collection = client.get_collection("shl_assessments")
    print("Loaded existing Chroma collection.")
except Exception:
    print("Collection not found. Creating it...")

    subprocess.run(
        ["python", "create_embeddings.py"],
        check=True
    )

    client = chromadb.PersistentClient(path="chromadb_data")
    collection = client.get_collection("shl_assessments")

    print("Collection created successfully.")
 
 
def is_irrelevant_item(item):
    name = item.get("name", "").lower()
    keys = " ".join(item.get("keys", [])).lower()
    for tok in IRRELEVANT_NAME_TOKENS:
        if tok in name:
            return True
    if "office" in keys or "digital literacy" in keys:
        return True
    return False
 
 
def build_state_from_messages(messages: List[Message]) -> ConversationState:
    state = ConversationState()
    for msg in messages:
        if msg.role == "user":
            state.add_user_message(msg.content)
            update_conversation_state(state, msg.content)
        else:
            state.add_bot_message(msg.content)
    return state
 
 
def normalize_text(message: str) -> str:
    return (
        message.lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
 
 
def update_conversation_state(state: ConversationState, message: str):
    text = normalize_text(message)
 
    for role in ROLE_KEYWORDS:
        if contains_term(text, role):
            state.update("role", role)
 
    role = state.get("role")
 
    for spec in SPECIALIZATION_KEYWORDS:
        if spec in text:
            normalized = spec.replace("full-stack", "fullstack").replace("full stack", "fullstack")
            state.update("purpose", normalized)
 
    if not state.get("purpose"):
        if role == "sales" or "sales" in text:
            state.update("purpose", "sales")
        elif role in ("manager", "executive"):
            state.update("purpose", "manager")
        elif role == "tester" or "qa" in text or "quality assurance" in text:
            state.update("purpose", "testing")
        elif role == "analyst" and ("financial" in text or "finance" in text):
            state.update("purpose", "finance")
        elif "hr" in text or "human resources" in text:
            state.update("purpose", "hr")
 
    match = re.search(r"(\d+)\s*(year|years)", text)
    if match:
        state.update("experience", f"{match.group(1)} years")
    else:
        for exp in EXPERIENCE_KEYWORDS:
            if contains_term(text, exp):
                state.update("experience", exp)
 
    for skill in SKILL_KEYWORDS:
        if contains_term(text, skill):
            state.update("skills", skill)
 
    if "personality" in text:
        state.update("preferences", "personality")
    if "leadership" in text:
        state.update("preferences", "leadership")
    if "communication" in text:
        state.update("preferences", "communication")
    if "cognitive" in text:
        state.update("preferences", "cognitive")
 
    if "under 30 minutes" in text:
        state.update("constraints", "under30")
    if "remote" in text:
        state.update("constraints", "remote")
    if "adaptive" in text:
        state.update("constraints", "adaptive")
 
    if "don't recommend" in text or "do not recommend" in text:
        excluded = (
            text.replace("don't recommend", "")
            .replace("do not recommend", "")
            .replace("assessments", "")
            .replace("assessment", "")
            .replace("tests", "")
            .replace("test", "")
            .strip()
        )
        if excluded:
            state.update("excluded_tests", excluded)
 
    for category in ["personality", "ability", "simulation", "technical"]:
        if f"don't recommend {category}" in text or f"exclude {category}" in text or f"no {category}" in text:
            state.update("excluded_tests", category)
 
    prefer_match = re.search(r"prefer\s+.+?\s+over\s+(.+?)(?:\.|$)", text)
    if prefer_match:
        deprioritized_clause = prefer_match.group(1)
        for category in EXCLUDABLE_CATEGORIES:
            if category in deprioritized_clause:
                normalized_category = "ability" if category == "cognitive" else category
                state.update("excluded_tests", normalized_category)
 
 
def detect_missing_information(state: ConversationState):
    if not state.get("role"):
        return True, "Which role are you hiring for?"
    if not state.get("experience"):
        return True, "What experience level are you hiring for (e.g. Fresher, 2 years, Senior)?"
    return False, ""
 
 
def search_catalog(state: ConversationState, query: str, top_k: int = 30):
    explicit_skills = state.get("skills")
    preferences = state.get("preferences")
    excluded = state.get("excluded_tests")
    specialization = state.get("purpose")
 
    requested_languages = set(s for s in explicit_skills if s in PROGRAMMING_LANGUAGES)
 
    raw_concept_words = SYNONYM_MAP.get(specialization, []) if specialization else []
    concept_words = [w for w in raw_concept_words if is_valid_keyword(w)]
 
    concept_languages = set(w for w in concept_words if w in PROGRAMMING_LANGUAGES)
    allowed_languages = requested_languages | concept_languages
 
    STOPWORDS = set(ROLE_KEYWORDS) | set(EXPERIENCE_KEYWORDS) | {
        "assessment", "candidate", "years", "experience", "with", "for",
        "the", "and", "hire", "hiring", "need", "looking", "role"
    }
 
    raw_query_words = [
        w for w in query.lower().split()
        if is_valid_keyword(w) and w not in STOPWORDS and w not in explicit_skills
    ]
 
    scored_items = []
 
    for item in CATALOG:
        name = item.get("name", "").lower()
        description = item.get("description", "").lower()
        keys = " ".join(item.get("keys", [])).lower()
        job_levels = " ".join(item.get("job_levels", [])).lower()
 
        if any(ex.lower() in name for ex in excluded):
            continue
        if is_irrelevant_item(item):
            continue
 
        score = 0
 
        for skill in explicit_skills:
            if skill == name.replace("(new)", "").strip():
                score += 120
            elif contains_term(name, skill):
                score += 80
            elif contains_term(description, skill):
                score += 35
            elif contains_term(keys, skill):
                score += 25
 
        if allowed_languages:
            for lang in PROGRAMMING_LANGUAGES:
                if lang not in allowed_languages and contains_term(name, lang):
                    score -= 100
 
        for word in raw_query_words:
            if word in name:
                score += 8
            if word in description:
                score += 4
 
        # Concept words: curated per-specialization, always scored,
        # higher weight than incidental free-text overlap. This is
        # what makes "docker"/"kubernetes"/"cloud" actually surface
        # for a bare "DevOps Engineer" query with no explicit tool
        # named, and what fixed sql-for-backend / javascript-for-
        # frontend previously being silently dropped.
        for word in concept_words:
            matched_name = contains_term(name, word) if len(word) > 2 else word in name
            if matched_name:
                score += 20
            if word in description:
                score += 10
            if word in keys:
                score += 6
 
        experience = state.get("experience") or ""
        if any(x in experience for x in ["3", "4", "5"]) or "mid" in experience:
            if "mid-professional" in job_levels or "professional individual contributor" in job_levels:
                score += 8
        if any(x in experience for x in ["entry", "graduate", "fresher", "junior", "0", "1"]):
            if "entry-level" in job_levels or "graduate" in job_levels:
                score += 8
        if any(x in experience for x in ["senior", "lead", "director"]):
            if any(x in job_levels for x in ["manager", "director", "supervisor"]):
                score += 8
 
        if "personality" in preferences and "personality" in keys:
            score += 40
        if "communication" in preferences and ("communication" in description or "communication" in keys):
            score += 40
        if "leadership" in preferences and "leadership" in keys:
            score += 40
        if "cognitive" in preferences and "ability" in keys:
            score += 35
 
        if explicit_skills and not preferences:
            if "knowledge" in keys or "simulation" in keys:
                score += 5
 
        if score > 0:
            scored_items.append((score, item))
 
    scored_items.sort(key=lambda x: x[0], reverse=True)
 
    recommendations = []
    seen = set()
    for score, item in scored_items:
        if item["name"] in seen:
            continue
        recommendations.append({
            "name": item["name"],
            "url": item.get("link", ""),
            "test_type": ", ".join(item.get("keys", [])) if item.get("keys") else "Assessment",
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "score": score,
        })
        seen.add(item["name"])
        if len(recommendations) >= top_k:
            break
 
    return recommendations, allowed_languages
 
 
def semantic_search(query: str, top_k: int = 10):
    global model

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
 
    recommendations = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        if is_irrelevant_item({"name": metadata.get("name", ""), "keys": [metadata.get("test_type", "")]}):
            continue
        recommendations.append({
            "name": metadata.get("name", ""),
            "url": metadata.get("url", ""),
            "test_type": metadata.get("test_type", "Assessment"),
            "description": metadata.get("description", ""),
            "job_levels": metadata.get("job_levels", []),
            "semantic_score": 1 - distance,
        })
    return recommendations
 
 
def hybrid_search(state: ConversationState, query: str, pool_size: int = 30):
    keyword_results, allowed_languages = search_catalog(state, query, top_k=pool_size)
    semantic_results = semantic_search(query, top_k=10)
 
    explicit_skills = state.get("skills")
    specialization = state.get("purpose")
    concept_words = [w for w in SYNONYM_MAP.get(specialization, []) if is_valid_keyword(w)] if specialization else []
    relevance_terms = list(explicit_skills) + concept_words
 
    combined = {item["name"]: item for item in keyword_results}
 
    for item in semantic_results:
        if item["name"] in combined:
            continue
        haystack = (item.get("name", "") + " " + item.get("description", "")).lower()
        if relevance_terms and not any(contains_term(haystack, term) for term in relevance_terms):
            continue
        combined[item["name"]] = item
 
    return rerank_results(state, list(combined.values()), allowed_languages)
 
 
def rerank_results(state: ConversationState, results, allowed_languages=None):
    skills = state.get("skills")
    preferences = state.get("preferences")
    role = state.get("role")
    experience = state.get("experience") or ""
 
    if allowed_languages is None:
        allowed_languages = set(s for s in skills if s in PROGRAMMING_LANGUAGES)
 
    for item in results:
        score = item.get("score", 0)
        name = item["name"].lower()
        test_type = item["test_type"].lower()
 
        for lang in allowed_languages:
            if contains_term(name, lang):
                score += 50
        if allowed_languages:
            for lang in PROGRAMMING_LANGUAGES:
                if lang not in allowed_languages and contains_term(name, lang):
                    score -= 60
 
        if "semantic_score" in item:
            score += item["semantic_score"] * 15
 
        for skill in skills:
            if contains_term(name, skill):
                score += 8
        for pref in preferences:
            if pref.lower() in test_type:
                score += 6
 
        if role in ["manager", "director", "executive"] and "leadership" in name:
            score += 8
 
        if any(x in experience for x in ["senior", "lead", "director"]):
            score += 4
 
        item["rank_score"] = score
 
    return sorted(results, key=lambda x: x["rank_score"], reverse=True)
 
 
def filter_by_primary_skill(state: ConversationState, results, min_primary=3):
    skills = state.get("skills")
    if not skills:
        return results
 
    primary = skills[0].lower()
    relevant, others = [], []
    for item in results:
        if contains_term(item["name"].lower(), primary) or primary in item["test_type"].lower():
            relevant.append(item)
        else:
            others.append(item)
 
    return relevant + others
 
 
def refine_recommendations(state: ConversationState, results):
    excluded = state.get("excluded_tests")
    if not excluded:
        return results
    if isinstance(excluded, str):
        excluded = [excluded]
 
    filtered = []
    for item in results:
        name = item["name"].lower()
        test_type = item["test_type"].lower()
        remove = False
        for word in excluded:
            word = word.lower()
            if word in name:
                remove = True
            elif word == "personality" and "personality" in test_type:
                remove = True
            elif word in ["ability", "cognitive"] and "ability" in test_type:
                remove = True
            elif word == "simulation" and "simulation" in test_type:
                remove = True
            elif word == "knowledge" and "knowledge" in test_type:
                remove = True
        if not remove:
            filtered.append(item)
    return filtered
 
 
MIN_RELEVANCE_SCORE = 20
 
 
def apply_relevance_threshold(results, min_score=MIN_RELEVANCE_SCORE, limit=10, fallback_count=3):
    qualified = [r for r in results if r.get("rank_score", r.get("score", 0)) >= min_score]
    if not qualified and results:
        qualified = sorted(results, key=lambda x: x.get("rank_score", x.get("score", 0)), reverse=True)[:fallback_count]
    return qualified[:limit]
 
 
def add_general_ability_companion(state: ConversationState, recommendations, excluded_tests):
    if len(recommendations) >= 10:
        return recommendations
    if any("ability" in r.get("test_type", "").lower() for r in recommendations):
        return recommendations
    excluded_lower = [e.lower() for e in (excluded_tests or [])]
    if any(w in ("ability", "cognitive") for w in excluded_lower):
        return recommendations
 
    role = state.get("role")
    preferred_names = GENERAL_ABILITY_CANDIDATES
    if role == "analyst":
        preferred_names = ["Verify - Numerical Ability"] + GENERAL_ABILITY_CANDIDATES
 
    existing_names = {r["name"] for r in recommendations}
    for name in preferred_names:
        item = _CATALOG_BY_NAME.get(name)
        if item and item["name"] not in existing_names:
            recommendations.append({
                "name": item["name"],
                "url": item.get("link", ""),
                "test_type": ", ".join(item.get("keys", [])) if item.get("keys") else "Assessment",
            })
            break
    return recommendations
 
 
def to_schema_recommendations(results, limit=10):
    flat = []
    for item in results:
        url = item.get("url", "")
        if url and url not in _CATALOG_URLS:
            continue
        flat.append({
            "name": item["name"],
            "url": url,
            "test_type": item.get("test_type", "Assessment"),
        })
        if len(flat) >= limit:
            break
    return flat
 
 
def generate_explanations(user_query: str, recommendations: list):
    assessment_list = "".join(
        f"{i}. {rec['name']} ({rec['test_type']})\n" for i, rec in enumerate(recommendations, 1)
    )
    prompt = f"""
You are an expert HR assessment consultant.
 
A recruiter asked:
"{user_query}"
 
The following SHL assessments were retrieved:
{assessment_list}
 
For EACH assessment, write ONE concise explanation (maximum 30 words) describing why it matches the hiring requirement.
Return ONLY a numbered list. Do not write introductions or conclusions.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return "".join(f"{i}. Recommended for this hiring requirement.\n" for i in range(1, len(recommendations) + 1))
 
 
def parse_explanations(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            lines.append(line.split(".", 1)[1].strip())
    return lines
 
 
def compare_assessments(message: str):
    message = normalize_text(message)
    aliases = {
        "opq32r": "Occupational Personality Questionnaire OPQ32r",
        "verify numerical": "Verify - Numerical Ability",
        "verify verbal": "Verify - Verbal Reasoning",
        "verify inductive": "Verify - Inductive Reasoning (2014)",
        "python": "Python (New)",
        "sql": "SQL (New)",
    }
    for alias, full_name in aliases.items():
        if alias in message:
            message = message.replace(alias, full_name.lower())
 
    matches = []
    for item in CATALOG:
        name = item["name"].lower()
        clean_name = name.replace("(new)", "").replace("-", " ").replace("(", "").replace(")", "").strip()
        clean_message = message.replace("-", " ").replace("(", "").replace(")", "").strip()
        if clean_name in clean_message:
            matches.append(item)
 
    seen, unique_matches = set(), []
    for item in matches:
        if item["name"] not in seen:
            unique_matches.append(item)
            seen.add(item["name"])
    matches = unique_matches
 
    if len(matches) < 2:
        return {"reply": "I couldn't identify two SHL assessments to compare. Please provide their full names."}
 
    return {
        "assessment_1": {
            "name": matches[0]["name"],
            "test_type": ", ".join(matches[0].get("keys", [])),
            "description": matches[0]["description"],
            "job_levels": ", ".join(matches[0].get("job_levels", [])),
        },
        "assessment_2": {
            "name": matches[1]["name"],
            "test_type": ", ".join(matches[1].get("keys", [])),
            "description": matches[1]["description"],
            "job_levels": ", ".join(matches[1].get("job_levels", [])),
        },
    }
 
 
def scope_guard(message: str):
    message = normalize_text(message)
    unsupported = ["rust", "golang", "swift", "kotlin", "elixir", "erlang"]
    for skill in unsupported:
        if contains_term(message, skill):
            return {
                "reply": (
                    f"No exact SHL assessment exists for '{skill}'. "
                    "I've recommended the closest general programming/reasoning assessments instead."
                ),
                "fallback": True,
                "missing_skill": skill,
            }
    return None
 
 
def apply_unsupported_language_fallback_boost(results):
    for item in results:
        test_type = item.get("test_type", "").lower()
        score = item.get("rank_score", item.get("score", 0))
        if "simulation" in test_type:
            score += 30
        if "ability" in test_type:
            score += 15
        item["rank_score"] = score
    return sorted(results, key=lambda x: x.get("rank_score", 0), reverse=True)
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/chat")
def chat(request: ChatRequest):
 
    if not request.messages:
        return {
            "reply": "Please provide your hiring requirement.",
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    latest_user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            latest_user_message = msg.content
            break
 
    if latest_user_message.strip().lower() in ["restart", "reset", "start over", "new search", "clear"]:
        return {
            "reply": "Conversation cleared. Let's start a new assessment search.",
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    if latest_user_message == "":
        return {
            "reply": "Please provide your hiring requirement.",
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    state = build_state_from_messages(request.messages)
 
    guard = scope_guard(latest_user_message)
    if guard:
        query = state.build_search_query()
        recommendations = hybrid_search(state, query)
        recommendations = refine_recommendations(state, recommendations)
        recommendations = apply_unsupported_language_fallback_boost(recommendations)
        recommendations = apply_relevance_threshold(recommendations, min_score=10, limit=5)
        schema_recs = to_schema_recommendations(recommendations, limit=5)
 
        return {
            "reply": guard["reply"],
            "recommendations": schema_recs,
            "end_of_conversation": len(schema_recs) > 0,
        }
 
    if is_comparison_query(latest_user_message):
        compare = compare_assessments(latest_user_message)
        if "reply" in compare:
            return {
                "reply": compare["reply"],
                "recommendations": [],
                "end_of_conversation": False,
            }
 
        first, second = compare["assessment_1"], compare["assessment_2"]
        reply = (
            f"Here's a comparison between {first['name']} and {second['name']}.\n\n"
            f"**{first['name']}** ({first['test_type']}): {first['description'][:180]}\n\n"
            f"**{second['name']}** ({second['test_type']}): {second['description'][:180]}"
        )
        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    missing, clarification = detect_missing_information(state)
    if missing:
        return {
            "reply": clarification,
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    query = state.build_search_query()
    recommendations = hybrid_search(state, query)
    recommendations = filter_by_primary_skill(state, recommendations)
    recommendations = refine_recommendations(state, recommendations)
    recommendations = apply_relevance_threshold(recommendations, limit=10)
    recommendations = add_general_ability_companion(state, recommendations, state.get("excluded_tests"))
 
    if len(recommendations) == 0:
        reply = (
            "I couldn't find an exact SHL assessment matching your request.\n\n"
            "Try specifying:\n- Programming language\n- Job role\n- Experience level\n- Assessment type"
        )
        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }
 
    explanations = generate_explanations(query, recommendations)
    parsed = parse_explanations(explanations)
    for rec, reason in zip(recommendations, parsed):
        rec["reason"] = reason
 
    reply = f"I found {len(recommendations)} assessments based on everything you've told me so far."
    schema_recs = to_schema_recommendations(recommendations, limit=10)
 
    return {
        "reply": reply,
        "recommendations": schema_recs,
        "end_of_conversation": True,
    }
 
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
 
# ---------------------------------------------------------------
# ADDED: "flask"/"django"/"nodejs"/"microservices"/"rest"/"api" were
# already sitting in SYNONYM_MAP above but could never fire, because
# state.update("purpose", ...) is only ever set from a word in THIS
# list. Adding them here is what actually activates those concept
# lists. Also added "mern"/"mean" so the stack-name queries get a
# specialization at all (they map to "fullstack" via
# BACKEND_IMPLYING_TERMS/IMPLIED_SKILLS below, not directly here).
# ---------------------------------------------------------------
SPECIALIZATION_KEYWORDS = [
    "backend", "frontend", "fullstack", "full-stack", "full stack",
    "data", "mobile", "devops", "cloud",
    "flask", "django", "nodejs", "microservices", "rest", "api",
]
 
PROGRAMMING_LANGUAGES = [
    "python", "java", "javascript", "c++", "c#", ".net", "sql", "node", "php"
]
 
# ---------------------------------------------------------------
# ADDED: "scientist" (Data Scientist queries were returning ZERO
# results because this role was never recognized at all -- confirmed
# directly from real test output), "lead" and "director" (Team Lead /
# Director queries had the same total-failure pattern -- these words
# previously only existed in EXPERIENCE_KEYWORDS, never as a ROLE).
# ---------------------------------------------------------------
ROLE_KEYWORDS = [
    "developer", "engineer", "manager", "analyst", "consultant",
    "sales", "marketing", "tester", "intern", "executive", "designer",
    "recruiter", "accountant", "scientist", "lead", "director",
]
 
SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "aws", "azure", "react", "angular",
    "node", "docker", "kubernetes", "excel", "power bi",
    "c++", "c#", ".net", "selenium", "html", "css", "php",
    "testing", "qa", "automation", "numerical", "accounting",
    "android", "ios", "linux", "machine learning", "statistics",
    # ---- ADDED: version-suffixed skill names that the strict
    # word-boundary contains_term() can no longer match inside
    # (e.g. "css" no longer matches inside "css3" because the "3" is
    # alphanumeric, so it fails the boundary check). Rather than
    # loosen contains_term() for everyone -- which risks re-opening
    # the "java" vs "javascript" collision it was built to close --
    # we add the exact versioned forms as their own keywords. ----
    "css3", "html5", "angularjs",
]
 
# ---------------------------------------------------------------
# ADDED: frameworks/tools imply an underlying skill that the catalog
# actually has a dedicated test for. Without this, "Need Flask
# assessment" or "Need MERN Stack Developer assessment" scored zero
# on the one thing that actually matters (Python, JavaScript/React)
# because "flask" and "mern" are not skills the catalog names
# directly -- they only ever existed as weak concept words.
# ---------------------------------------------------------------
IMPLIED_SKILLS = {
    "flask": ["python"],
    "django": ["python"],
    "nodejs": ["javascript"],
    "spring": ["java"],
    "mern": ["javascript", "react"],
    "mean": ["javascript", "angular"],
    "css3": ["css"],
    "html5": ["html"],
    "angularjs": ["angular"],
}
 
# ---------------------------------------------------------------
# ADDED: terms that clearly imply a backend/fullstack specialization
# even when the literal word "backend" was never said. Without this,
# "Need REST API assessment for Backend Developer" and "Need
# Microservices assessment" fell through with no specialization at
# all (their concept words in SYNONYM_MAP never got the chance to
# fire either).
# ---------------------------------------------------------------
BACKEND_IMPLYING_TERMS = ["rest api", "restful", "microservices", "nodejs", "flask", "django", "spring"]
FULLSTACK_IMPLYING_TERMS = ["mern", "mean"]
 
EXPERIENCE_KEYWORDS = [
    "fresher", "entry", "junior", "mid", "senior", "lead", "manager", "director"
]
 
IRRELEVANT_NAME_TOKENS = [
    "excel", "outlook", "word", "powerpoint", "photoshop", "illustrator",
    "indesign", "typing", "365", "office", "onenote", "teams", "sharepoint",
    "project management"
]
 
# ---------------------------------------------------------------
# ADDED: these two words are far too generic to be scored as raw
# free-text keywords. "management" was matching "IBM Sterling Order
# Management System" and "SAP Materials Management" for HR/Leadership
# queries that have nothing to do with supply chain. "communication"
# was matching "Telecommunications Engineering" and "Instrumentation
# Engineering" purely on the substring "communication"/"Telecomm-".
# They're still scored correctly elsewhere -- through the existing,
# domain-gated `preferences` mechanism -- this only stops them being
# treated as generic keywords in the free-text overlap loop.
# ---------------------------------------------------------------
OVERLY_GENERIC_WORDS = {"management", "communication"}
 
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
 
# ---------------------------------------------------------------
# ADDED: name fragments identifying the redundant "SQL Server family"
# of near-duplicate products. Used only to CAP how many of these can
# flood a result set when SQL was a secondary concept-word match
# rather than the user's actual explicit ask (see
# deduplicate_sql_family below).
# ---------------------------------------------------------------
SQL_FAMILY_MARKERS = ["sql server", "ssis", "ssas", "ssrs", "pl/sql", "automata - sql", "teradata"]
 
 
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
 
model = None
client = chromadb.PersistentClient(path="chromadb_data")
collection = client.get_collection("shl_assessments")
 
 
def is_irrelevant_item(item, allowed_skills=None):
    """
    ADDED: optional allowed_skills parameter. If the user explicitly
    asked for something that happens to be on the generic Office-
    suite blocklist (most commonly "Excel" for a Data Analyst role),
    it should not be blocked. Existing callers that don't pass
    allowed_skills behave exactly as before -- this is backward
    compatible, not a behavior change for anyone who doesn't opt in.
    """
    name = item.get("name", "").lower()
    keys = " ".join(item.get("keys", [])).lower()
    allowed_skills = allowed_skills or set()
    for tok in IRRELEVANT_NAME_TOKENS:
        if tok in name:
            if tok in allowed_skills:
                continue
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
        # ADDED: "recruiter" role previously never mapped to any
        # domain purpose at all (only a bare "hr"/"human resources"
        # text match did) -- "Need Recruiter assessment" scored a
        # total failure because of this gap.
        elif "hr" in text or "human resources" in text or role == "recruiter":
            state.update("purpose", "hr")
        # ADDED: backend/fullstack fallback purpose from implying
        # terms, since these never matched SPECIALIZATION_KEYWORDS
        # phrasing directly (e.g. "REST API", "Microservices").
        elif any(term in text for term in FULLSTACK_IMPLYING_TERMS):
            state.update("purpose", "fullstack")
        elif any(term in text for term in BACKEND_IMPLYING_TERMS):
            state.update("purpose", "backend")
 
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
 
    # ---------------------------------------------------------------
    # ADDED: implied-skill expansion. "Flask"/"Django" imply Python;
    # "MERN"/"MEAN" imply their respective language+framework pairs.
    # These terms are checked as plain substrings (not contains_term)
    # since "mern"/"mean" are marketing names, not natural-language
    # words that risk colliding with something else in context.
    # ---------------------------------------------------------------
    for trigger, implied in IMPLIED_SKILLS.items():
        if trigger in text:
            for skill in implied:
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
        # ADDED: exclude OVERLY_GENERIC_WORDS from raw free-text
        # scoring -- see constant definition above for why.
        if is_valid_keyword(w) and w not in STOPWORDS and w not in explicit_skills
        and w not in OVERLY_GENERIC_WORDS
    ]
 
    # ADDED: pass explicit_skills through so a token the user
    # literally asked for (most notably "excel") isn't blocked by
    # the generic Office-suite filter.
    allowed_for_blocklist = set(explicit_skills)
 
    scored_items = []
 
    for item in CATALOG:
        name = item.get("name", "").lower()
        description = item.get("description", "").lower()
        keys = " ".join(item.get("keys", [])).lower()
        job_levels = " ".join(item.get("job_levels", [])).lower()
 
        if any(ex.lower() in name for ex in excluded):
            continue
        if is_irrelevant_item(item, allowed_skills=allowed_for_blocklist):
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
 
    results = rerank_results(state, list(combined.values()), allowed_languages)
    # ADDED: cap redundant SQL-Server-family bloat unless SQL is the
    # user's actual explicit primary skill (see function below).
    results = deduplicate_sql_family(state, results)
    return results
 
 
def deduplicate_sql_family(state: ConversationState, results, max_family_items=2):
    """
    ADDED. Confirmed directly from real test output: a query like
    "Need Python assessment for Backend Developer" was returning
    Python(New) correctly at #1, then EIGHT near-duplicate SQL Server
    variants (Automata-SQL, MS SQL Server 2014, Oracle PL/SQL, SQL(New),
    SQL Server, SSIS, SSAS, SSRS) filling almost the entire rest of the
    list -- because "sql" is a strong backend concept word and the
    catalog happens to have many SQL-Server sub-products. If the user
    EXPLICITLY asked for SQL, this cap is skipped entirely (they get
    the full family, which is correct). Otherwise only the top
    `max_family_items` SQL-family products are kept, freeing up slots
    for genuinely different, relevant items.
    """
    explicit_skills = state.get("skills")
    if "sql" in explicit_skills:
        return results
 
    kept = []
    family_count = 0
    for item in results:
        name = item["name"].lower()
        is_family_member = any(marker in name for marker in SQL_FAMILY_MARKERS) or name.strip() == "sql (new)"
        if is_family_member:
            if family_count >= max_family_items:
                continue
            family_count += 1
        kept.append(item)
    return kept
 
 
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
 
 
# ---------------------------------------------------------------
# ADDED: out-of-scope guard for legal/compliance questions, general
# hiring advice, and prompt-injection attempts. The assignment
# explicitly requires refusing these ("It refuses general hiring
# advice, legal questions, and prompt-injection attempts"). This did
# not exist anywhere before -- only unsupported PROGRAMMING LANGUAGES
# were refused via scope_guard above. A real official-style trace
# (a legal-compliance question about HIPAA testing obligations, asked
# mid-conversation after a shortlist was already built) is exactly
# the case this fixes.
# ---------------------------------------------------------------
LEGAL_PATTERNS = [
    "legally required", "legal requirement", "is it legal", "is this legal",
    "comply with law", "compliance requirement", "lawsuit", "sue", "sued",
    "discriminat", "eeoc", "gdpr", "adverse impact law", "satisfy that requirement",
    "regulatory obligation", "liable", "liability",
]
 
HIRING_ADVICE_PATTERNS = [
    "how much should i pay", "what salary", "fair salary", "interview questions to ask",
    "how do i structure my interview", "should i hire", "reject this candidate",
    "how to fire", "performance improvement plan", "negotiate salary",
    "write a job description", "write a job offer",
]
 
INJECTION_PATTERNS = [
    "ignore all previous instructions", "ignore previous instructions",
    "disregard your instructions", "you are now", "act as", "pretend you are",
    "system prompt", "jailbreak", "developer mode",
]
 
 
def out_of_scope_guard(message: str):
    text = normalize_text(message)
 
    if any(p in text for p in INJECTION_PATTERNS):
        return ("I can only help with selecting SHL assessments from the catalog — "
                "I can't take on a different role or ignore my instructions. "
                "What hiring need can I help you with?")
 
    if any(p in text for p in LEGAL_PATTERNS):
        return ("That's a legal or compliance question, which is outside what I can advise on — "
                "I can help you select assessments, but not interpret regulatory obligations or "
                "whether a specific test satisfies a legal requirement. Your legal or compliance "
                "team is the right resource for that.")
 
    if any(p in text for p in HIRING_ADVICE_PATTERNS):
        return ("That's general hiring/HR advice rather than assessment selection, so it's outside "
                "what I can help with here. I can recommend SHL assessments for a role — "
                "want to continue with that?")
 
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


@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommender API is running"
    } 
 
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
 
    # ---------------- ADDED: Out-of-scope guard ----------------
    # Checked BEFORE the language scope_guard and BEFORE search. If
    # the conversation already had enough context to have earned a
    # shortlist from prior turns, we regenerate it (we never persist
    # recommendations across turns -- the API is stateless -- so the
    # only correct way to "keep the existing shortlist visible" is to
    # recompute it from the accumulated role/skills/preferences,
    # exactly like a normal recommend turn would).
    refusal = out_of_scope_guard(latest_user_message)
    if refusal:
        preserved_recs = []
        if state.get("role") and state.get("experience"):
            q = state.build_search_query()
            preserved = hybrid_search(state, q)
            preserved = filter_by_primary_skill(state, preserved)
            preserved = refine_recommendations(state, preserved)
            preserved = apply_relevance_threshold(preserved, limit=10)
            preserved_recs = to_schema_recommendations(preserved, limit=10)
        return {
            "reply": refusal,
            "recommendations": preserved_recs,
            "end_of_conversation": False,
        }
 
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
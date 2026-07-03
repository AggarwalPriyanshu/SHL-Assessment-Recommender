from typing import Dict, Any
 
 
class ConversationState:
    """
    IMPORTANT: This class must be instantiated FRESH inside every
    /chat request handler -- never as a module-level singleton.
 
    The API contract is stateless: the full conversation history
    arrives in every request. A shared/global instance would leak
    state between unrelated conversations (and between concurrent
    requests), which breaks both the statelessness requirement and
    correctness of recommendations. See app.py: a new ConversationState()
    is created per request and rebuilt by replaying request.messages.
    """
 
    def __init__(self):
        self.reset()
 
    def reset(self):
 
        self.state = {
 
            # Main hiring information
            "role": None,
            "experience": None,
            "industry": None,
            "purpose": None,
            "language": None,
 
            # Dynamic information
            "skills": [],
            "preferences": [],
            "constraints": [],
 
            # Recommendation refinement
            "excluded_tests": [],
            "current_recommendations": [],
 
            # Conversation tracking
            "clarification_stage": None,
            "history": []
        }
 
    ######################################################
    # Conversation History
    ######################################################
 
    def add_user_message(self, message: str):
        self.state["history"].append({"speaker": "user", "message": message})
 
    def add_bot_message(self, message: str):
        self.state["history"].append({"speaker": "assistant", "message": message})
 
    ######################################################
    # Generic Update
    ######################################################
 
    def update(self, key, value):
 
        if key not in self.state:
            return
 
        # Recommendations should always REPLACE
        if key == "current_recommendations":
            self.state[key] = value
            return
 
        # Scalars (role / experience / etc.)
        if not isinstance(self.state[key], list):
            self.state[key] = value
            return
 
        # List values
        if isinstance(value, list):
            for item in value:
                if item not in self.state[key]:
                    self.state[key].append(item)
        else:
            if value not in self.state[key]:
                self.state[key].append(value)
 
    ######################################################
    # Skill Management
    ######################################################
 
    def add_skill(self, skill):
        skill = skill.lower()
        if skill not in self.state["skills"]:
            self.state["skills"].append(skill)
 
    def remove_skill(self, skill):
        skill = skill.lower()
        if skill in self.state["skills"]:
            self.state["skills"].remove(skill)
 
    ######################################################
    # Preferences
    ######################################################
 
    def add_preference(self, pref):
        pref = pref.lower()
        if pref not in self.state["preferences"]:
            self.state["preferences"].append(pref)
 
    ######################################################
    # Constraints
    ######################################################
 
    def add_constraint(self, constraint):
        constraint = constraint.lower()
        if constraint not in self.state["constraints"]:
            self.state["constraints"].append(constraint)
 
    ######################################################
    # Excluded Tests
    ######################################################
 
    def exclude_test(self, test_name):
        if test_name not in self.state["excluded_tests"]:
            self.state["excluded_tests"].append(test_name)
 
    ######################################################
    # Recommendation Memory
    ######################################################
 
    def save_recommendations(self, recommendations):
        self.update("current_recommendations", recommendations)
 
    def clear_recommendations(self):
        self.state["current_recommendations"] = []
 
    def clear_history(self):
        self.state["history"] = []
 
    ######################################################
    # Clarification
    ######################################################
 
    def set_clarification(self, stage):
        self.state["clarification_stage"] = stage
 
    def clear_clarification(self):
        self.state["clarification_stage"] = None
 
    ######################################################
    # Build Final Search Query
    ######################################################
 
    def build_search_query(self):
        parts = []
 
        if self.state["role"]:
            parts.append(self.state["role"])
        if self.state["experience"]:
            parts.append(self.state["experience"])
        if self.state["industry"]:
            parts.append(self.state["industry"])
        if self.state["purpose"]:
            parts.append(self.state["purpose"])
        if self.state["language"]:
            parts.append(self.state["language"])
 
        parts.extend(self.state["skills"])
        parts.extend(self.state["preferences"])
        parts.extend(self.state["constraints"])
 
        return " ".join(parts).strip()
 
    def has_context(self):
        return any([
            self.state["role"],
            self.state["experience"],
            self.state["industry"],
            self.state["purpose"],
            self.state["skills"],
            self.state["preferences"],
        ])
 
    def last_user_message(self):
        for msg in reversed(self.state["history"]):
            if msg["speaker"] == "user":
                return msg["message"]
        return ""
 
    def last_bot_message(self):
        for msg in reversed(self.state["history"]):
            if msg["speaker"] == "assistant":
                return msg["message"]
        return ""
 
    ######################################################
    # Access
    ######################################################
 
    def get(self, key):
        return self.state.get(key)
 
    def get_state(self):
        return dict(self.state)
 
    def __str__(self):
        return str(self.state)
 
 
# NOTE: deliberately NOT instantiating a module-level singleton here.
# `conversation_state = ConversationState()` was removed on purpose --
# see the class docstring. Each request in app.py must create its own
# `ConversationState()` instance and rebuild it from the incoming
# `messages` array.
 
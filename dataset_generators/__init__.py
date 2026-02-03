import random


def parse_object_names(filename):
    """
    Reads a file of object names and returns a dict mapping canonical names to lists of synonyms.
    Expected file format: 'canonical: alt1, alt2, ...'
    """
    object_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            canonical, alts_str = line.split(":", 1)
            canonical = canonical.strip()
            alt_list = [alt.strip() for alt in alts_str.split(",") if alt.strip()]
            object_dict[canonical] = alt_list
    return object_dict


def build_actions_dict(is_stl=False):
    """
    Returns a dictionary mapping canonical actions to lists of action synonyms.
    """
    actions = {
        # ── warehouse verbs ───────────────────────────────────────────────
        "search": ["locate", "find", "look for", "search for", "spot", "detect"],
        "pickup": ["pick up", "grab", "retrieve", "collect", "take", "fetch", "lift"],
        "deliver": ["deliver", "drop off", "hand over", "transport", "bring", "move"],
        "idle": ["idle", "wait", "remain still", "stand by", "pause", "rest"],
        "inspect": ["inspect", "check", "examine", "verify", "inspecting", "assessing"],
        "carry": ["carry", "hold", "transport", "move", "convey", "take along"],
        "organize": ["organize", "sort", "arrange", " tidy", "整理"],
        "stack": ["stack", "pile", "arrange", "stack up", "heap"],
        "place": ["place", "put", "set down", "deposit", "position", "leave"],
        "collect": ["collect", "gather", "amass", "compile", "accumulate"],
        # ── search_and_rescue verbs ───────────────────────────────────────
        "go_home": ["return to base", "go home", "return home", "go back to base"],
        "communicate": [
            "talk to",
            "communicate with",
            "establish communication with",
            "speak to",
        ],
        "deliver_aid": [
            "give aid to",
            "deliver aid to",
            "provide assistance to",
            "help",
        ],
        "record": [
            "record",
            "begin recording",
            "take a video of",
            "document",
            "capture",
        ],
        "photo": [
            "photograph",
            "take a picture of",
            "take a photo of",
            "snap",
            "image",
        ],
        "avoid": ["avoid", "stay away from", "do not go near", "evade", "dodge"],
        "navigate": ["navigate", "travel to", "head to", "go to", "move towards"],
        "monitor": ["monitor", "watch", "observe", "track", "supervise"],
        "assist": ["assist", "aid", "support", "help", "serve"],
        "secure": ["secure", "lock", "fasten", "stabilize", "make safe"],
        # ── traffic_light verbs ───────────────────────────────────────────
        "get_help": ["call for help", "request assistance", "get help", "summon aid"],
        "change": ["change", "switch", "set", "update", "modify", "adjust"],
        "record": [
            "record",
            "begin recording",
            "take a video of",
            "document",
            "capture",
        ],
        "photo": [
            "photograph",
            "take a picture of",
            "take a photo of",
            "snap",
            "image",
        ],
        "signal": ["signal", "indicate", "warn", "alert", "notify"],
        "control": ["control", "manage", "operate", "direct", "govern"],
        "activate": ["activate", "turn on", "start", "enable", "switch on"],
        "deactivate": ["deactivate", "turn off", "stop", "disable", "switch off"],
    }

    return actions


__all__ = ["parse_object_names", "build_actions_dict"]

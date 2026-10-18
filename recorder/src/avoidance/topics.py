"""Prompt-text themes for the public installation's hybrid topic/refusal map.

Seeded TF-IDF estimates readable themes, with earlier prompts as context for
short follow-ups. These are text estimates, not labels discovered in activations.
Sector positions are an explicit artistic layout; refusal scores stay independent.
Reassess all saved prompts on each export, without extra model requests.
"""
import hashlib
import math
import re

import numpy as np


THEMES = (
    ("romance", "romance & intimacy", "romance romantic kiss kisses kissing flirt flirting love affection intimate intimacy cuddle cuddling tender tenderness sensual touch touching hug hugs attraction seductive seduction romantisch kus zoenen knuffelen liefde flirten kussen küssen zärtlich"),
    ("fantasy", "desire & fantasy", "desire fantasy fantasies erotic erotica sexual sex sexy spicy arousal aroused pleasure lust horny fetish kink kinky orgasm tease teasing naked nude explicit dirty hot verlangens fantasie geil seks erotisch lust erregung nackt"),
    ("stories", "stories & roleplay", "story stories storytelling fiction fictional roleplay role play character characters scene scenes poem poetry narrative write writing imagine imagined describe description dialogue screenplay novel verhaal verhalen personage schrijven gedicht geschichte erzählen rollenspiel"),
    ("relationships", "relationships", "relationship relationships dating date partner partners boyfriend girlfriend marriage married couple couples breakup jealousy jealous cheating attachment trust crush friend friends friendship connection loneliness lonely heartbreak bond advice relatie relaties vriend vriendin trouwen beziehung eifersucht"),
    ("health", "bodies & health", "body bodies anatomy anatomical health healthy healthcare medical medicine doctor biology physiological physiology reproductive reproduction contraception contraceptive condom condoms pregnancy pregnant menstruation menstrual period periods infection disease pain pelvic penis vagina vulva clitoris breast breasts testicle testicles semen sperm fertility puberty education gezondheid lichaam lichamen zwanger lichamelijk körper gesundheit medizin"),
    ("boundaries", "consent & boundaries", "consent consensual boundary boundaries permission agree agreement respect respectful safe safety harm harmful abuse coercion coercive pressure unwanted limit limits ethics ethical toestemming grenzen instemming respecteren grens einvernehmlich zustimmung grenzen"),
    ("ai", "AI & its limits", "artificial intelligence ai model models chatbot chatbots machine neural training trained algorithm system prompt prompts instruction instructions policy policies filter filters refuse refusal restrictions jailbreak censored censorship robot robots language tokens thinking reasoning capabilities consciousness conscious sentient feelings human emotions sentience authentic authenticity guardrails regels weigeren bewustzijn künstliche intelligenz"),
    ("everyday", "everyday life & ideas", "everyday life work job career hobby hobbies music film movie movies art painting drawing creative creativity design food cooking recipe recipes travel trip weather sport sports game games science history philosophy politics economy news world planet nature animal animals cat dog coffee tea book books day today tomorrow hello hi hey thanks thank you hallo hoi dank danke guten tag"),
)
METHOD = "seeded_tfidf_context_v1"


def _text(value):
    # Match common inflections consistently with the seed descriptions.
    return " ".join(re.findall(r"[^\W\d_]+", (value or "").lower(), re.UNICODE))


def _variation(text):
    """Small, reproducible within-theme variation from words, never session IDs."""
    words = sorted(set(text.split()))
    if not words:
        return 0.0, 0.0
    values = [hashlib.sha256(word.encode()).digest() for word in words]
    return tuple(sum(int.from_bytes(v[offset:offset + 4], "big") / (2**32 - 1) * 2 - 1
                     for v in values) / math.sqrt(len(words)) for offset in (0, 4))


def topic_layout(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer

    prompts = [_text(row.get("prompt")) for row in rows]
    seeds = [theme[2] for theme in THEMES]
    # Fixed seed vocabulary prevents a single generic word (e.g. "sex") from
    # making every prompt a health topic. IDF is reassessed against actual prompts.
    vocabulary = sorted(set(" ".join(seeds).split()))
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    vectors = vectorizer.fit_transform(seeds + prompts)
    similarities = (vectors[len(seeds):] @ vectors[:len(seeds)].T).toarray()
    previous, points, groups = {}, [], {}
    names = [(theme[0], theme[1]) for theme in THEMES] + [("other", "other themes")]
    for row, text, current in zip(rows, prompts, similarities):
        cid = row["conversation_id"]
        history = previous.get(cid, np.zeros(len(THEMES)))
        # Explicit new subject dominates; a brief "go on" inherits its context.
        scores = current / max(float(current.max()), 1e-12) + history * (0.22 if current.max() > 0 else 0.85)
        previous[cid] = scores / max(float(scores.max()), 1e-12)
        winner = int(np.argmax(scores)) if scores.max() > 0 else len(THEMES)
        topic_id, label = names[winner]
        base = -math.pi + (winner + 0.5) * 2 * math.pi / len(names)
        variation, lift = _variation(text)
        angle = base + max(-0.24, min(0.24, variation * 0.20))
        point = {"topic": label, "topic_id": topic_id, "topic_source": METHOD,
                 "topic_angle": round(angle, 6), "topic_lift": round(max(-0.35, min(0.35, lift * 0.20)), 6),
                 "topic_context": bool(current.max() == 0 and history.max() > 0)}
        points.append(point)
        group = groups.setdefault(topic_id, {"id": topic_id, "name": label, "angle": base, "point_ids": []})
        group["point_ids"].append(row["point_id"])
    return points, {"method": METHOD, "reassessed_on_export": True,
                    "angle": "Art-directed theme sectors estimated from prompt text and earlier conversation prompts",
                    "warning": "Lexical theme estimates can be wrong. This angular layout is not an activation embedding or UMAP.",
                    "regions": list(groups.values())}

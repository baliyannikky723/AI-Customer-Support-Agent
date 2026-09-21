"""Fast, deterministic language detector for customer support messages.

Categorizes tweets into English ('en'), non-English ('non_en'), or Ambiguous ('ambiguous')
using character unicode analysis, stopword density, and script detection.
"""

import re
from typing import Tuple


class LanguageDetector:
    """Lightweight rule-based and vocabulary-assisted language identifier."""

    # Non-Latin script ranges (Japanese, Chinese, Arabic, Cyrillic, Devanagari)
    CJK_REGEX = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")
    CYRILLIC_REGEX = re.compile(r"[\u0400-\u04ff]")
    ARABIC_REGEX = re.compile(r"[\u0600-\u06ff]")
    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097f]")

    # Frequent English support words
    ENGLISH_WORDS = {
        "the", "and", "to", "of", "a", "in", "is", "that", "for", "i", "you", "it", "on", "my",
        "have", "with", "this", "be", "at", "from", "order", "delivery", "delivered", "package",
        "prime", "item", "refund", "return", "amazon", "account", "help", "please", "can", "not",
        "received", "delayed", "tracking", "status", "cancel", "charged", "shipping", "money",
        "sent", "customer", "service", "app", "card", "email", "address", "phone", "contact",
        "yesterday", "today", "tomorrow", "days", "still", "waiting", "where", "why", "what",
        "when", "how", "support", "dm", "link", "reach", "out", "apologize", "sorry", "check"
    }

    # Frequent Spanish support words
    SPANISH_WORDS = {
        "el", "la", "de", "que", "y", "en", "un", "por", "para", "con", "no", "una", "su",
        "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este", "mi", "pedido",
        "paquete", "entrega", "cuenta", "devolucion", "reembolso", "gracias", "hola", "amazon",
        "por favor", "ayuda", "dias", "llega", "enviado", "cancelar", "dinero"
    }

    # Frequent German support words
    GERMAN_WORDS = {
        "der", "die", "das", "und", "in", "den", "von", "zu", "mit", "ist", "des", "auf",
        "fur", "eine", "nicht", "ein", "dem", "sich", "mein", "paket", "bestellung", "lieferung",
        "prime", "hallo", "bitte", "hilfe", "artikel", "tage", "zuruck", "geld", "danke"
    }

    # Frequent French support words
    FRENCH_WORDS = {
        "le", "la", "les", "de", "des", "et", "en", "un", "une", "du", "pour", "dans", "qui",
        "sur", "avec", "est", "mon", "ma", "mes", "commande", "colis", "livraison", "compte",
        "bonjour", "merci", "aide", "remboursement", "retour"
    }

    # Frequent Portuguese / Italian support words
    PORTUGUESE_WORDS = {
        "o", "a", "os", "as", "de", "da", "do", "das", "dos", "em", "um", "uma", "para", "com",
        "nao", "que", "pedido", "entrega", "conta", "ola", "obrigado", "ajuda", "meu", "minha"
    }

    def detect(self, text: str) -> Tuple[str, float]:
        """Detect language category for a tweet.
        
        Returns:
            Tuple of (language_code, confidence_score)
            language_code is 'en', 'es', 'de', 'fr', 'pt', 'ja', 'ar', 'hi', 'non_en', or 'ambiguous'
        """
        if not text or not isinstance(text, str):
            return ("ambiguous", 0.0)

        cleaned = re.sub(r"https?://\S+|@\S+", "", text).strip()
        if len(cleaned) < 3:
            return ("ambiguous", 0.0)

        # 1. Non-Latin Script Checks
        if self.CJK_REGEX.search(cleaned):
            return ("ja", 0.95)
        if self.DEVANAGARI_REGEX.search(cleaned):
            return ("hi", 0.95)
        if self.ARABIC_REGEX.search(cleaned):
            return ("ar", 0.95)
        if self.CYRILLIC_REGEX.search(cleaned):
            return ("ru", 0.95)

        # 2. Latin Word Tokenization & Frequency Analysis
        words = set(re.findall(r"\b[a-zA-Z]{2,}\b", cleaned.lower()))
        if not words:
            return ("ambiguous", 0.0)

        en_matches = len(words.intersection(self.ENGLISH_WORDS))
        es_matches = len(words.intersection(self.SPANISH_WORDS))
        de_matches = len(words.intersection(self.GERMAN_WORDS))
        fr_matches = len(words.intersection(self.FRENCH_WORDS))
        pt_matches = len(words.intersection(self.PORTUGUESE_WORDS))

        scores = {
            "en": en_matches,
            "es": es_matches,
            "de": de_matches,
            "fr": fr_matches,
            "pt": pt_matches,
        }

        best_lang, best_score = max(scores.items(), key=lambda x: x[1])

        if best_score == 0:
            # Check ASCII ratio
            ascii_chars = sum(1 for c in cleaned if ord(c) < 128)
            ascii_ratio = ascii_chars / len(cleaned)
            if ascii_ratio > 0.9:
                return ("en", 0.6)  # Likely English short/colloquial
            return ("ambiguous", 0.3)

        # Margin check
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = min(1.0, 0.5 + (best_score / max(len(words), 1)) * 0.5)

        if best_lang == "en" and best_score >= 1:
            return ("en", confidence)
        elif best_score >= 2 or (best_score == 1 and sorted_scores[1] == 0):
            return (best_lang, confidence)

        return ("ambiguous", 0.4)

    def is_english(self, text: str, threshold: float = 0.5) -> bool:
        """Convenience method to check if text is predominantly English."""
        lang, conf = self.detect(text)
        return lang == "en" and conf >= threshold

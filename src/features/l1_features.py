BADWORDS = [
    "select","union","insert","update","delete","drop","sleep","benchmark",
    "or","and","information_schema",
    "<script","javascript:","onerror","onload",
    "../","..\\","/etc/passwd","cmd.exe","powershell",
    "wget","curl","base64",";","|","&&","--","/*","*/","@@"
]

# Adjust to your API/business context
ALLOWED_SPECIAL = set("-_.~:/?&=%@,+#[]{}()\"'")

def _count_alnum(s: str) -> int:
    return sum(ch.isalnum() for ch in s)

def _count_special(s: str) -> int:
    return sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)

def _count_illegal_special(s: str) -> int:
    illegal = 0
    for ch in s:
        if (not ch.isalnum()) and (not ch.isspace()):
            if ch not in ALLOWED_SPECIAL:
                illegal += 1
    return illegal

def _count_badwords(s: str) -> int:
    ss = s.lower()
    return sum(1 for w in BADWORDS if w in ss)

def extract_l1_ratios(payload: str) -> dict:
    payload = payload or ""
    length = max(len(payload), 1)

    alnum = _count_alnum(payload)
    special = _count_special(payload)
    illegal = _count_illegal_special(payload)
    denom = max(alnum, 1)
    bad = _count_badwords(payload)

    return {
        "alnum_ratio": alnum / length,
        "badwords_ratio": bad / denom,
        "special_ratio": special / length,
        "illegal_special_ratio": (illegal / max(special, 1)),
    }

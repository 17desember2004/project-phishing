import re
import tldextract
from urllib.parse import urlparse, unquote
from collections import Counter
import math
 
# ============================================================
# DAFTAR TLD LEGITIMATE YANG LEBIH LENGKAP
# Sumber: IANA + statistik domain global
# Kenapa penting: model dilatih dengan TLDLegitimateProb dari
# dataset — kalau TLD tidak dikenali → nilai 0 → model bias
# ============================================================
LEGIT_TLDS = {
    # Global umum
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
    # Indonesia
    'co.id', 'id', 'ac.id', 'go.id', 'sch.id', 'net.id',
    'or.id', 'web.id', 'biz.id', 'my.id',
    # Negara umum
    'co.uk', 'uk', 'org.uk', 'ac.uk', 'gov.uk',
    'com.au', 'net.au', 'org.au', 'edu.au', 'gov.au',
    'co.jp', 'jp', 'ac.jp', 'go.jp',
    'com.sg', 'sg', 'edu.sg', 'gov.sg',
    'com.my', 'my', 'edu.my', 'gov.my',
    'co.nz', 'nz',
    'co.za', 'za',
    'com.br', 'br',
    'com.mx', 'mx',
    'com.ar', 'ar',
    'com.ph', 'ph',
    'com.vn', 'vn',
    'com.tw', 'tw',
    'com.hk', 'hk',
    'com.cn', 'cn',
    'co.kr', 'kr',
    'co.in', 'in',
    # Eropa
    'de', 'fr', 'it', 'es', 'nl', 'be', 'ch', 'at',
    'pl', 'cz', 'hu', 'ro', 'bg', 'hr', 'sk', 'si',
    'pt', 'gr', 'se', 'no', 'dk', 'fi', 'ie',
    # Lainnya yang umum
    'io', 'ai', 'app', 'dev', 'web',
    'info', 'biz', 'mobi', 'name', 'pro', 'tel',
    'store', 'shop', 'online', 'site', 'website',
    'cloud', 'tech', 'media', 'news',
}
 
# TLD yang sering dipakai phishing (domain gratis/murah)
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq',   # domain gratis Freenom
    'xyz', 'top', 'click', 'link',
    'download', 'review', 'country',
    'stream', 'racing', 'party',
    'win', 'bid', 'loan', 'work',
    'date', 'faith', 'trade',
    'webcam', 'accountant', 'cricket',
    'science', 'pw',
}
 
# ============================================================
# HITUNG ENTROPY KARAKTER (Shannon Entropy)
# Kenapa: URL phishing cenderung punya karakter lebih acak
# → entropy lebih tinggi
# ============================================================
def shannon_entropy(text):
    if not text:
        return 0.0
    counter = Counter(text)
    length  = len(text)
    entropy = 0.0
    for count in counter.values():
        prob     = count / length
        entropy -= prob * math.log2(prob)
    return entropy
 
 
# ============================================================
# HITUNG TLD LEGIT PROB
# Mengembalikan nilai antara 0.0 - 1.0
# Bukan binary, tapi graded agar lebih mirip dataset asli
# ============================================================
def hitung_tld_legit_prob(suffix: str) -> float:
    if not suffix:
        return 0.0
    suffix_lower = suffix.lower()
    # TLD suspicious → prob rendah
    if suffix_lower in SUSPICIOUS_TLDS:
        return 0.05
    # TLD legitimate → prob tinggi
    if suffix_lower in LEGIT_TLDS:
        return 0.9
    # TLD tidak dikenal → prob sedang-rendah
    # (bisa jadi TLD baru yang legitimate, tapi lebih sering phishing)
    if len(suffix_lower) <= 3:
        return 0.5   # TLD pendek — agak umum
    return 0.2       # TLD panjang tidak dikenal — lebih mencurigakan
 
 
# ============================================================
# FEATURE EXTRACTION UTAMA
# Input  : URL string
# Output : dict berisi 20 fitur yang SAMA dengan training data
# ============================================================
def extract_features(url: str) -> dict:
 
    # Pastikan URL punya scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
 
    parsed = urlparse(url)
    ext    = tldextract.extract(url)
 
    domain = ext.domain or ''
    suffix = ext.suffix  or ''
    subdom = ext.subdomain or ''
 
    # ── URL LENGTH ──────────────────────────────────────────
    # Hitung panjang URL lengkap
    url_length    = len(url)
    domain_length = len(domain)
    tld_length    = len(suffix)
 
    # ── DOMAIN IP ───────────────────────────────────────────
    # Cek apakah netloc berupa IP address (misal 192.168.1.1)
    netloc_clean  = parsed.netloc.split(':')[0]  # buang port
    is_domain_ip  = int(
        bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', netloc_clean))
    )
 
    # ── SUBDOMAIN ───────────────────────────────────────────
    # Hitung jumlah subdomain (pisah berdasarkan titik)
    subdom_parts   = [s for s in subdom.split('.') if s and s != 'www']
    no_of_subdomain = len(subdom_parts)
 
    # ── KARAKTER URL ────────────────────────────────────────
    letters       = sum(c.isalpha()  for c in url)
    digits        = sum(c.isdigit()  for c in url)
    special_chars = sum(not c.isalnum() for c in url)
 
    letter_ratio  = letters       / url_length if url_length else 0.0
    digit_ratio   = digits        / url_length if url_length else 0.0
    special_ratio = special_chars / url_length if url_length else 0.0
 
    # ── SIMBOL KHUSUS ───────────────────────────────────────
    no_equals = url.count('=')
    no_qmark  = url.count('?')
    no_amp    = url.count('&')
 
    # ── OBFUSCATION (URL Encoding) ──────────────────────────
    # Bandingkan URL asli dengan versi yang sudah di-decode
    # Kalau berbeda → ada karakter yang di-encode (%20, %3A dll)
    decoded           = unquote(url)
    has_obfuscation   = int(decoded != url)
    len_diff          = abs(len(decoded) - len(url))
    no_obfuscated     = len_diff
    obfuscation_ratio = len_diff / url_length if url_length else 0.0
 
    # ── CHAR CONTINUATION RATE ──────────────────────────────
    # Hitung berapa kali karakter yang sama muncul berturutan
    # Phishing kadang pakai "aaaa" atau "----" untuk isi URL
    continuation = sum(
        1 for i in range(len(url) - 1)
        if url[i] == url[i + 1]
    )
    continuation_rate = continuation / url_length if url_length else 0.0
 
    # ── URL CHAR PROB (Shannon Entropy) ─────────────────────
    # Entropy tinggi = karakter lebih acak = lebih mencurigakan
    # Ini pendekatan yang konsisten dengan dataset PhiUSIIL
    url_char_prob = shannon_entropy(url)
 
    # ── TLD LEGITIMATE PROB ─────────────────────────────────
    # Probabilitas bahwa TLD ini dipakai oleh domain legitimate
    tld_legit_prob = hitung_tld_legit_prob(suffix)
 
    # ── SUSUN DICT FITUR (urutan harus sama dengan training) ─
    features = {
        'URLLength'                  : url_length,
        'DomainLength'               : domain_length,
        'TLDLength'                  : tld_length,
        'IsDomainIP'                 : is_domain_ip,
        'NoOfSubDomain'              : no_of_subdomain,
        'NoOfLettersInURL'           : letters,
        'LetterRatioInURL'           : letter_ratio,
        'NoOfDegitsInURL'            : digits,
        'DegitRatioInURL'            : digit_ratio,
        'NoOfEqualsInURL'            : no_equals,
        'NoOfQMarkInURL'             : no_qmark,
        'NoOfAmpersandInURL'         : no_amp,
        'NoOfOtherSpecialCharsInURL' : special_chars,
        'SpacialCharRatioInURL'      : special_ratio,
        'CharContinuationRate'       : continuation_rate,
        'URLCharProb'                : url_char_prob,
        'TLDLegitimateProb'          : tld_legit_prob,
        'HasObfuscation'             : has_obfuscation,
        'NoOfObfuscatedChar'         : no_obfuscated,
        'ObfuscationRatio'           : obfuscation_ratio,
    }
 
    return features
 
 
# ============================================================
# TEST MANUAL — jalankan: python src/feature_extraction.py
# ============================================================
if __name__ == "__main__":
 
    test_urls = [
        # Legitimate
        "https://www.google.com",
        "https://www.klikbca.com",
        "https://github.com/user/repo",
        "https://tokopedia.com/produk/abc",
        "https://amikom.ac.id",
        "https://bpjs-kesehatan.go.id",
        # Phishing
        "http://paypal-login-security.xyz/verify",
        "http://192.168.1.1/bank-login",
        "http://free-money-claim.tk/bonus",
        "http://bca-verifikasi-login.xyz/masuk",
        "http://google-secure-account-verify.com/login",
        "http://tokopedia-bonus-klik.ml/hadiah",
    ]
 
    print(f"{'URL':<55} {'TLDProb':>8} {'Entropy':>8} "
          f"{'URLLen':>7} {'SpecRat':>8} {'IsIP':>5}")
    print("-" * 95)
 
    for u in test_urls:
        f = extract_features(u)
        label = "✓ LEGIT " if f['TLDLegitimateProb'] >= 0.5 else "⚠ SUSPECT"
        print(f"{u[:54]:<55} "
              f"{f['TLDLegitimateProb']:>8.2f} "
              f"{f['URLCharProb']:>8.3f} "
              f"{f['URLLength']:>7} "
              f"{f['SpacialCharRatioInURL']:>8.3f} "
              f"{f['IsDomainIP']:>5}  {label}")
 
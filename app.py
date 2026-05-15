import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tldextract
from urllib.parse import urlparse
 
from src.feature_extraction import extract_features
 
# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    with open("models/feature_list.json") as f:
        fitur = json.load(f)
    return model, fitur
 
model, fitur_tersedia = load_model()
 
# ============================================================
# DOMAIN WHITELIST
# ============================================================
DOMAIN_RESMI = {
    "google", "youtube", "gmail", "github", "gitlab",
    "microsoft", "apple", "amazon", "openai", "anthropic",
    "yahoo", "bing", "wikipedia", "reddit", "stackoverflow",
    "figma", "canva", "notion", "miro", "airtable",
    "trello", "asana", "monday", "clickup", "basecamp",
    "slack", "zoom", "teams", "meet", "webex",
    "dropbox", "box", "drive", "onedrive", "icloud",
    "adobe", "atlassian", "jira", "confluence", "bitbucket",
    "vercel", "netlify", "heroku", "digitalocean", "aws",
    "cloudflare", "firebase", "supabase", "mongodb",
    "npmjs", "pypi", "dockerhub", "docker",
    "ebay", "etsy", "aliexpress", "alibaba",
    "tokopedia", "shopee", "bukalapak", "lazada", "blibli",
    "traveloka", "tiket", "jd",
    "facebook", "instagram", "whatsapp", "threads",
    "twitter", "x", "linkedin", "tiktok", "telegram",
    "discord", "steam", "spotify", "netflix",
    "pinterest", "snapchat", "tumblr", "twitch",
    "vimeo", "dailymotion", "soundcloud",
    "vidio", "rctiplus", "mola", "disneyplus",
    "bca", "klikbca", "mandiri", "bri", "bni", "cimb",
    "danamon", "permatabank", "ocbc", "maybank", "btn",
    "bsi", "bsm",
    "paypal", "ovo", "dana", "linkaja", "gopay", "gojek",
    "grab", "stripe", "wise", "revolut",
    "detik", "kompas", "tribun", "cnnindonesia", "liputan6",
    "tempo", "republika", "antaranews",
    "kumparan", "grid", "okezone", "merdeka", "suara",
    "kemenkeu", "pajak", "bpjs", "kpk", "kominfo",
    "kemdikbud", "dikti", "lapor", "covid19",
    "halodoc", "alodokter", "klikdokter", "hellosehat",
    "jne", "jnt", "sicepat", "anteraja", "pos",
    "medium", "substack", "wordpress", "blogger", "wix",
    "quora", "producthunt", "ycombinator",
}
 
SUSPICIOUS_KEYWORDS = [
    "login", "secure", "verify", "update", "banking",
    "account", "verification", "wallet", "bonus", "gift",
    "claim", "free", "promo", "hadiah", "reward",
    "unlock", "confirm", "reset", "support", "help",
    "signin", "password", "credential", "authenticate",
    "security", "alert", "warning", "notification",
    "invoice", "payment", "transfer", "topup",
]
 
# ============================================================
# MATERI EDUKASI
# ============================================================
MATERI_EDUKASI = [
    {
        "judul": "Apa itu Phishing?",
        "ikon": "🎣",
        "isi": (
            "**Phishing** adalah salah satu bentuk penipuan di internet yang paling sering terjadi. "
            "Pelaku membuat website atau link palsu yang tampilannya sangat mirip dengan situs resmi "
            "seperti bank, toko online, atau media sosial. Tujuannya satu: mencuri data pribadi kamu.\n\n"
            "Data yang sering dicuri antara lain username dan password akun, nomor kartu kredit atau "
            "rekening bank, kode OTP (One-Time Password), hingga data KTP atau informasi pribadi lainnya.\n\n"
            "**Contoh kasus nyata di Indonesia:**\n\n"
            "Kamu menerima SMS yang isinya: *'Akun BCA Anda diblokir, segera verifikasi di "
            "http://bca-login-secure.xyz'*. Link tersebut terlihat seperti BCA asli, "
            "tapi sebenarnya adalah website palsu yang dirancang untuk mencuri data login kamu. "
            "BCA asli menggunakan domain `klikbca.com` atau `bca.co.id`, bukan domain acak seperti itu."
        ),
    },
    {
        "judul": "Ciri-ciri URL Phishing",
        "ikon": "🔍",
        "isi": (
            "URL (alamat website) adalah hal pertama yang harus kamu perhatikan sebelum mengklik sebuah link. "
            "Berikut ciri-ciri URL yang patut dicurigai:\n\n"
            "**1. Nama domain yang aneh atau menyerupai brand terkenal**\n"
            "Misalnya `bca-login-secure.xyz` atau `tokopedia-bonus.ml`. "
            "Domain asli BCA adalah `bca.co.id`, bukan yang mengandung kata tambahan seperti *login*, *secure*, atau *bonus*.\n\n"
            "**2. TLD (akhiran domain) tidak umum**\n"
            "TLD seperti `.tk`, `.ml`, `.xyz`, `.gq`, dan `.cf` sangat murah atau gratis, "
            "sehingga sering digunakan oleh pembuat phishing. "
            "Situs resmi biasanya menggunakan `.com`, `.co.id`, `.go.id`, atau `.org`.\n\n"
            "**3. URL yang sangat panjang dan membingungkan**\n"
            "Phishing sering menggunakan URL panjang seperti "
            "`http://promo-gratis.xyz/tokopedia/login/verifikasi/akun?id=xxx` "
            "untuk menyembunyikan domain aslinya di tengah-tengah teks yang panjang.\n\n"
            "**4. Menggunakan IP address langsung**\n"
            "Contoh: `http://192.168.1.1/login`. Website resmi tidak pernah menggunakan angka seperti ini "
            "sebagai alamat utamanya.\n\n"
            "**5. Subdomain mencurigakan**\n"
            "Contoh: `login.bca.secure.malicious.com`. Domain utama sebenarnya adalah `malicious.com`, "
            "bukan `bca.com`. Selalu baca domain dari kanan ke kiri sebelum tanda `/` pertama.\n\n"
            "**6. Karakter yang disembunyikan (obfuscation)**\n"
            "Beberapa phishing menggunakan kode seperti `%70%61%79%70%61%6C` yang sebenarnya "
            "berarti `paypal` — sengaja disembunyikan agar tidak mudah terdeteksi."
        ),
    },
    {
        "judul": "Cara Melindungi Diri dari Phishing",
        "ikon": "🛡️",
        "isi": (
            "Melindungi diri dari phishing tidak sulit jika kamu tahu caranya. "
            "Berikut langkah-langkah praktis yang bisa langsung kamu terapkan:\n\n"
            "**Sebelum mengklik link:**\n"
            "Arahkan kursor mouse ke atas link (tanpa diklik) untuk melihat URL aslinya di pojok bawah browser. "
            "Jika URL terlihat aneh atau tidak sesuai dengan nama pengirimnya, jangan diklik. "
            "Sebaiknya ketik langsung alamat website di browser daripada mengklik link dari pesan.\n\n"
            "**Saat membuka website:**\n"
            "Selalu periksa nama domain di address bar browser dengan teliti. "
            "Perhatikan bahwa ikon gembok (HTTPS) bukan jaminan keamanan — website phishing pun bisa "
            "menggunakan HTTPS. Yang penting adalah nama domainnya benar. "
            "Jangan pernah memasukkan kode OTP yang tidak kamu minta sendiri.\n\n"
            "**Jika sudah terlanjur mengklik:**\n"
            "Jangan memasukkan data apapun. Tutup halaman tersebut segera. "
            "Jika sudah terlanjur mengisi data, segera ganti password akun yang bersangkutan "
            "dan hubungi bank atau layanan terkait untuk memblokir akun. "
            "Laporkan kejadian ke BSSN di `bssn.go.id` atau hubungi Aduan BRTI di nomor 159."
        ),
    },
    {
        "judul": "Bagaimana AI Mendeteksi Phishing?",
        "ikon": "🤖",
        "isi": (
            "Sistem ini menggunakan kecerdasan buatan (AI) untuk menganalisis URL secara otomatis "
            "tanpa perlu membuka website tersebut. Ini penting karena membuka website phishing saja "
            "sudah bisa berbahaya.\n\n"
            "**Cara kerjanya:**\n"
            "Sistem membaca struktur URL dan mengekstrak 20 karakteristik (fitur) dari URL tersebut, "
            "seperti panjang URL, jenis domain, jumlah karakter khusus, tingkat keacakan teks, "
            "dan lainnya. Kemudian, algoritma **LightGBM** — salah satu metode machine learning "
            "terbaik untuk klasifikasi — membandingkan fitur-fitur tersebut dengan pola yang dipelajari "
            "dari 235.795 URL nyata.\n\n"
            "**Mengapa hasilnya bisa dipercaya?**\n"
            "Model dilatih dengan data yang sangat besar dan dioptimasi menggunakan teknik **Optuna** "
            "untuk menemukan kombinasi parameter terbaik. Selain itu, sistem ini menerapkan "
            "**Explainable AI (XAI)** — artinya AI tidak hanya memberi jawaban, tapi juga "
            "menjelaskan *mengapa* suatu URL dianggap berbahaya atau aman, sehingga kamu bisa "
            "memahami dan belajar darinya."
        ),
    },
    {
        "judul": "Apa itu Explainable AI dan Mengapa Penting?",
        "ikon": "📊",
        "isi": (
            "Bayangkan kamu bertanya kepada seorang ahli keamanan: *'Apakah link ini berbahaya?'* "
            "Jika dia hanya menjawab 'Ya, berbahaya' tanpa penjelasan, kamu tidak akan belajar apa-apa. "
            "Tapi jika dia menjelaskan *'Link ini berbahaya karena menggunakan domain palsu yang meniru BCA, "
            "ditambah TLD .xyz yang tidak lazim untuk bank'* — kamu jadi paham dan bisa mengenalinya sendiri "
            "di masa depan.\n\n"
            "Itulah konsep **Explainable AI (XAI)**. Berbeda dengan AI biasa yang hanya memberi label akhir "
            "(disebut *black box*), XAI memberikan transparansi penuh tentang alasan di balik setiap keputusan.\n\n"
            "**Manfaat XAI dalam sistem ini:**\n"
            "Kamu tidak hanya tahu hasilnya (aman/phishing), tapi juga memahami faktor apa yang membuat URL "
            "tersebut dicurigai. Dengan begitu, lama-kelamaan kamu bisa mengenali phishing sendiri tanpa "
            "harus selalu bergantung pada sistem. Ini adalah tujuan utama dari edukasi keamanan digital — "
            "membangun kesadaran dan kemampuan masyarakat untuk melindungi diri secara mandiri."
        ),
    },
    {
        "judul": "Latihan: Kenali Phishing Sendiri",
        "ikon": "🧪",
        "isi": (
            "Setelah membaca materi di atas, coba uji pemahamanmu. "
            "Perhatikan URL-URL berikut dan pikirkan mana yang aman dan mana yang phishing:\n\n"
            "| URL | Penjelasan |\n"
            "|-----|------------|\n"
            "| `https://bca.co.id` | **Aman** — domain resmi BCA, menggunakan .co.id |\n"
            "| `https://bca-login-secure.xyz` | **Phishing** — meniru BCA dengan kata *login* dan *secure*, TLD .xyz tidak resmi |\n"
            "| `https://tokopedia.com/promo` | **Aman** — domain resmi Tokopedia, path /promo wajar |\n"
            "| `http://tokopedia-bonus-hadiah.ml` | **Phishing** — domain palsu dengan kata *bonus* dan *hadiah*, TLD .ml |\n"
            "| `https://github.com` | **Aman** — platform developer terpercaya |\n"
            "| `http://192.168.1.1/verify` | **Phishing** — menggunakan IP address langsung, bukan nama domain |\n"
            "| `https://paypal-account-verify.tk/login` | **Phishing** — meniru PayPal, banyak kata mencurigakan, TLD .tk |\n\n"
            "**Rumus cepat mengenali phishing:**\n"
            "Baca domain dari kanan ke kiri (sebelum tanda `/` pertama). "
            "Yang paling penting bukan kata-kata di tengah URL, tapi nama domain utamanya. "
            "Contoh: di URL `http://login.bca.secure.malicious.com/verify`, "
            "domain utamanya adalah `malicious.com` — bukan BCA sama sekali!"
        ),
    },
]
 
# ============================================================
# FUNGSI-FUNGSI
# ============================================================
def cek_domain_resmi(url: str) -> bool:
    ext    = tldextract.extract(url)
    domain = ext.domain.lower()
    suffix = ext.suffix.lower()
    full   = f"{domain}.{suffix}"
    if domain in DOMAIN_RESMI:
        return True
    trusted_suffixes = ('go.id', 'ac.id', 'sch.id', 'mil.id')
    if any(full.endswith(s) for s in trusted_suffixes):
        return True
    return False
 
def domain_mencurigakan(url: str) -> tuple:
    ext         = tldextract.extract(url)
    domain      = ext.domain.lower()
    subdomain   = ext.subdomain.lower()
    full_domain = f"{subdomain}.{domain}" if subdomain else domain
    for brand in DOMAIN_RESMI:
        if brand in full_domain and domain != brand:
            for kw in SUSPICIOUS_KEYWORDS:
                if kw in full_domain:
                    return True, f"Domain meniru '{brand}' dengan kata '{kw}'"
            if brand in domain and domain != brand:
                return True, f"Domain menyerupai brand '{brand}'"
    kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_domain)
    if kw_count >= 2:
        found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in full_domain]
        return True, f"Domain mengandung {kw_count} kata mencurigakan: {found[:3]}"
    return False, ""
 
def kemungkinan_aman(features: dict, url: str) -> tuple:
    ext    = tldextract.extract(url)
    suffix = ext.suffix.lower()
    TRUSTED_TLDS = {"com", "net", "org", "io", "co.id", "id", "co", "app", "dev", "tech", "ai"}
    tld_ok          = suffix in TRUSTED_TLDS or features["TLDLegitimateProb"] >= 0.85
    url_pendek      = features["URLLength"] < 50
    entropy_rendah  = features["URLCharProb"] < 4.0
    no_subdomain    = features["NoOfSubDomain"] == 0
    no_obfuscation  = not features["HasObfuscation"]
    no_ip           = not features["IsDomainIP"]
    digit_sedikit   = features["DegitRatioInURL"] < 0.15
    special_sedikit = features["SpacialCharRatioInURL"] < 0.20
    semua_aman = all([tld_ok, url_pendek, entropy_rendah, no_subdomain,
                      no_obfuscation, no_ip, digit_sedikit, special_sedikit])
    if semua_aman:
        return True, "Struktur URL sangat bersih (URL pendek, TLD terpercaya, tidak ada obfuscation)"
    return False, ""
 
 
def buat_penjelasan(features: dict) -> tuple:
    """
    Untuk phishing  : narasi peringatan langsung ke pengguna (jangan lakukan X, karena Y).
    Untuk aman      : narasi mengalir yang menjelaskan mengapa URL tergolong aman.
    Mengembalikan (narasi_masalah, narasi_aman) sebagai string paragraf.
    """
    poin_masalah = []   # dipakai saat phishing → gaya peringatan + larangan
    poin_aman    = []   # dipakai saat aman    → gaya penjelasan positif
 
    # ── helper: buat kalimat peringatan ──────────────────────
    def peringatan(larangan: str, alasan: str) -> str:
        """Gabungkan larangan + alasan menjadi satu kalimat peringatan."""
        return f"{larangan} {alasan}"
 
    # --- Panjang URL ---
    l = features["URLLength"]
    if l > 100:
        poin_masalah.append(peringatan(
            "Jangan masukkan data apapun pada link ini.",
            f"URL ini sengaja dibuat sangat panjang ({l} karakter) agar nama asli situsnya "
            "tersembunyi di balik deretan karakter yang membingungkan — trik klasik yang sering "
            "dipakai penipu supaya korban tidak sadar sedang diarahkan ke situs palsu."
        ))
    elif l > 75:
        poin_masalah.append(peringatan(
            "Waspadai link ini sebelum melakukan apapun.",
            f"URL yang terlalu panjang ({l} karakter) seperti ini kerap digunakan untuk "
            "menyembunyikan tujuan asli dari sebuah link sehingga terlihat sah padahal tidak."
        ))
    else:
        poin_aman.append(
            f"Panjang URL tergolong normal ({l} karakter), "
            "artinya tidak ada upaya menyembunyikan nama domain asli di balik teks yang panjang."
        )
 
    # --- TLD ---
    tld_prob = features["TLDLegitimateProb"]
    if tld_prob >= 0.8:
        poin_aman.append(
            "Akhiran domain yang digunakan — seperti .com, .co.id, atau .go.id — "
            "adalah akhiran yang dipakai mayoritas website resmi di dunia, "
            "sehingga ini menjadi tanda positif bahwa URL bukan phishing."
        )
    elif tld_prob >= 0.4:
        poin_aman.append(
            "Akhiran domain yang digunakan cukup umum dan tidak terlalu mencurigakan, "
            "meski tetap perlu diperhatikan bersama faktor lainnya."
        )
    else:
        poin_masalah.append(peringatan(
            "Jangan percayai link ini hanya karena tampilannya terlihat meyakinkan.",
            "Akhiran domain (TLD) yang digunakan — seperti .tk, .xyz, .ml, atau .gq — "
            "bisa didaftarkan secara gratis sehingga sangat sering disalahgunakan penipu. "
            "Bank, toko online, dan instansi resmi tidak pernah menggunakan akhiran seperti ini."
        ))
 
    # --- Obfuscation ---
    if features["HasObfuscation"]:
        poin_masalah.append(peringatan(
            f"Jangan buka atau klik link ini.",
            f"Ada {features['NoOfObfuscatedChar']} karakter yang sengaja disembunyikan "
            "dalam URL menggunakan kode seperti %70%61%79. "
            "Penipu memakai teknik ini supaya bagian-bagian mencurigakan tidak terbaca, "
            "baik oleh manusia maupun sistem keamanan."
        ))
    else:
        poin_aman.append(
            "Tidak ada karakter yang disembunyikan dalam URL ini — "
            "semua bagian bisa dibaca dengan jelas, tanda URL tidak menyembunyikan apapun."
        )
 
    # --- IP domain ---
    if features["IsDomainIP"]:
        poin_masalah.append(peringatan(
            "Jangan sekali-kali memasukkan data pribadi, password, atau OTP di link ini.",
            "Website ini menggunakan deretan angka (IP address seperti 192.168.1.1) "
            "sebagai alamat utamanya. Tidak ada bank, toko online, atau layanan resmi manapun "
            "yang menggunakan cara seperti ini — ini tanda yang sangat kuat bahwa link ini berbahaya."
        ))
    else:
        poin_aman.append(
            "Website ini menggunakan nama domain yang normal, bukan deretan angka, "
            "sesuai standar semua website resmi yang ada."
        )
 
    # --- Subdomain ---
    ns = features["NoOfSubDomain"]
    if ns > 3:
        poin_masalah.append(peringatan(
            "Jangan tertipu meski nama di link ini terlihat familiar.",
            f"URL ini punya {ns} tingkatan subdomain yang sangat tidak wajar. "
            "Penipu sengaja menambahkan nama brand terkenal di bagian depan, "
            "misalnya login.bca.aman.situspalsu.com — terlihat seperti BCA, "
            "padahal pemilik aslinya adalah situspalsu.com. "
            "Selalu baca nama domain dari kanan ke kiri untuk tahu siapa pemilik aslinya."
        ))
    elif ns > 2:
        poin_masalah.append(peringatan(
            "Periksa dulu siapa pemilik asli website ini sebelum melakukan apapun.",
            f"URL memiliki {ns} tingkatan subdomain yang tergolong banyak. "
            "Baca nama domain dari kanan ke kiri — bagian paling kanan sebelum tanda / "
            "adalah pemilik asli website tersebut."
        ))
    else:
        poin_aman.append(
            "Jumlah subdomain dalam URL ini normal dan tidak menunjukkan adanya upaya "
            "menyamar sebagai website lain."
        )
 
    # --- Rasio angka ---
    dr = features["DegitRatioInURL"]
    if dr > 0.35:
        poin_masalah.append(peringatan(
            "Jangan asumsikan link ini aman hanya karena dikirim oleh seseorang yang kamu kenal.",
            f"Sebanyak {dr:.0%} karakter URL ini adalah angka acak — proporsi yang sangat tidak wajar. "
            "URL seperti ini hampir selalu dibuat oleh program otomatis, bukan manusia, "
            "dan merupakan ciri khas link phishing yang dibuat massal."
        ))
    elif dr > 0.25:
        poin_masalah.append(peringatan(
            "Tetap waspada dengan link ini.",
            f"Proporsi angka dalam URL cukup tinggi ({dr:.0%}), "
            "sedikit di atas rata-rata URL normal."
        ))
 
    # --- Karakter spesial ---
    sr = features["SpacialCharRatioInURL"]
    if sr > 0.35:
        poin_masalah.append(peringatan(
            "Jangan masukkan informasi apapun di website yang dibuka dari link ini.",
            f"URL dipenuhi karakter spesial seperti -, _, %, = ({sr:.0%} dari keseluruhan URL). "
            "Website resmi memiliki URL yang bersih dan mudah dibaca — "
            "bukan deretan karakter aneh seperti ini."
        ))
    elif sr > 0.25:
        poin_masalah.append(peringatan(
            "Hati-hati dengan link ini.",
            f"Karakter spesial dalam URL cukup banyak ({sr:.0%}), "
            "sedikit di atas rata-rata URL normal yang biasa ditemui."
        ))
 
    # --- Tanda tanya ---
    if features["NoOfQMarkInURL"] > 1:
        poin_masalah.append(peringatan(
            "Jangan klik link ini tanpa memverifikasi kebenarannya terlebih dahulu.",
            "Ada lebih dari satu tanda tanya (?) dalam URL ini yang tidak lazim. "
            "Ini bisa menandakan URL menyimpan banyak parameter tersembunyi yang tidak wajar, "
            "yang sering dipakai untuk mengelabui sistem keamanan."
        ))
 
    # --- Tanda sama dengan ---
    eq = features["NoOfEqualsInURL"]
    if eq > 3:
        poin_masalah.append(peringatan(
            "Jangan lanjutkan ke website ini.",
            f"Ada {eq} tanda sama dengan (=) dalam URL — jumlah yang sangat berlebihan. "
            "Parameter sebanyak ini bisa mengindikasikan adanya pengalihan otomatis "
            "ke halaman berbahaya tanpa sepengetahuan pengguna."
        ))
 
    # --- Entropy ---
    entropy = features["URLCharProb"]
    if entropy > 4.5:
        poin_masalah.append(peringatan(
            "Jangan percayai link ini.",
            f"Susunan karakter dalam URL sangat acak dan tidak beraturan (skor: {entropy:.2f}). "
            "URL yang dibuat manusia biasanya memiliki pola yang teratur dan mudah dibaca. "
            "Keacakan setinggi ini hampir selalu berarti URL dibuat oleh program otomatis — "
            "ciri khas link phishing yang disebarkan massal."
        ))
    elif entropy <= 4.0:
        poin_aman.append(
            f"Susunan karakter URL teratur dan mudah dibaca (skor: {entropy:.2f}), "
            "tidak ada indikasi URL ini dibuat secara acak oleh program otomatis."
        )
 
    # ── Gabungkan menjadi satu narasi paragraf mengalir ──────
    def gabung(poin_list):
        if not poin_list:
            return ""
        if len(poin_list) == 1:
            return poin_list[0]
        hasil = poin_list[0]
        penghubung = [
            " Selain itu, ",
            " Lebih lanjut, ",
            " Yang juga perlu diperhatikan, ",
            " Ditambah lagi, ",
            " Satu hal lagi yang ditemukan: ",
        ]
        for i, p in enumerate(poin_list[1:]):
            sambung = penghubung[i % len(penghubung)]
            hasil += sambung + p[0].lower() + p[1:]
        return hasil
 
    return gabung(poin_masalah), gabung(poin_aman)
 
 
def buat_chart_kontribusi(features: dict, pred: int) -> plt.Figure:
    kontribusi = {}
    l = features["URLLength"]
    kontribusi["Panjang URL"]        = min((l - 30) / 100, 1.0) if l > 30 else -0.3
    kontribusi["TLD Prob"]           = -(features["TLDLegitimateProb"] - 0.5) * 2
    kontribusi["Entropy URL"]        = (features["URLCharProb"] - 3.5) / 2.0
    kontribusi["Obfuscation"]        = 0.8 if features["HasObfuscation"] else -0.2
    kontribusi["Domain IP"]          = 1.0 if features["IsDomainIP"] else -0.1
    ns = features["NoOfSubDomain"]
    kontribusi["Jml Subdomain"]      = min(ns * 0.25, 1.0) if ns > 1 else -0.2
    kontribusi["Rasio Angka"]        = (features["DegitRatioInURL"] - 0.1) * 3
    kontribusi["Rasio Kar. Spesial"] = (features["SpacialCharRatioInURL"] - 0.15) * 3
    kontribusi["Jml Tanda '?'"]      = min(features["NoOfQMarkInURL"] * 0.3, 0.9)
    eq = features["NoOfEqualsInURL"]
    kontribusi["Jml Tanda '='"]      = min((eq - 1) * 0.2, 0.9) if eq > 1 else -0.1
 
    items  = sorted(kontribusi.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("← Mendorong AMAN          Mendorong PHISHING →", color="#ccc", fontsize=8)
    ax.tick_params(colors="#ccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val + (0.03 if val >= 0 else -0.03),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center",
                ha="left" if val >= 0 else "right",
                color="#ccc", fontsize=8)
    red_patch   = mpatches.Patch(color="#ef4444", label="Indikasi Phishing")
    green_patch = mpatches.Patch(color="#22c55e", label="Indikasi Aman")
    ax.legend(handles=[red_patch, green_patch], loc="lower right",
              facecolor="#2a2a3e", labelcolor="#ccc", fontsize=8, framealpha=0.8)
    ax.set_title("Kontribusi Fitur terhadap Keputusan (XAI)", color="#fff", fontsize=10, pad=10)
    plt.tight_layout()
    return fig
 
# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Deteksi Link Phishing",
    page_icon="🛡️",
    layout="centered",
)
 
st.markdown("""
<style>
/* ─── Kotak narasi merah (bahaya) ─── */
.narasi-bahaya {
    background: #2d1515;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #fca5a5;
    font-size: 15px;
    line-height: 1.9;
}
.narasi-bahaya b {
    color: #f87171;
    font-size: 15px;
}
/* ─── Kotak narasi hijau (aman) ─── */
.narasi-aman {
    background: #0f2318;
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 12px 0;
    color: #86efac;
    font-size: 15px;
    line-height: 1.9;
}
.narasi-aman b {
    color: #4ade80;
    font-size: 15px;
}
/* ─── Hasil box ─── */
.hasil-phishing {
    background: #fee2e2; border: 2px solid #ef4444;
    border-radius: 12px; padding: 20px; margin: 10px 0; color: #1a1a1a;
}
.hasil-phishing p, .hasil-phishing h2 { color: #1a1a1a !important; }
.hasil-aman {
    background: #dcfce7; border: 2px solid #22c55e;
    border-radius: 12px; padding: 20px; margin: 10px 0; color: #1a1a1a;
}
.hasil-aman p, .hasil-aman h2 { color: #1a1a1a !important; }
/* ─── Info box ─── */
.info-box {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-radius: 10px; padding: 15px; margin: 10px 0; color: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)
 
# ── HEADER ──────────────────────────────────────────────────
st.title("🛡️ Sistem Deteksi Link Phishing")
st.caption("")
 
tab_cek, tab_edukasi, tab_tentang = st.tabs([
    "🔍 Cek URL",
    "📚 Edukasi Keamanan Digital",
    " ",
])
 
# ════════════════════════════════════════════════════════════
# TAB 1: CEK URL
# ════════════════════════════════════════════════════════════
with tab_cek:
    url_input = st.text_input(
        "🔗 Masukkan URL:",
        placeholder="Contoh: https://tokopedia.com  atau  http://free-money.tk/claim",
    )
    cek = st.button("🔍 Cek URL", type="primary", use_container_width=True)
 
    if cek:
        if not url_input.strip():
            st.warning("⚠️ Masukkan URL terlebih dahulu.")
        else:
            try:
                with st.spinner("Menganalisis URL..."):
                    url = url_input.strip()
                    if not url.startswith(("http://", "https://")):
                        url = "http://" + url
 
                    features  = extract_features(url)
                    df_input  = pd.DataFrame([features])
                    df_input  = df_input.reindex(columns=fitur_tersedia, fill_value=0)
 
                    prob          = model.predict_proba(df_input)[0]
                    prob_phishing = float(prob[0])
                    prob_aman     = float(prob[1])
                    pred_raw      = int(model.predict(df_input)[0])
 
                    trusted                     = cek_domain_resmi(url)
                    is_spoof, spoof_alasan      = domain_mencurigakan(url)
                    is_likely_safe, safe_alasan = kemungkinan_aman(features, url)
 
                    if is_spoof:
                        pred          = 0
                        prob_phishing = max(prob_phishing, 0.97)
                        sumber        = f"Brand spoofing: {spoof_alasan}"
                    elif trusted:
                        pred      = 1
                        prob_aman = max(prob_aman, 0.95)
                        sumber    = "Whitelist domain terpercaya"
                    elif is_likely_safe:
                        pred      = 1
                        prob_aman = max(prob_aman, 0.80)
                        sumber    = f"Heuristik aman: {safe_alasan}"
                    else:
                        pred   = 0 if prob_phishing >= 0.75 else 1
                        conf   = max(prob_phishing, prob_aman) * 100
                        sumber = f"Model ML (keyakinan: {conf:.1f}%)"
 
                    narasi_masalah, narasi_aman = buat_penjelasan(features)
 
                # ── HASIL ──────────────────────────────────
                st.markdown("---")
                if pred == 0:
                    st.markdown(f"""
                    <div class="hasil-phishing">
                        <h2>⚠️ LINK TERDETEKSI PHISHING</h2>
                        <p style="font-size:18px">Tingkat keyakinan phishing: <b>{prob_phishing*100:.1f}%</b></p>
                        <p>Hindari memasukkan password, OTP, atau data pribadi pada link ini.</p>
                        <p style="font-size:13px;color:#666">Sumber keputusan: {sumber}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="hasil-aman">
                        <h2>✅ LINK TERDETEKSI AMAN</h2>
                        <p style="font-size:18px">Tingkat keamanan: <b>{prob_aman*100:.1f}%</b></p>
                        <p>URL tidak menunjukkan pola phishing yang kuat.</p>
                        <p style="font-size:13px;color:#666">Sumber keputusan: {sumber}</p>
                    </div>""", unsafe_allow_html=True)
 
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Panjang URL",  f"{features['URLLength']} kar")
                m2.metric("TLD Prob",     f"{features['TLDLegitimateProb']:.2f}")
                m3.metric("Entropy",      f"{features['URLCharProb']:.2f}")
                m4.metric("Subdomain",    features['NoOfSubDomain'])
 
                st.markdown("---")
 
                # ── PENJELASAN ANALISIS — narasi paragraf ──
                st.subheader("📋 Penjelasan Analisis")
                st.caption(
                    "Berikut penjelasan mengapa URL ini dinilai aman atau berbahaya, "
                    "ditulis dalam bahasa yang mudah dipahami."
                )
 
                if pred == 0:
                    # ── Susun teks peringatan phishing ──
                    teks_bahaya = ""
                    if is_spoof:
                        # Brand spoofing → tambahkan penjelasan spoof di depan
                        teks_bahaya = (
                            f"Jangan buka atau masukkan data apapun di link ini! "
                            f"URL ini terdeteksi meniru layanan resmi — {spoof_alasan.lower()}. "
                            "Penipu sengaja membuat nama domain yang sangat mirip dengan brand asli "
                            "supaya korban mengira sedang mengunjungi website yang sah."
                        )
                        if narasi_masalah:
                            teks_bahaya += " Selain itu, " + narasi_masalah[0].lower() + narasi_masalah[1:]
                    else:
                        teks_bahaya = narasi_masalah
 
                    if teks_bahaya:
                        st.markdown(
                            f'<div class="narasi-bahaya">'
                            f'🚨 <b>Peringatan — Apa yang Harus Kamu Lakukan?</b><br><br>'
                            f'{teks_bahaya}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
 
                    if narasi_aman:
                        st.markdown(
                            f'<div class="narasi-aman">'
                            f'ℹ️ <b>Catatan: Meski begitu, ada sisi yang tergolong normal</b><br><br>'
                            f'{narasi_aman}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
 
                else:
                    if narasi_aman:
                        st.markdown(
                            f'<div class="narasi-aman">'
                            f'✅ <b>Mengapa URL ini aman?</b><br><br>'
                            f'{narasi_aman}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
 
                    if narasi_masalah:
                        st.markdown(
                            f'<div class="narasi-bahaya">'
                            f'⚠️ <b>Tetap perhatikan hal ini:</b><br><br>'
                            f'{narasi_masalah}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
 
                st.subheader("🛡️ Tips Menghindari Phishing")
                st.markdown("""<div class="info-box">
                ✅ Periksa nama domain dengan teliti sebelum mengklik<br>
                ✅ Hindari login melalui link yang dikirim via WhatsApp, SMS, atau email<br>
                ✅ Jangan masukkan kode OTP yang tidak kamu minta sendiri<br>
                ✅ <code>bca-login-secure.xyz</code> bukan BCA asli — BCA asli: <code>klikbca.com</code> atau <code>bca.co.id</code><br>
                ✅ Website resmi bank Indonesia menggunakan <code>.co.id</code> atau <code>.id</code><br>
                ✅ HTTPS (gembok hijau) saja bukan jaminan aman — phishing pun bisa menggunakan HTTPS<br>
                ✅ Pelajari lebih lanjut di tab <b>📚 Edukasi Keamanan Digital</b>
                </div>""", unsafe_allow_html=True)
 
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Pastikan URL valid. Contoh: https://www.google.com")
 
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Contoh URL Aman")
            for u in ["https://google.com", "https://tokopedia.com",
                      "https://bri.co.id", "https://amikom.ac.id",
                      "https://bpjs-kesehatan.go.id", "https://github.com",
                      "https://figma.com", "https://canva.com"]:
                st.code(u)
        with col2:
            st.markdown("#### ⚠️ Contoh URL Phishing")
            for u in ["http://paypal-login-security.xyz",
                      "http://free-money.tk/claim",
                      "http://192.168.1.1/verify",
                      "http://bca-login-secure.xyz",
                      "http://google-secure-verify.com/login",
                      "http://tokopedia-bonus-hadiah.ml"]:
                st.code(u)
 
        st.markdown("""<div class="info-box">
        <b>ℹ️ Cara Kerja Sistem:</b><br><br>
        1. Masukkan URL apapun yang ingin diperiksa<br>
        2. Sistem mengekstrak 20 fitur dari struktur URL — tanpa membuka website tersebut<br>
        3. Model LightGBM memberi prediksi berdasarkan 235.795 URL data latih<br>
        4. Sistem memeriksa apakah ada peniruan brand resmi (brand spoofing)<br>
        5. Hasil dilengkapi penjelasan agar kamu memahami <b>mengapa</b> URL dinilai aman atau phishing<br>
        6. Kunjungi tab <b>📚 Edukasi</b> untuk belajar mengenali phishing secara mandiri
        </div>""", unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════════════════════
# TAB 2: EDUKASI KEAMANAN DIGITAL
# ════════════════════════════════════════════════════════════
with tab_edukasi:
    st.header("📚 Edukasi Keamanan Digital")
    st.markdown(
        "Halaman ini berisi materi edukasi tentang keamanan digital, khususnya tentang phishing. "
        "Tujuannya agar masyarakat **memahami dan bisa mengenali phishing secara mandiri** — "
        "bukan hanya bergantung pada sistem AI. "
        "Materi di sini bersifat umum dan tidak berkaitan dengan URL yang sedang kamu cek."
    )
    st.info(
        "💡 Setelah membaca materi di bawah, coba gunakan tab **🔍 Cek URL** dan perhatikan "
        "penjelasan analisisnya. Kamu akan mulai bisa mengenali pola phishing sendiri!"
    )
    st.markdown("---")
 
    for materi in MATERI_EDUKASI:
        with st.expander(f"{materi['ikon']}  {materi['judul']}", expanded=False):
            st.markdown(materi["isi"])
 
 

## # ════════════════════════════════════════════════════════════
## # TAB 3: TENTANG SISTEM
## # ════════════════════════════════════════════════════════════
## ## with tab_tentang:
  ## ## st.header(" ") ##
    ## ##st.markdown("""
##**Judul:** Sistem Deteksi Link Phishing Berbasis LightGBM dengan Pendekatan
##Explainable AI (SHAP) untuk Edukasi Keamanan Digital
 
##---
 
###### 🧠 Model Machine Learning
## - **Algoritma:** LightGBM (Light Gradient Boosting Machine)
## - **Optimasi Hyperparameter:** Optuna (Bayesian Optimization)
## - **Dataset:** PhiUSIIL Phishing URL Dataset — 235.795 URL
## - **Fitur:** 20 fitur berbasis struktur URL (tanpa mengakses konten website)
 
#### 🔍 Explainable AI (XAI)
## Sistem ini menerapkan prinsip XAI agar keputusan model bisa dipahami manusia:
## - Setiap prediksi disertai **penjelasan narasi** dalam bahasa yang mudah dipahami masyarakat awam
##- Penjelasan ditulis mengalir seperti kalimat manusia, bukan daftar teknis
## - Tab **Edukasi Keamanan Digital** mendidik pengguna untuk mengenali phishing secara mandiri
 
#### 🏗️ Arsitektur Keputusan
## ```
## URL Input
  ##  ↓
## [1] Brand Spoofing Check  →  PHISHING (jika meniru brand resmi)
 ##   ↓
## [2] Domain Whitelist      →  AMAN (jika domain terdaftar resmi)
  ##  ↓
## [3] Heuristik Struktur    →  AMAN (jika URL sangat bersih)
   ## ↓
## [4] Model LightGBM        →  PHISHING (prob ≥ 0.75) / AMAN
## ```
 
#### 📊 Fitur yang Dianalisis
##  | Kategori | Fitur |
## |----------|-------|
## | Panjang | URLLength, DomainLength, TLDLength |
## | Struktur | IsDomainIP, NoOfSubDomain |
## | Karakter | LetterRatio, DegitRatioInURL, SpacialCharRatioInURL |
## | Parameter | NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL |
##| Pola | URLCharProb, CharContinuationRate |
##| Keamanan | HasObfuscation, NoOfObfuscatedChar, ObfuscationRatio |
## | TLD | TLDLegitimateProb |
##    """)
 
# ── FOOTER ───────────────────────────────────────────────────
## st.markdown("---")
st.caption("🛡️ Sistem Deteksi Link Phishing versi Gue")
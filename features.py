import re
import numpy as np
import pandas as pd

UNIT = {"L": 1e5, "Cr": 1e7}

# keyword -> city, checked in order (first match wins) against the lower-cased region name
CITY_RULES = [
    ("Thane", ["thane", "kalwa", "mumbra", "diva", "ghodbunder", "kalher", "manpada", "hiranandani estates",
               "owale", "panch pakhdi", "kasheli", "vevoor", "patlipada", "shil phata", "gauripada", "kewale", "bhoiwada"]),
    ("Panvel", ["panvel", "kamothe", "kalamboli", "taloja", "taloje", "ulwe", "dronagiri", "uran", "roadpali",
                "karanjade", "khanda colony", "rasayani", "khopoli", "pen", "navade", "koproli"]),
    ("Navi Mumbai", ["kharghar", "vashi", "nerul", "airoli", "belapur", "seawoods", "sanpada", "ghansoli",
                     "koper khairane", "koparkhairane", "rabale", "juinagar", "sector", "greater khanda"]),
    ("Mira-Bhayandar", ["mira road", "bhayandar"]),
    ("Vasai-Virar", ["vasai", "virar", "nala sopara", "nalasopara", "naigaon", "palghar", "boisar", "naupada",
                     "agashi", "umroli", "saphale", "vangani", "warai"]),
    ("Kalyan-Dombivli", ["kalyan", "dombivali", "dombivli", "ambernath", "ambarnath", "badlapur", "ulhasnagar",
                         "titwala", "shahapur", "karjat", "neral", "asangaon", "vasind", "bhiwandi", "palava",
                         "nilje", "anjurdive", "khatiwali", "palidevad", "usarghar", "khardi", "vichumbe", "adaigaon", "shelu", "ambivali"]),
    ("Mumbai - South", ["colaba", "cuffe", "fort", "churchgate", "marine lines", "malabar", "girgaon", "tardeo",
                        "mahalaxmi", "worli", "lower parel", "parel", "byculla", "agripada", "mumbai central",
                        "mazagaon", "mazgaon", "napeansea", "peddar", "babulnath", "prabhadevi", "sewri",
                        "kamathipura", "pestom", "ashok nagar", "mahim", "dadar", "matunga", "dharavi", "sion",
                        "wadala", "gtb nagar", "antop hill", "hindu colony", "rambaug"]),
    ("Mumbai - Eastern Suburbs", ["chembur", "ghatkopar", "kurla", "vikhroli", "vikroli", "kanjurmarg", "bhandup",
                                  "mulund", "powai", "govandi", "deonar", "chandivali", "nahur", "tilak nagar",
                                  "l i c colony", "mahavir", "gandhar", "unnat", "police colony", "maneklal"]),
    ("Mumbai - Western Suburbs", ["bandra kurla", "bandra", "khar", "santacruz", "juhu", "vile parle", "ville parle", "andheri",
                                  "versova", "jogeshwari", "goregaon", "malad", "kandivali", "borivali", "dahisar",
                                  "pali hill", "manjarli", "gorai", "mumbai", "uttan", "kasaradavali"]),
]
CITIES = [c for c, _ in CITY_RULES]


def region_clean(r: str) -> str:
    return " ".join(str(r).strip().split()).title()


def _has(r: str, k: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(k), r) is not None  # word-start match: 'diva' must not hit 'Kandivali'


def city_of(region: str) -> str:
    r = region.lower()
    if "bandra kurla" in r:
        return "Mumbai - Western Suburbs"
    if r.startswith("sector"):  # "Sector 21 Kamothe" etc. -> decide from node name
        for city, keys in CITY_RULES:
            if any(_has(r, k) for k in keys if k != "sector"):
                return city
        return "Navi Mumbai"
    for city, keys in CITY_RULES:
        if any(_has(r, k) for k in keys):
            return city
    return "Other MMR"


def load_raw(path="data/mumbai.csv") -> pd.DataFrame:
    d = pd.read_csv(path)
    d["price_inr"] = d["price"] * d["price_unit"].map(UNIT)
    d["region"] = d["region"].map(region_clean)
    d["region"] = d["region"].replace({"Mumbai": "Mumbai Suburbs", "Sector": "Navi Mumbai Sector"})
    return d


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw listing columns (bhk, type, locality, area, region, status, age) -> model features."""
    d = pd.DataFrame(index=df.index)
    region = df["region"].map(region_clean)
    d["region"] = region
    d["city"] = region.map(city_of)
    d["locality"] = df["locality"].astype(str).str.strip()
    d["type"] = df["type"]
    d["status"] = df["status"]
    d["age"] = df["age"]
    d["bhk"] = df["bhk"].astype(float)
    d["area"] = df["area"].astype(float)
    d["log_area"] = np.log(d["area"])
    d["area_per_bhk"] = d["area"] / d["bhk"].clip(lower=1)
    d["is_ready"] = (d["status"] == "Ready to move").astype(int)
    d["city_status"] = d["city"] + "|" + d["status"]
    d["region_bhk"] = d["region"] + "|" + d["bhk"].astype(int).astype(str)
    return d

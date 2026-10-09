# ============================================================
# ANIMAL AI
# CNN NHẬN DIỆN 200 LOÀI ĐỘNG VẬT + GEMINI CHATBOT
# ============================================================

import os
import json
import html
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms
from dotenv import load_dotenv


# ============================================================
# 1. CẤU HÌNH PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Ưu tiên model V6 và class map đi kèm với model V6.
# Nếu các file V6 không tồn tại, tự dùng tên file của phiên bản cũ.
MODEL_PATH_CANDIDATES = [
    BASE_DIR / "models" / "animal_ai_efficientnet_b0_200_v6_best.pth",
    BASE_DIR / "models" / "animal_ai_efficientnet_b0_200_best.pth",
]

CLASS_NAMES_CANDIDATES = [
    BASE_DIR / "models" / "class_names_200_v6_train.json",
    BASE_DIR / "models" / "class_names_200.json",
]

SPECIES_CSV_CANDIDATES = [
    BASE_DIR / "dataset" / "selected_200_species_v4.csv",
    BASE_DIR / "data" / "dataset_plan.csv",
]

MODEL_PATH = next(
    (path for path in MODEL_PATH_CANDIDATES if path.exists()),
    MODEL_PATH_CANDIDATES[0],
)

CLASS_NAMES_PATH = next(
    (path for path in CLASS_NAMES_CANDIDATES if path.exists()),
    CLASS_NAMES_CANDIDATES[0],
)

SPECIES_CSV_PATH = next(
    (path for path in SPECIES_CSV_CANDIDATES if path.exists()),
    SPECIES_CSV_CANDIDATES[0],
)

IMAGE_SIZE = 192
TOP_K = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Animal AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. ENV
# ============================================================

load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# 4. CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        linear-gradient(
            135deg,
            #f8f7ff 0%,
            #ffffff 50%,
            #f5f9ff 100%
        );
}

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #272343;
    line-height: 1.15;
    margin-bottom: 5px;
}

.subtitle {
    color: #6b6e80;
    font-size: 17px;
    margin-bottom: 25px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero-card {
    background:
        linear-gradient(
            135deg,
            #f0efff,
            #f8f7ff,
            #eef7ff
        );

    border: 1px solid #e3e1f4;
    border-radius: 22px;

    padding: 28px 32px;
    margin-bottom: 25px;

    box-shadow:
        0 10px 30px rgba(69, 32, 197, 0.07);
}

.hero-title {
    font-size: 24px;
    font-weight: 800;
    color: #312e5b;
    margin-bottom: 10px;
}

.hero-text {
    font-size: 15px;
    line-height: 1.75;
    color: #626579;
}


/* =========================================================
   RESULT
   ========================================================= */

.animal-card {
    padding: 24px;
    border-radius: 20px;

    background: white;

    border: 1px solid #e6e4f1;

    margin-top: 10px;
    margin-bottom: 18px;

    box-shadow:
        0 8px 25px rgba(30, 30, 70, 0.06);
}

.prediction-label {
    font-size: 13px;
    color: #77798a;
    margin-bottom: 5px;
}

.prediction-main {
    font-size: 29px;
    font-weight: 800;
    color: #3420a8;
    margin-bottom: 5px;
}

.scientific-name {
    font-size: 18px;
    font-style: italic;
    color: #555b6d;
}

.confidence {
    font-size: 18px;
    font-weight: 700;
    color: #2f7d55;
    margin-top: 12px;
}


/* =========================================================
   INFO
   ========================================================= */

.info-card {
    padding: 20px;

    border-radius: 18px;

    background: white;

    border: 1px solid #e5e7eb;

    margin-bottom: 15px;

    box-shadow:
        0 6px 20px rgba(30, 30, 70, 0.04);
}

.info-title {
    font-size: 18px;
    font-weight: 750;
    color: #38345c;
    margin-bottom: 10px;
}

.info-text {
    font-size: 15px;
    line-height: 1.45;
    color: #606476;
}

/* Phân loại: các dòng hiển thị sát nhau, không tạo khoảng trống lớn */
.taxonomy-row {
    margin: 0 0 5px;
    padding: 0;
    line-height: 1.45;
}

.taxonomy-row:last-child {
    margin-bottom: 0;
}

.taxonomy-row strong {
    color: #45415f;
}


/* =========================================================
   TOP 5
   ========================================================= */

.top-card {
    padding: 5px 5px;

    background: white;

    border: 1px solid #e6e4f1;

    border-radius: 18px;

    box-shadow:
        0 6px 20px rgba(30, 30, 70, 0.04);
}

.top-row {
    display: flex;
    align-items: center;

    padding: 13px 16px;

    border-bottom: 1px solid #eeeeF4;
}

.top-row:last-child {
    border-bottom: none;
}

.top-rank {
    width: 45px;

    font-weight: 800;

    color: #4b3bc7;
}

.top-name {
    flex: 1;

    color: #4d5060;

    font-size: 14px;
}

.top-score {
    font-weight: 750;

    color: #38345c;

    min-width: 70px;

    text-align: right;
}


/* =========================================================
   CHAT
   ========================================================= */

.chat-info {
    padding: 16px 20px;
    border-radius: 17px;
    background: #f5f3ff;
    border: 1px solid #e4dfff;
    margin-bottom: 18px;
    color: #55536b;
    line-height: 1.5;
    box-sizing: border-box;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #f8f7ff,
            #ffffff
        );
}


/* =========================================================
   UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: white;

    border-radius: 16px;

    padding: 8px;

    border: 1px dashed #c9c4ea;
}


/* =========================================================
   WIKIPEDIA: THÔNG TIN + ẢNH MINH HỌA
   ========================================================= */

.wiki-card {
    width: 100%;
    padding: 20px;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
    box-shadow: 0 6px 20px rgba(30, 30, 70, 0.04);
    box-sizing: border-box;
    overflow: hidden;
}

.wiki-title {
    font-size: 18px;
    font-weight: 750;
    color: #38345c;
    line-height: 1.4;
    margin: 0 0 14px;
}

.wiki-layout {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
}

.wiki-image {
    display: block;
    width: 150px;
    min-width: 150px;
    max-width: 40%;
    height: 125px;
    object-fit: cover;
    object-position: center;
    border-radius: 12px;
    border: 1px solid #eceaf4;
    background: #f8f7ff;
    flex-shrink: 0;
}

.wiki-content {
    flex: 1;
    min-width: 0;
    overflow-wrap: anywhere;
}

.wiki-name {
    font-size: 17px;
    font-weight: 700;
    color: #403781;
    line-height: 1.45;
    margin: 0 0 8px;
    overflow-wrap: anywhere;
}

.wiki-summary {
    font-size: 14px;
    font-weight: 600;
    color: #555b6d;
    line-height: 1.6;
    margin: 0 0 8px;
    overflow-wrap: anywhere;
}

.wiki-description {
    font-size: 14px;
    color: #606476;
    line-height: 1.65;
    margin: 0;
    overflow-wrap: anywhere;
}

.wiki-layout.wiki-no-image {
    display: block;
}

.wiki-no-image .wiki-content {
    width: 100%;
}

@media (max-width: 600px) {
    .wiki-card {
        padding: 16px;
    }

    .wiki-layout {
        flex-direction: column;
        gap: 12px;
    }

    .wiki-image {
        width: 100%;
        min-width: 0;
        max-width: 100%;
        height: auto;
        max-height: 240px;
        object-fit: contain;
    }

    .wiki-name {
        font-size: 16px;
    }

    .wiki-summary,
    .wiki-description {
        font-size: 13px;
        line-height: 1.55;
    }
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    border-radius: 12px;

    border: 1px solid #ddd9f3;

    background: white;

    color: #4b3bc7;

    font-weight: 600;
}

.stButton > button:hover {
    border-color: #6c5ce7;

    background: #f8f6ff;

    color: #3d2cb5;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 5. SESSION STATE
# ============================================================

if "current_animal" not in st.session_state:
    st.session_state.current_animal = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "gemini_interaction_id" not in st.session_state:
    st.session_state.gemini_interaction_id = None


# ============================================================
# 6. GEMINI SDK
# ============================================================

try:

    from google import genai

    GEMINI_SDK_AVAILABLE = True

except Exception:

    genai = None

    GEMINI_SDK_AVAILABLE = False


GEMINI_MODEL = "gemini-3.6-flash"


def get_api_key() -> Optional[str]:

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key:
        return (
            key
            .strip()
            .strip('"')
            .strip("'")
        )

    try:

        key = st.secrets.get(
            "GEMINI_API_KEY"
        )

        if key:
            return (
                str(key)
                .strip()
                .strip('"')
                .strip("'")
            )

    except Exception:
        pass

    return None


GEMINI_API_KEY = get_api_key()


@st.cache_resource
def create_gemini_client(
    api_key
):

    if not GEMINI_SDK_AVAILABLE:
        return None

    if not api_key:
        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


GEMINI_CLIENT = create_gemini_client(
    GEMINI_API_KEY
)


# ============================================================
# 7. LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        return []

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception:

        return []

    if isinstance(
        data,
        list
    ):

        return data

    if isinstance(
        data,
        dict
    ):

        if "classes" in data:

            return data[
                "classes"
            ]

        try:

            return [
                data[str(i)]
                for i in range(
                    len(data)
                )
            ]

        except Exception:

            return list(
                data.values()
            )

    return []


CLASS_NAMES = load_class_names()


# ============================================================
# 8. LOAD SPECIES CSV
# ============================================================

@st.cache_data
def load_species_data():

    if not SPECIES_CSV_PATH.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            SPECIES_CSV_PATH
        )

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        return df

    except Exception:

        return pd.DataFrame()


SPECIES_DF = load_species_data()


# ============================================================
# 9. HELPER
# ============================================================

def clean_text(
    value
):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:
        pass

    return str(
        value
    ).strip()


def normalize_species_label(value):
    """Chuyển nhãn dataset như 053_Mammals_Procyon_lotor thành tên khoa học."""
    value = clean_text(value)
    if not value:
        return value

    parts = value.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]

    group_prefixes = {
        "birds", "bird", "aves",
        "mammals", "mammal", "mammalia",
        "reptiles", "reptile", "reptilia",
    }
    if parts and parts[0].lower() in group_prefixes:
        parts = parts[1:]

    # Chỉ đổi nhãn có dấu gạch dưới; tên khoa học dạng chuẩn được giữ nguyên.
    if len(parts) > 1:
        return " ".join(parts).strip()
    return value


def get_species_row(
    species_name
):

    if SPECIES_DF.empty:
        return None

    if "scientific_name" not in SPECIES_DF.columns:
        return None

    target = clean_text(
        species_name
    ).lower()

    for _, row in SPECIES_DF.iterrows():

        current = clean_text(
            row.get(
                "scientific_name",
                ""
            )
        ).lower()

        if current == target:
            return row

    return None


def get_common_name(
    row,
    species_name
):

    if row is None:
        return species_name

    columns = [
        "common_name",
        "common",
        "name"
    ]

    for column in columns:

        if column in row.index:

            value = clean_text(
                row[column]
            )

            if value:
                return value

    return species_name


def get_group_from_folder(
    value
):

    value = str(
        value
    )

    if "_Aves_" in value:
        return "Chim"

    if "_Mammalia_" in value:
        return "Động vật có vú"

    if "_Reptilia_" in value:
        return "Bò sát"

    return "Không xác định"


def get_group(
    row,
    species_name
):

    if (
        row is not None
        and "group" in row.index
    ):

        group = clean_text(
            row["group"]
        )

        mapping = {
            "Birds": "Chim",
            "Mammals": "Động vật có vú",
            "Reptiles": "Bò sát",
        }

        if group:
            return mapping.get(
                group,
                group
            )

    return get_group_from_folder(
        species_name
    )


def get_taxonomy(
    row
):
    """Đọc phân loại từ CSV, chuẩn hóa các tên cột thường gặp."""
    if row is None:
        return {}

    result = {}
    aliases = {
        "kingdom": ["kingdom", "gioi", "giới"],
        "phylum": ["phylum", "nganh", "ngành"],
        "class": ["class", "lop", "lớp"],
        "order": ["order", "bo", "bộ"],
        "family": ["family", "ho", "họ"],
        "genus": ["genus", "chi"],
    }
    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in row.index
    }

    for field, names in aliases.items():
        for name in names:
            actual_column = normalized_columns.get(name)
            if actual_column is not None:
                value = clean_text(row[actual_column])
                if value:
                    result[field] = value
                    break

    # Hỗ trợ cột taxonomy dạng "kingdom: Animalia, phylum: Chordata, ..."
    taxonomy_column = normalized_columns.get("taxonomy")
    if taxonomy_column is not None:
        taxonomy_text = clean_text(row[taxonomy_column])
        if taxonomy_text:
            for item in taxonomy_text.split(","):
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                for field, names in aliases.items():
                    if key in names and value:
                        result.setdefault(field, value)
                        break

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def wikidata_taxonomy(scientific_name):
    """Lấy phân loại theo tên khoa học chính xác từ GBIF, rồi Wikidata."""
    scientific_name = clean_text(scientific_name)
    if not scientific_name:
        return {}

    user_agent = "AnimalAI/1.0 (educational project)"
    target_name = " ".join(scientific_name.casefold().split())
    fields = ("kingdom", "phylum", "class", "order", "family", "genus")

    # Cách 1: GBIF Species API. Chỉ dùng kết quả khớp tên khoa học,
    # không dùng kết quả gần giống để tránh gán nhầm phân loại.
    try:
        response = requests.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": scientific_name, "strict": "true"},
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        returned_names = {
            " ".join(clean_text(data.get(key, "")).casefold().split())
            for key in ("scientificName", "canonicalName")
            if clean_text(data.get(key, ""))
        }
        if target_name in returned_names:
            result = {
                field: clean_text(data.get(field, ""))
                for field in fields
                if clean_text(data.get(field, ""))
            }
            if result:
                return result
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass

    rank_ids = {
        "Q36732": "kingdom",
        "Q38348": "phylum",
        "Q37517": "class",
        "Q36602": "order",
        "Q35409": "family",
        "Q34740": "genus",
    }

    # Cách 2: tìm item Wikidata có P225 (taxon name) khớp chính xác.
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": scientific_name,
                "language": "en",
                "uselang": "en",
                "type": "item",
                "limit": 20,
                "format": "json",
            },
            headers={"User-Agent": user_agent},
            timeout=10,
        )
        response.raise_for_status()
        entity_id = None
        for item in response.json().get("search", []):
            item_id = item.get("id", "")
            if not item_id:
                continue
            entity_response = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": item_id,
                    "props": "claims|labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": user_agent},
                timeout=10,
            )
            entity_response.raise_for_status()
            entity = entity_response.json().get("entities", {}).get(item_id, {})
            exact_names = [
                claim.get("mainsnak", {}).get("datavalue", {}).get("value", "")
                for claim in entity.get("claims", {}).get("P225", [])
            ]
            if any(
                " ".join(clean_text(name).casefold().split()) == target_name
                for name in exact_names
            ):
                entity_id = item_id
                break

        if not entity_id:
            return {}

        result = {}
        visited = set()
        current_id = entity_id
        for _ in range(25):
            if not current_id or current_id in visited:
                break
            visited.add(current_id)
            entity_response = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": current_id,
                    "props": "claims|labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": user_agent},
                timeout=10,
            )
            entity_response.raise_for_status()
            entity = entity_response.json().get("entities", {}).get(current_id, {})
            claims = entity.get("claims", {})
            rank_claims = claims.get("P105", [])
            rank_id = ""
            if rank_claims:
                rank_id = (
                    rank_claims[0].get("mainsnak", {})
                    .get("datavalue", {}).get("value", {}).get("id", "")
                )
            field = rank_ids.get(rank_id)
            label = clean_text(entity.get("labels", {}).get("en", {}).get("value", ""))
            if field and label:
                result.setdefault(field, label)
            parent_claims = claims.get("P171", [])
            if not parent_claims:
                break
            current_id = (
                parent_claims[0].get("mainsnak", {})
                .get("datavalue", {}).get("value", {}).get("id", "")
            )
        if result:
            return result
    except (requests.RequestException, ValueError, TypeError, AttributeError, KeyError):
        pass

    # Cách 3: SPARQL dự phòng. Chọn giá trị theo rank ID thay vì nhãn rank,
    # tránh lỗi do nhãn ngôn ngữ hoặc thứ tự kết quả không ổn định.
    try:
        escaped_name = json.dumps(scientific_name, ensure_ascii=False)
        query = (
            "SELECT ?rank ?taxonLabel WHERE { "
            "?species wdt:P225 ?scientificName. "
            f"FILTER(LCASE(STR(?scientificName)) = LCASE({escaped_name})) "
            "?species wdt:P171* ?taxon. "
            "?taxon wdt:P105 ?rank. "
            "VALUES ?rank { wd:Q36732 wd:Q38348 wd:Q37517 wd:Q36602 wd:Q35409 wd:Q34740 } "
            "?taxon rdfs:label ?taxonLabel. FILTER(LANG(?taxonLabel) = 'en') "
            "}"
        )
        response = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers={"User-Agent": user_agent, "Accept": "application/sparql-results+json"},
            timeout=12,
        )
        response.raise_for_status()
        result = {}
        for item in response.json().get("results", {}).get("bindings", []):
            rank_uri = clean_text(item.get("rank", {}).get("value", ""))
            rank_id = rank_uri.rsplit("/", 1)[-1]
            field = rank_ids.get(rank_id)
            label = clean_text(item.get("taxonLabel", {}).get("value", ""))
            if field and label:
                result.setdefault(field, label)
        return result
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return {}


# ============================================================
# 10. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    if not CLASS_NAMES:
        return None

    try:

        model = models.efficientnet_b0(
            weights=None
        )

        num_classes = len(
            CLASS_NAMES
        )

        model.classifier[1] = (
            torch.nn.Linear(
                model.classifier[1].in_features,
                num_classes
            )
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        if isinstance(
            checkpoint,
            dict
        ):

            if (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            elif (
                "state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "state_dict"
                    ]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith(
                "module."
            ):

                new_key = new_key[
                    7:
                ]

            if new_key.startswith(
                "group_head"
            ):

                continue

            cleaned_state_dict[
                new_key
            ] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        model.to(
            DEVICE
        )

        model.eval()

        return model

    except Exception as error:

        st.error(
            "Không thể load model CNN."
        )

        st.code(
            str(error)
        )

        return None


MODEL = load_model()


# ============================================================
# 11. IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),

    ]
)


# ============================================================
# 12. PREDICT
# ============================================================

def predict_image(
    image
):

    if MODEL is None:
        return []

    image = image.convert(
        "RGB"
    )

    tensor = IMAGE_TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.inference_mode():

        output = MODEL(
            tensor
        )

        if isinstance(
            output,
            (
                tuple,
                list
            )
        ):

            output = output[0]

        probabilities = F.softmax(
            output,
            dim=1
        )

        k = min(
            TOP_K,
            probabilities.shape[1]
        )

        values, indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )

    results = []

    for probability, index in zip(
        values[0],
        indices[0]
    ):

        index = int(
            index.item()
        )

        probability = float(
            probability.item()
        )

        if (
            0
            <= index
            < len(CLASS_NAMES)
        ):

            species_name = normalize_species_label(
                CLASS_NAMES[index]
            )

        else:

            species_name = "Unknown"

        results.append(
            {
                "species":
                    species_name,

                "confidence":
                    probability,

                "index":
                    index,
            }
        )

    return results


# ============================================================
# 13. WIKIPEDIA
# ============================================================

@st.cache_data(ttl=86400)
def wikipedia_summary(scientific_name):
    """Tra cứu Wikipedia theo tên khoa học; ưu tiên viwiki, rồi enwiki.

    Chỉ chấp nhận bài có tiêu đề/taxon khớp tên khoa học hoặc được Wikidata
    liên kết trực tiếp. Không dùng kết quả tìm kiếm gần đúng để tránh nhầm loài.
    """
    scientific_name = clean_text(scientific_name)
    if not scientific_name:
        return None

    headers = {"User-Agent": "AnimalAI/1.0 (educational project; contact: educational use)"}
    title = scientific_name.replace(" ", "_")

    # Thử bài có tên khoa học chính xác trên Wikipedia tiếng Việt rồi tiếng Anh.
    for language in ("vi", "en"):
        summary_url = (
            f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
            + requests.utils.quote(title, safe="()_")
        )
        try:
            response = requests.get(summary_url, headers=headers, timeout=8)
            if response.status_code != 200:
                continue
            data = response.json()
            extract = clean_text(data.get("extract", ""))
            if not extract or clean_text(data.get("type", "")) == "disambiguation":
                continue

            # Chỉ chấp nhận nếu tiêu đề trả về chính là tên khoa học.
            returned_title = clean_text(data.get("title", ""))
            if returned_title.casefold().replace("_", " ") != scientific_name.casefold():
                continue

            content_urls = data.get("content_urls") or {}
            desktop = content_urls.get("desktop") or {}
            thumbnail = data.get("thumbnail") or {}
            original_image = data.get("originalimage") or {}
            return {
                "title": returned_title,
                "description": clean_text(data.get("description", "")),
                "extract": extract,
                "thumbnail": clean_text(thumbnail.get("source", "")),
                "image": clean_text(original_image.get("source", "")),
                "url": clean_text(desktop.get("page", "")),
                "language": language,
            }
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            continue

    # Fallback: lấy sitelink Wikipedia từ đúng Wikidata item có scientific name (P225).
    try:
        search_response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities", "search": scientific_name,
                "language": "en", "uselang": "en", "type": "item",
                "limit": 10, "format": "json",
            },
            headers=headers, timeout=8,
        )
        search_response.raise_for_status()
        for item in search_response.json().get("search", []):
            item_id = item.get("id", "")
            if not item_id:
                continue
            entity_response = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities", "ids": item_id,
                    "props": "claims|labels|sitelinks", "languages": "en",
                    "sitefilter": "enwiki|viwiki", "format": "json",
                },
                headers=headers, timeout=8,
            )
            entity_response.raise_for_status()
            entity = entity_response.json().get("entities", {}).get(item_id, {})
            claims = entity.get("claims", {})
            exact_name = any(
                claim.get("mainsnak", {}).get("datavalue", {}).get("value") == scientific_name
                for claim in claims.get("P225", [])
            )
            if not exact_name:
                continue

            sitelinks = entity.get("sitelinks", {})
            for site in ("viwiki", "enwiki"):
                page_title = clean_text(sitelinks.get(site, {}).get("title", ""))
                if not page_title:
                    continue
                language = "vi" if site == "viwiki" else "en"
                page_url = f"https://{language}.wikipedia.org/wiki/" + requests.utils.quote(page_title.replace(" ", "_"), safe="()_")
                summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(page_title.replace(" ", "_"), safe="()_")
                page_response = requests.get(summary_url, headers=headers, timeout=8)
                if page_response.status_code != 200:
                    continue
                data = page_response.json()
                extract = clean_text(data.get("extract", ""))
                if not extract or clean_text(data.get("type", "")) == "disambiguation":
                    continue
                content_urls = data.get("content_urls") or {}
                desktop = content_urls.get("desktop") or {}
                thumbnail = data.get("thumbnail") or {}
                original_image = data.get("originalimage") or {}
                return {
                    "title": clean_text(data.get("title", page_title)),
                    "description": clean_text(data.get("description", "")),
                    "extract": extract,
                    "thumbnail": clean_text(thumbnail.get("source", "")),
                    "image": clean_text(original_image.get("source", "")),
                    "url": clean_text(desktop.get("page", page_url)),
                    "language": language,
                    "wikidata_url": f"https://www.wikidata.org/wiki/{item_id}",
                }
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass

    return None


# ============================================================
# 14. GEMINI CONTEXT
# ============================================================

def build_animal_context(
    animal
):

    if not animal:
        return (
            "Chưa có động vật nào "
            "được nhận diện."
        )

    species_name = animal.get(
        "scientific_name",
        "Không rõ"
    )

    common_name = animal.get(
        "common_name",
        "Không rõ"
    )

    group = animal.get(
        "group",
        "Không rõ"
    )

    confidence = animal.get(
        "confidence",
        0
    )

    lines = [

        f"Tên khoa học: {species_name}",

        f"Tên thường gọi: {common_name}",

        f"Nhóm: {group}",

        (
            "Độ tin cậy CNN: "
            f"{confidence:.2f}%"
        ),

    ]

    taxonomy = animal.get(
        "taxonomy"
    )

    if taxonomy:

        lines.append(
            f"Phân loại: {taxonomy}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# 15. GEMINI CHAT
# ============================================================

def ask_gemini(
    question,
    animal=None,
    wiki=None
):

    if GEMINI_CLIENT is None:

        if not GEMINI_SDK_AVAILABLE:

            return (
                "❌ Chưa cài thư viện "
                "`google-genai`.\n\n"
                "Bạn chạy:\n"
                "`python -m pip install -U google-genai`"
            )

        if not GEMINI_API_KEY:

            return (
                "❌ Chưa tìm thấy "
                "`GEMINI_API_KEY`.\n\n"
                "Bạn kiểm tra file `.env`."
            )

        return (
            "❌ Không tạo được Gemini client."
        )

    system_instruction = """
Bạn là Animal AI Assistant.

Bạn chuyên hỗ trợ người dùng tìm hiểu
về các loài động vật.

QUY TẮC:

1. Luôn trả lời bằng tiếng Việt.

2. Giải thích dễ hiểu, tự nhiên,
   phù hợp với sinh viên.

3. Khi người dùng hỏi về con vật hiện tại,
   hãy ưu tiên context được cung cấp từ
   CNN, dataset và Wikipedia.

4. Có thể giải thích về:
   - thức ăn
   - môi trường sống
   - tập tính
   - đặc điểm
   - phân loại
   - sinh sản
   - tuổi thọ
   - kẻ săn mồi
   - tình trạng bảo tồn
   - fun fact

5. Không tự bịa thông tin.

6. Nếu không chắc chắn,
   hãy nói rõ thông tin chưa được xác minh.

7. Kết quả CNN chỉ là dự đoán,
   không được khẳng định CNN chắc chắn đúng.

8. Nếu confidence thấp,
   hãy nhắc người dùng rằng kết quả
   có thể chưa chính xác.

9. Trả lời vừa phải.
   Không cần quá dài nếu câu hỏi đơn giản.

10. Có thể sử dụng emoji vừa phải.
"""

    animal_context = (
        build_animal_context(
            animal
        )
    )

    wiki_context = ""

    if wiki:

        wiki_context = f"""
THÔNG TIN THAM KHẢO TỪ WIKIPEDIA:

Tiêu đề:
{wiki.get("title", "")}

Mô tả:
{wiki.get("description", "")}

Nội dung:
{wiki.get("extract", "")}
"""

    prompt = f"""
THÔNG TIN CON VẬT HIỆN TẠI:

{animal_context}

{wiki_context}

--------------------------------

CÂU HỎI NGƯỜI DÙNG:

{question}
"""

    try:

        # ====================================================
        # INTERACTIONS API
        # ====================================================

        interaction_args = {
            "model":
                GEMINI_MODEL,

            "input":
                prompt,

            "system_instruction":
                system_instruction,

        }

        previous_id = (
            st.session_state
            .get(
                "gemini_interaction_id"
            )
        )

        if previous_id:

            interaction_args[
                "previous_interaction_id"
            ] = previous_id

        interaction = (
            GEMINI_CLIENT
            .interactions
            .create(
                **interaction_args
            )
        )

        # Lưu ID để hội thoại có context
        interaction_id = getattr(
            interaction,
            "id",
            None
        )

        if interaction_id:

            st.session_state[
                "gemini_interaction_id"
            ] = interaction_id

        # ----------------------------------------------------
        # output_text
        # ----------------------------------------------------

        answer = getattr(
            interaction,
            "output_text",
            None
        )

        if answer:

            return str(
                answer
            ).strip()

        # ----------------------------------------------------
        # fallback
        # ----------------------------------------------------

        outputs = getattr(
            interaction,
            "outputs",
            None
        )

        if outputs:

            text_parts = []

            for item in outputs:

                text_value = getattr(
                    item,
                    "text",
                    None
                )

                if text_value:

                    text_parts.append(
                        str(
                            text_value
                        )
                    )

            if text_parts:

                return "\n".join(
                    text_parts
                ).strip()

        return (
            "⚠️ Gemini không trả về "
            "nội dung."
        )

    except Exception as error:

        return (
            "❌ Gemini gặp lỗi.\n\n"
            f"Chi tiết: `{error}`"
        )


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div style="
    font-size:24px;
    font-weight:800;
    color:#302d53;
">
🐾 Animal AI
</div>

<div style="
    color:#73778a;
    font-size:13px;
    line-height:1.6;
    margin-top:5px;
">
Nhận diện động vật bằng CNN
kết hợp trợ lý Gemini AI.
</div>
""",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "📊 Dataset"
    )

    st.write(
        f"🐾 **200 loài**"
    )

    st.write(
        "🦌 80 động vật có vú"
    )

    st.write(
        "🐦 70 loài chim"
    )

    st.write(
        "🐊 50 loài bò sát"
    )

    st.divider()

    st.subheader(
        "🤖 Trạng thái"
    )

    if MODEL is not None:

        st.success(
            "CNN: Sẵn sàng"
        )

    else:

        st.error(
            "CNN: Chưa load được"
        )

    if GEMINI_CLIENT is not None:

        st.success(
            "Gemini: Đã kết nối"
        )

        st.caption(
            f"Model: {GEMINI_MODEL}"
        )

    elif not GEMINI_SDK_AVAILABLE:

        st.warning(
            "Gemini: Chưa cài SDK"
        )

    elif not GEMINI_API_KEY:

        st.warning(
            "Gemini: Chưa có API key"
        )

    else:

        st.warning(
            "Gemini: Chưa kết nối"
        )

    st.divider()

    st.caption(
        "EfficientNet-B0"
    )

    st.caption(
        "CNN + Wikipedia + Gemini"
    )


# ============================================================
# 17. HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
🐾 Animal AI
</div>

<div class="subtitle">
Nhận diện 200 loài động vật bằng AI
và tìm hiểu thêm với Gemini.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 18. HERO
# ============================================================

st.markdown(
    """
<div class="hero-card">

<div class="hero-title">
🔍 Bạn muốn tìm hiểu con vật nào?
</div>

<div class="hero-text">
Tải một hình ảnh động vật lên.
Animal AI sẽ sử dụng mô hình CNN để
dự đoán loài, sau đó bạn có thể hỏi
Gemini về thức ăn, môi trường sống,
tập tính và những điều thú vị.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 19. TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📷 Nhận diện động vật",
        "💬 Hỏi đáp AI",
    ]
)


# ============================================================
# 20. TAB 1 - NHẬN DIỆN
# ============================================================

with tab1:

    st.subheader(
        "📷 Tải ảnh động vật"
    )

    uploaded_file = st.file_uploader(
        "Chọn một hình ảnh",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help=(
            "Nên sử dụng ảnh có "
            "con vật tương đối rõ."
        )
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

        except Exception as error:

            st.error(
                "❌ Không thể mở ảnh."
            )

            st.code(
                str(error)
            )

            image = None

        if image is not None:

            # ------------------------------------------------
            # Reset chat nếu upload ảnh mới
            # ------------------------------------------------

            current_file_name = (
                uploaded_file.name
            )

            if (
                st.session_state
                .last_uploaded_file
                != current_file_name
            ):

                st.session_state[
                    "last_uploaded_file"
                ] = current_file_name

                st.session_state[
                    "current_animal"
                ] = None

                st.session_state[
                    "gemini_interaction_id"
                ] = None

            # ------------------------------------------------
            # IMAGE + RESULT
            # ------------------------------------------------

            col_image, col_result = (
                st.columns(
                    [1, 1],
                    gap="large"
                )
            )

            with col_image:

                st.image(
                    image,
                    caption=(
                        "Ảnh đã tải lên"
                    ),
                    use_container_width=True
                )

            with col_result:

                if MODEL is None:

                    st.error(
                        "❌ Không tìm thấy model CNN."
                    )

                    st.code(
                        str(MODEL_PATH)
                    )

                    predictions = []

                else:

                    with st.spinner(
                        "🤖 CNN đang nhận diện..."
                    ):

                        predictions = (
                            predict_image(
                                image
                            )
                        )

                if predictions:

                    best = predictions[0]

                    species_name = (
                        best[
                            "species"
                        ]
                    )

                    confidence = (
                        best[
                            "confidence"
                        ]
                    )

                    row = get_species_row(
                        species_name
                    )

                    common_name = (
                        get_common_name(
                            row,
                            species_name
                        )
                    )

                    group = get_group(
                        row,
                        species_name
                    )

                    st.markdown(
                        f"""
<div class="animal-card">

<div class="prediction-label">
KẾT QUẢ DỰ ĐOÁN
</div>

<div class="prediction-main">
🐾 {common_name}
</div>

<div class="scientific-name">
{species_name}
</div>

<div class="confidence">
🎯 Độ tin cậy:
{confidence * 100:.2f}%
</div>

<div style="
    margin-top:10px;
    color:#666b7c;
">
<b>Nhóm:</b> {group}
</div>

</div>
""",
                        unsafe_allow_html=True
                    )

                    if confidence < 0.5:

                        st.warning(
                            "⚠️ Độ tin cậy chưa cao. "
                            "Đây chỉ là dự đoán của CNN."
                        )

                    elif confidence >= 0.8:

                        st.success(
                            "✅ CNN đang khá tự tin "
                            "với kết quả này."
                        )

                else:

                    st.info(
                        "Chưa có kết quả nhận diện."
                    )

            # =================================================
            # TOP 5
            # =================================================

            if predictions:

                st.markdown(
                    "### 🏆 Top 5 dự đoán"
                )

                top_rows = ""

                for rank, prediction in enumerate(
                    predictions,
                    start=1
                ):

                    name = (
                        prediction[
                            "species"
                        ]
                    )

                    probability = (
                        prediction[
                            "confidence"
                        ]
                    )

                    row = get_species_row(
                        name
                    )

                    display_name = (
                        get_common_name(
                            row,
                            name
                        )
                    )

                    top_rows += f"""
<div class="top-row">

<div class="top-rank">
#{rank}
</div>

<div class="top-name">
<b>{display_name}</b>
<br>
<i>{name}</i>
</div>

<div class="top-score">
{probability * 100:.2f}%
</div>

</div>
"""

                st.markdown(
                    f"""
<div class="top-card">
{top_rows}
</div>
""",
                    unsafe_allow_html=True
                )

                # =================================================
                # THÔNG TIN LOÀI
                # =================================================

                st.markdown(
                    "### 📚 Thông tin loài"
                )

                row = get_species_row(
                    species_name
                )

                common_name = (
                    get_common_name(
                        row,
                        species_name
                    )
                )

                group = get_group(
                    row,
                    species_name
                )

                info_col1, info_col2 = (
                    st.columns(
                        2,
                        gap="large"
                    )
                )

                # -------------------------------------------------
                # TAXONOMY
                # -------------------------------------------------

                with info_col1:

                    taxonomy = get_taxonomy(row)

                    # Nếu CSV thiếu cấp phân loại, bổ sung từ Wikidata
                    # bằng tên khoa học khớp chính xác.
                    missing_taxonomy = any(
                        not clean_text(taxonomy.get(key, ""))
                        for key in ("kingdom", "phylum", "class", "order", "family", "genus")
                    )
                    if missing_taxonomy:
                        online_taxonomy = wikidata_taxonomy(species_name)
                        for key, value in online_taxonomy.items():
                            taxonomy.setdefault(key, value)

                    taxonomy_labels = {
                        "kingdom":
                            "Giới",

                        "phylum":
                            "Ngành",

                        "class":
                            "Lớp",

                        "order":
                            "Bộ",

                        "family":
                            "Họ",

                        "genus":
                            "Chi",
                    }

                    # Hiển thị đủ các cấp phân loại theo thứ tự cố định.
                    # Nếu CSV không có dữ liệu của cấp nào, vẫn hiện nhãn
                    # để bảng không bị thiếu dòng.
                    taxonomy_order = [
                        "kingdom",
                        "phylum",
                        "class",
                        "order",
                        "family",
                        "genus",
                    ]

                    taxonomy_html = ""

                    for key in taxonomy_order:
                        label = taxonomy_labels[key]
                        value = clean_text(taxonomy.get(key, ""))
                        if not value:
                            value = "Chưa có dữ liệu"

                        taxonomy_html += (
                            f'<div class="taxonomy-row">'
                            f'<strong>{html.escape(label)}:</strong> {html.escape(value)}'
                            f'</div>'
                        )

                    st.markdown(
                        f"""
<div class="info-card">
    <div class="info-title">🧬 Phân loại</div>
    <div class="info-text">
        <div class="taxonomy-row">
            <strong>Tên thường gọi:</strong> {html.escape(common_name)}
        </div>
        <div class="taxonomy-row">
            <strong>Tên khoa học:</strong> <i>{html.escape(species_name)}</i>
        </div>
        <div class="taxonomy-row">
            <strong>Nhóm:</strong> {html.escape(group)}
        </div>
        {taxonomy_html}
    </div>
</div>
""",
                        unsafe_allow_html=True
                    )
                    taxonomy_source_url = (
                        "https://www.wikidata.org/w/index.php?search="
                        + requests.utils.quote(species_name)
                    )
                    st.link_button(
                        "🔎 Kiểm tra nguồn phân loại (Wikidata)",
                        taxonomy_source_url,
                        use_container_width=False,
                    )
                    st.caption(
                        "Thông tin phân loại đang mô tả loài CNN dự đoán; "
                        "nếu ảnh bị nhận diện sai, hãy đối chiếu nguồn này với ảnh và Top 5."
                    )

                # -------------------------------------------------
                # WIKIPEDIA
                # -------------------------------------------------

                with info_col2:

                    with st.spinner(
                        "🌐 Đang tìm Wikipedia tiếng Việt..."
                    ):
                        wiki = wikipedia_summary(species_name)

                    if wiki:
                        wiki_title = clean_text(
                            wiki.get("title", species_name)
                        )
                        wiki_description = clean_text(
                            wiki.get("description", "")
                        )
                        wiki_extract = clean_text(
                            wiki.get("extract", "")
                        )
                        wiki_image = (
                            wiki.get("thumbnail", "")
                            or wiki.get("image", "")
                        )
                        wiki_url = clean_text(
                            wiki.get("url", "")
                        )

                        # Giới thiệu dài hơn để thẻ Wikipedia cân đối với bảng phân loại.
                        if len(wiki_extract) > 650:
                            wiki_extract = (
                                wiki_extract[:650].rsplit(" ", 1)[0]
                                + "..."
                            )

                        # Escape nội dung để tránh HTML từ bài viết làm vỡ thẻ.
                        title_html = html.escape(wiki_title)
                        summary_html = html.escape(wiki_description)
                        extract_html = html.escape(wiki_extract)
                        image_html = html.escape(wiki_image, quote=True)

                        wiki_text_html = ""
                        if summary_html:
                            wiki_text_html += (
                                '<div class="wiki-summary">'
                                + summary_html
                                + '</div>'
                            )
                        if extract_html:
                            wiki_text_html += (
                                '<div class="wiki-description">'
                                + extract_html
                                + '</div>'
                            )

                        if image_html:
                            wiki_body = (
                                '<div class="wiki-layout">'
                                f'<img class="wiki-image" src="{image_html}" '
                                f'alt="Ảnh minh họa {title_html}" loading="lazy">'
                                '<div class="wiki-content">'
                                f'<div class="wiki-name">{title_html}</div>'
                                f'{wiki_text_html}'
                                '</div></div>'
                            )
                        else:
                            wiki_body = (
                                '<div class="wiki-layout wiki-no-image">'
                                '<div class="wiki-content">'
                                f'<div class="wiki-name">{title_html}</div>'
                                f'{wiki_text_html}'
                                '</div></div>'
                            )

                        # HTML viết liền để Streamlit không biến thẻ đóng thành code.
                        wiki_card_html = (
                            '<div class="wiki-card">'
                            '<div class="wiki-title">🌐 Wikipedia</div>'
                            f'{wiki_body}'
                            '</div>'
                        )
                        st.markdown(
                            wiki_card_html,
                            unsafe_allow_html=True
                        )

                        if wiki_url:
                            st.link_button(
                                "🔗 Xem trang Wikipedia",
                                wiki_url,
                                use_container_width=False
                            )
                        st.caption(
                            "Nguồn Wikipedia thuộc loài mà CNN dự đoán. "
                            "Nếu kết quả CNN sai, thông tin và ảnh nguồn cũng sẽ không khớp với con vật trong ảnh."
                        )
                    else:
                        st.info(
                            "Chưa lấy được bài Wikipedia khớp chính xác. "
                            "Bảng dưới đây vẫn cung cấp đường dẫn tìm kiếm để bạn kiểm tra kết quả CNN."
                        )
                        wiki_search_url = (
                            "https://vi.wikipedia.org/w/index.php?search="
                            + requests.utils.quote(species_name)
                        )
                        en_wiki_search_url = (
                            "https://en.wikipedia.org/w/index.php?search="
                            + requests.utils.quote(species_name)
                        )
                        st.link_button(
                            "🔎 Tìm tên này trên Wikipedia tiếng Việt",
                            wiki_search_url,
                            use_container_width=False,
                        )
                        st.link_button(
                            "🌐 Tìm tên này trên Wikipedia tiếng Anh",
                            en_wiki_search_url,
                            use_container_width=False,
                        )
                        st.caption(
                            "Không có bài khớp chính xác không đồng nghĩa loài không tồn tại. "
                            "Hãy kiểm tra tên khoa học và các dự đoán Top 5."
                        )

                # =================================================
                # LƯU CURRENT ANIMAL
                # =================================================

                taxonomy_text = ""

                if taxonomy:

                    taxonomy_text = (
                        "; ".join(
                            [
                                f"{k}: {v}"
                                for k, v
                                in taxonomy.items()
                            ]
                        )
                    )

                st.session_state[
                    "current_animal"
                ] = {

                    "scientific_name":
                        species_name,

                    "common_name":
                        common_name,

                    "group":
                        group,

                    "confidence":
                        confidence * 100,

                    "taxonomy":
                        taxonomy_text,
                }

                st.success(
                    "✅ Đã lưu kết quả nhận diện. "
                    "Bạn có thể sang tab "
                    "**💬 Hỏi đáp AI** để hỏi Gemini."
                )


# ============================================================
# 21. TAB 2 - GEMINI CHATBOT
# ============================================================

with tab2:

    st.subheader(
        "💬 Hỏi đáp với Animal AI"
    )

    current_animal = (
        st.session_state
        .get(
            "current_animal"
        )
    )

    # --------------------------------------------------------
    # CURRENT ANIMAL
    # --------------------------------------------------------

    if current_animal:

        st.markdown(
            f"""
<div class="chat-info">

🐾 <b>Động vật hiện tại:</b>

{current_animal["common_name"]}

<br>

<i>
{current_animal["scientific_name"]}
</i>

<br>

<b>Nhóm:</b>
{current_animal["group"]}

<br>

<b>Độ tin cậy CNN:</b>
{current_animal["confidence"]:.2f}%

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.info(
            "📷 Hãy tải ảnh và nhận diện "
            "một con vật trước."
        )

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.markdown(
        "### 💡 Câu hỏi gợi ý"
    )

    suggestions = [

        "Con vật này ăn gì?",

        "Nó sống ở đâu?",

        "Đặc điểm nổi bật của nó là gì?",

        "Tập tính của nó như thế nào?",

        "Cho mình 5 fun fact thú vị.",

        "Con vật này có nguy hiểm không?",

    ]

    suggestion_cols = st.columns(
        2
    )

    for index, suggestion in enumerate(
        suggestions
    ):

        with suggestion_cols[
            index % 2
        ]:

            if st.button(
                suggestion,
                key=(
                    f"suggestion_{index}"
                ),
                use_container_width=True
            ):

                st.session_state[
                    "pending_question"
                ] = suggestion

                st.rerun()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in (
        st.session_state
        .chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    typed_prompt = st.chat_input(
        "Hỏi Animal AI..."
    )

    pending_question = (
        st.session_state
        .pop(
            "pending_question",
            None
        )
    )

    prompt = (
        pending_question
        or typed_prompt
    )

    if prompt:

        # ----------------------------------------------------
        # Lấy Wikipedia context
        # ----------------------------------------------------

        wiki_context = None

        if current_animal:

            wiki_context = (
                wikipedia_summary(
                    current_animal[
                        "scientific_name"
                    ]
                )
            )

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state[
            "chat_history"
        ].append(
            {
                "role":
                    "user",

                "content":
                    prompt,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Animal AI đang suy nghĩ..."
            ):

                answer = ask_gemini(
                    question=prompt,

                    animal=current_animal,

                    wiki=wiki_context,
                )

            st.markdown(
                answer
            )

        st.session_state[
            "chat_history"
        ].append(
            {
                "role":
                    "assistant",

                "content":
                    answer,
            }
        )


# ============================================================
# 22. FOOTER
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    """
<div style="
    text-align:center;
    color:#8a8da0;
    font-size:13px;
    padding:15px 0;
">

🐾 <b>Animal AI</b>

<br>

CNN EfficientNet-B0
•
200 species
•
Wikipedia
•
Gemini AI

</div>
""",
    unsafe_allow_html=True
)
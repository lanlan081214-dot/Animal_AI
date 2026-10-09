
from pathlib import Path
import pandas as pd
import re

ROOT = Path(r"D:\MindX\CSA\Animal_AI")
SPECIES_CSV = ROOT / "dataset" / "species_list.csv"
OUT_CSV = ROOT / "dataset" / "selected_200_species_v5.csv"
REPORT_CSV = ROOT / "reports" / "selected_200_species_v5_report.csv"
MISSING_TXT = ROOT / "reports" / "selected_200_species_v5_missing.txt"

TARGETS = {"Mammals": 100, "Reptiles": 70, "Birds": 30}

# Uu tien cac loai ma nguoi dung de tim anh tren Internet.
# Script se chi chon loai THUC SU CO trong species_list.csv.
PREFERRED = {
    "Mammals": [
        "Ailuropoda melanoleuca","Panthera leo","Panthera tigris","Panthera pardus","Panthera onca",
        "Acinonyx jubatus","Puma concolor","Canis lupus","Vulpes vulpes","Ursus arctos",
        "Ursus americanus","Ursus maritimus","Elephas maximus","Loxodonta africana","Giraffa camelopardalis",
        "Equus quagga","Equus zebra","Rhinoceros unicornis","Ceratotherium simum","Diceros bicornis",
        "Hippopotamus amphibius","Sus scrofa","Cervus elaphus","Odocoileus virginianus","Capreolus capreolus",
        "Alces alces","Rangifer tarandus","Antidorcas marsupialis","Gazella dorcas","Oryx gazella",
        "Kobus ellipsiprymnus","Damaliscus pygargus","Lama guanicoe","Vicugna vicugna","Camelus dromedarius",
        "Macropus giganteus","Phascolarctos cinereus","Vombatus ursinus","Ornithorhynchus anatinus","Tachyglossus aculeatus",
        "Didelphis virginiana","Dasypus novemcinctus","Myrmecophaga tridactyla","Tamandua mexicana","Choloepus didactylus",
        "Bradypus variegatus","Lontra canadensis","Lutra lutra","Enhydra lutris",
        "Castor canadensis","Ondatra zibethicus","Sciurus vulgaris","Sciurus carolinensis","Marmota monax",
        "Glaucomys volans","Rattus rattus","Mus musculus","Peromyscus maniculatus","Hystrix cristata",
        "Ateles geoffroyi","Alouatta palliata","Cebus capucinus","Papio anubis","Macaca mulatta",
        "Mandrillus sphinx","Lemur catta","Galago senegalensis","Tarsius syrichta","Pongo abelii",
        "Gorilla gorilla","Pan troglodytes","Hylobates lar","Cacajao calvus","Nasua nasua",
        "Procyon lotor","Mephitis mephitis","Meles meles","Mustela erminea","Mustela putorius",
        "Felis catus","Lynx rufus","Leopardus pardalis","Arctocephalus pusillus","Phoca vitulina",
        "Halichoerus grypus","Tursiops truncatus","Orcinus orca","Megaptera novaeangliae","Eschrichtius robustus",
        "Physeter macrocephalus","Balaenoptera musculus","Balaenoptera acutorostrata","Pteropus vampyrus","Pipistrellus pipistrellus",
        "Saimiri sciureus","Cerdocyon thous","Aonyx cinereus","Helarctos malayanus","Melursus ursinus",
    ],
    "Reptiles": [
        "Alligator mississippiensis","Crocodylus niloticus","Crocodylus porosus","Crocodylus acutus",
        "Dracaena guianensis","Varanus komodoensis","Varanus salvator","Varanus niloticus","Varanus exanthematicus",
        "Iguana iguana","Amblyrhynchus cristatus","Conolophus subcristatus","Chamaeleo calyptratus",
        "Furcifer pardalis","Pogona vitticeps","Tiliqua rugosa","Intellagama lesueurii","Agama agama",
        "Anolis carolinensis","Sceloporus occidentalis","Crotaphytus collaris","Podarcis muralis",
        "Lacerta agilis","Phelsuma laticauda","Hemidactylus turcicus","Eublepharis macularius",
        "Testudo hermanni","Testudo graeca","Gopherus agassizii","Terrapene carolina",
        "Chelydra serpentina","Apalone ferox","Chelonia mydas","Caretta caretta","Eretmochelys imbricata",
        "Dermochelys coriacea","Trachemys scripta","Chrysemys picta","Actinemys marmorata","Clemmys guttata",
        "Python regius","Python bivittatus","Morelia spilota","Boa constrictor","Eunectes murinus",
        "Pantherophis guttatus","Pantherophis obsoletus","Lampropeltis getula","Lampropeltis triangulum",
        "Thamnophis elegans","Thamnophis sirtalis","Natrix natrix","Nerodia erythrogaster",
        "Heterodon platirhinos","Coronella austriaca","Diadophis punctatus","Indotyphlops braminus",
        "Crotalus adamanteus","Sistrurus miliarius","Crotalus atrox","Vipera berus","Vipera aspis",
        "Bitis gabonica","Naja naja","Ophiophagus hannah","Micrurus fulvius","Dendroaspis polylepis",
        "Heloderma suspectum","Pituophis catenifer",
    ],
    "Birds": [
        "Haliaeetus leucocephalus","Aquila chrysaetos","Buteo buteo","Accipiter cooperii",
        "Tyto alba","Bubo virginianus","Strix varia","Falco peregrinus",
        "Struthio camelus","Spheniscus demersus","Aptenodytes forsteri",
        "Phoenicopterus roseus","Pavo cristatus","Psittacus erithacus","Ara ararauna",
        "Ramphastos toco","Dacelo novaeguineae","Megaceryle alcyon","Alcedo atthis",
        "Ardea herodias","Ardea alba","Nycticorax nycticorax","Pelecanus occidentalis",
        "Cygnus olor","Anas platyrhynchos","Branta canadensis","Meleagris gallopavo",
        "Gallus gallus","Corvus brachyrhynchos","Cacatua galerita",
    ],
}

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def genus_of(name):
    return str(name).strip().split()[0] if str(name).strip() else ""

def normalize_group(x):
    s = norm(x)
    if s in {"mammal", "mammalia", "mammals"}:
        return "Mammals"
    if s in {"reptile", "reptilia", "reptiles"}:
        return "Reptiles"
    if s in {"bird", "aves", "birds"}:
        return "Birds"
    return None

print("=" * 72)
print("ANIMAL AI - BUILD DATASET 200 V5")
print("Uu tien loai de tim anh tren Internet + giam trung genus")
print("=" * 72)

if not SPECIES_CSV.exists():
    raise FileNotFoundError(f"Khong tim thay: {SPECIES_CSV}")

df = pd.read_csv(SPECIES_CSV)

required = {"scientific_name", "group"}
missing_cols = required - set(df.columns)
if missing_cols:
    raise ValueError(f"species_list.csv thieu cot: {sorted(missing_cols)}")

df["group_norm"] = df["group"].map(normalize_group)
df["scientific_name_norm"] = df["scientific_name"].map(norm)
df["genus"] = df["scientific_name"].map(genus_of)

if "image_count" in df.columns:
    df["image_count_num"] = pd.to_numeric(df["image_count"], errors="coerce").fillna(0)
else:
    df["image_count_num"] = 0

# Chon theo thu tu uu tien, nhung toi da 2 loai / genus.
selected = []
selected_keys = set()
missing_preferred = []

for group, names in PREFERRED.items():
    available = df[df["group_norm"] == group].copy()
    lookup = {
        row["scientific_name_norm"]: row
        for _, row in available.iterrows()
    }

    genus_count = {}
    chosen = []

    for name in names:
        key = norm(name)
        row = lookup.get(key)

        if row is None:
            missing_preferred.append((group, name, "not_in_species_list.csv"))
            continue

        genus = row["genus"]
        if genus_count.get(genus, 0) >= 2:
            missing_preferred.append((group, name, "skipped_third_species_same_genus"))
            continue

        genus_count[genus] = genus_count.get(genus, 0) + 1
        chosen.append(row)

    # Neu danh sach uu tien chua du so luong, bo sung species co nhieu anh,
    # uu tien genus chua xuat hien.
    need = TARGETS[group] - len(chosen)

    if need > 0:
        remaining = available[
            ~available["scientific_name_norm"].isin([norm(r["scientific_name"]) for r in chosen])
        ].copy()

        remaining = remaining.sort_values(
            ["image_count_num", "scientific_name_norm"],
            ascending=[False, True]
        )

        # Vong 1: genus moi
        for _, row in remaining.iterrows():
            if need <= 0:
                break
            genus = row["genus"]
            if genus in genus_count:
                continue
            chosen.append(row)
            genus_count[genus] = 1
            need -= 1

        # Vong 2: cho phep them species trong genus da co neu van thieu
        if need > 0:
            for _, row in remaining.iterrows():
                if need <= 0:
                    break
                key = norm(row["scientific_name"])
                if key in {norm(r["scientific_name"]) for r in chosen}:
                    continue
                genus = row["genus"]
                if genus_count.get(genus, 0) >= 2:
                    continue
                chosen.append(row)
                genus_count[genus] = genus_count.get(genus, 0) + 1
                need -= 1

    if need > 0:
        raise RuntimeError(
            f"Khong du species cho {group}: can {TARGETS[group]}, chi chon duoc {len(chosen)}"
        )

    selected.extend(chosen[:TARGETS[group]])

# Giu thu tu: Mammals -> Reptiles -> Birds
selected_df = pd.DataFrame(selected).copy()
selected_df.insert(0, "class_index", range(1, len(selected_df) + 1))

# Tao ten file/thu muc de prepare script dung de dang hon
if "common_name" not in selected_df.columns:
    selected_df["common_name"] = ""

selected_df["selection_reason"] = "preferred_or_fallback"
selected_df["internet_friendly_priority"] = selected_df["scientific_name_norm"].apply(
    lambda x: next(
        (i + 1 for i, n in enumerate(
            PREFERRED[
                selected_df.loc[
                    selected_df["scientific_name_norm"] == x, "group_norm"
                ].iloc[0]
            ]
        ) if norm(n) == x),
        9999
    )
)

cols = [
    "class_index",
    "scientific_name",
    "common_name",
    "group_norm",
    "genus",
    "image_count_num",
    "selection_reason",
]
cols = [c for c in cols if c in selected_df.columns]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

selected_df[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

report_cols = [
    "class_index","scientific_name","common_name","group_norm","genus",
    "image_count_num"
]
selected_df[report_cols].to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

with open(MISSING_TXT, "w", encoding="utf-8") as f:
    f.write("CAC LOAI UU TIEN KHONG DUNG DUOC / BI BO QUA\n")
    f.write("=" * 70 + "\n")
    for group, name, reason in missing_preferred:
        f.write(f"{group}\t{name}\t{reason}\n")

print(f"\nDa tao: {OUT_CSV}")
print(f"Report: {REPORT_CSV}")
print(f"Missing: {MISSING_TXT}")

print("\nSO LUONG CUOI:")
print(selected_df["group_norm"].value_counts())

print("\nDANH SACH 200 LOAI:")
for _, r in selected_df.iterrows():
    print(
        f'{int(r["class_index"]):03d} | '
        f'{r["group_norm"]:8s} | '
        f'{r["scientific_name"]} | '
        f'images={int(r["image_count_num"])}'
    )

print("\nLuu y:")
print("- Panda / Bald Eagle chi duoc chon neu species_list.csv thuc su co.")
print("- Script KHONG them species moi vao iNat-Anim.")
print("- Neu loai uu tien khong co, script tu dong bu bang loai cung group co nhieu anh.")
print("- Chua xoa data_200 hoac data_200_v2 trong buoc nay.")

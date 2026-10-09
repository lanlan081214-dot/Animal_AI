import json
import random
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import h5py
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "dataset" / "selected_200_species_v6_120mammals_50reptiles_30birds.csv"
JSON_PATH = ROOT / "dataset" / "inat_anim.json"
H5_PATH = ROOT / "dataset" / "low-res-images.hdf5"
OUT_DIR = ROOT / "data_200_v6"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

SEED = 42
MAX_IMAGES_PER_CLASS = 120
EXPECTED_GROUPS = {"Mammals": 120, "Reptiles": 50, "Birds": 30}

def norm_group(x):
    return {"Mammals":"Mammals","Mammalia":"Mammals",
            "Reptiles":"Reptiles","Reptilia":"Reptiles",
            "Birds":"Birds","Aves":"Birds"}.get(str(x).strip(), str(x).strip())

def norm_name(x):
    return str(x).strip().replace("_", " ")

def folder_name(i, group, species):
    return f"{i:03d}_{group}_{species.replace(' ','_').replace('/','-')}"

def quality(arr):
    a = np.asarray(arr)
    g = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    mean, std = float(g.mean()), float(g.std())
    dx = float(np.abs(g[:,1:] - g[:,:-1]).mean())
    dy = float(np.abs(g[1:,:] - g[:-1,:]).mean())
    detail = (dx + dy) / 2
    bright = max(0.0, 1.0 - abs(mean-128.0)/128.0)
    return 40*bright + 35*min(std/70,1) + 25*min(detail/35,1)

def sha(arr):
    a = np.asarray(arr).astype(np.uint8, copy=False)
    return hashlib.sha1(a.tobytes()).hexdigest()

def main():
    print("="*78)
    print("ANIMAL AI - PREPARE DATASET V6 FIXED")
    print("120 MAMMALS + 50 REPTILES + 30 BIRDS")
    print("QUALITY SELECTION + GLOBAL DUPLICATE FILTER + 70/15/15 SPLIT")
    print("="*78)

    random.seed(SEED); np.random.seed(SEED)

    print("\n[1/7] Doc selected_200_species_v6 CSV...")
    if not CSV_PATH.exists(): raise FileNotFoundError(f"Khong tim thay: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    sc = next((c for c in ["scientific_name","scientificName","species","name"] if c in df.columns), None)
    gc = next((c for c in ["group_norm","group","Group"] if c in df.columns), None)
    if not sc or not gc: raise ValueError("CSV thieu cot scientific_name/species va group_norm/group.")
    df = df[[sc,gc]].copy()
    df["scientific_name"] = df[sc].map(norm_name)
    df["group_norm"] = df[gc].map(norm_group)
    df = df[["scientific_name","group_norm"]].drop_duplicates().reset_index(drop=True)
    print("So luong trong CSV:")
    print(df["group_norm"].value_counts())
    if len(df) != 200: raise ValueError(f"CSV phai co 200 species, hien co {len(df)}.")
    for g,n in EXPECTED_GROUPS.items():
        if int((df.group_norm == g).sum()) != n: raise ValueError(f"{g} khong dung {n} species.")

    species_to_group = dict(zip(df.scientific_name, df.group_norm))
    selected = set(species_to_group)

    print("\n[2/7] Doc inat_anim.json...")
    if not JSON_PATH.exists(): raise FileNotFoundError(f"Khong tim thay: {JSON_PATH}")
    if not H5_PATH.exists(): raise FileNotFoundError(f"Khong tim thay: {H5_PATH}")
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    images, anns, cats = data.get("images",[]), data.get("annotations",[]), data.get("categories",[])
    print(f"Images      : {len(images):,}")
    print(f"Annotations : {len(anns):,}")
    print(f"Categories  : {len(cats):,}")

    cat_name = {}
    for c in cats:
        name = c.get("scientific_name") or c.get("name") or c.get("scientificName")
        if c.get("id") is not None and name: cat_name[c["id"]] = norm_name(name)
    image_cat = {a.get("image_id"): a.get("category_id") for a in anns if a.get("image_id") is not None and a.get("category_id") in cat_name}

    print("\n[3/7] Tao danh sach anh theo 200 species...")
    records=[]
    for im in images:
        iid=im.get("id"); cid=image_cat.get(iid); sp=cat_name.get(cid)
        if sp in selected:
            records.append({"image_id":iid,"species":sp,"group":species_to_group[sp]})
    available=Counter(r["species"] for r in records)
    print(f"Image records: {len(records):,}")
    ok=sum(available[s] >= 40 for s in selected)
    print(f"Species du anh (>= 40): {ok}/200")
    if ok != 200:
        bad=sorted((s,available[s]) for s in selected if available[s]<40)
        raise RuntimeError("Species thieu anh: " + str(bad[:20]))

    print("\n[4/7] Cham diem chat luong + loai duplicate TOAN BO dataset...")
    by_species=defaultdict(list); seen={}; dup=0
    with h5py.File(H5_PATH,"r") as h5:
        ds=h5["images"]
        for i,r in enumerate(records,1):
            try: arr=ds[int(r["image_id"])]
            except Exception: continue
            q=quality(arr); h=sha(arr)
            cand={**r,"quality_score":q,"sha1":h}
            old=seen.get(h)
            if old is None:
                seen[h]=cand; by_species[r["species"]].append(cand)
            else:
                dup += 1
                if q > old["quality_score"]:
                    try: by_species[old["species"]].remove(old)
                    except ValueError: pass
                    seen[h]=cand; by_species[r["species"]].append(cand)
            if i%1000==0 or i==len(records):
                print(f"  Processed {i:,}/{len(records):,} | unique {len(seen):,} | duplicates {dup:,}")

    print(f"\nDuplicate bi loai: {dup:,}")
    selected_records=[]
    selection_rows=[]
    for i,row in df.iterrows():
        sp=row.scientific_name; g=row.group_norm
        cand=by_species[sp]
        cand.sort(key=lambda x:(-x["quality_score"], int(x["image_id"])))
        chosen=cand[:MAX_IMAGES_PER_CLASS]
        if len(chosen)<MAX_IMAGES_PER_CLASS:
            raise RuntimeError(f"{sp} chi con {len(chosen)} anh unique, can 120.")
        selected_records.extend(chosen)
        qs=[x["quality_score"] for x in chosen]
        selection_rows.append({
            "class_index":i,"scientific_name":sp,"group_norm":g,
            "available_before_duplicate":available[sp],
            "unique_after_duplicate":len(cand),"selected":len(chosen),
            "quality_min":round(min(qs),3),"quality_mean":round(float(np.mean(qs)),3),
            "quality_max":round(max(qs),3)})
        if (i+1)%20==0 or i==0: print(f"  {i+1:3d}/200 species")

    print(f"\nSelected unique images: {len(selected_records):,}")

    print("\n[5/7] Chia train / val / test = 70/15/15...")
    rng=random.Random(SEED)
    split_rows=[]
    for sp in df.scientific_name:
        items=[x for x in selected_records if x["species"]==sp]
        rng.shuffle(items)
        split_rows += [{**x,"split":"train"} for x in items[:84]]
        split_rows += [{**x,"split":"val"} for x in items[84:102]]
        split_rows += [{**x,"split":"test"} for x in items[102:]]

    split_df=pd.DataFrame(split_rows)
    if len(split_df)!=24000: raise RuntimeError("Khong du 24,000 records.")
    check=split_df.groupby(["split","species"]).size()
    if not ((check.xs("train",level=0)==84).all() and (check.xs("val",level=0)==18).all() and (check.xs("test",level=0)==18).all()):
        raise RuntimeError("Split khong deu 84/18/18.")

    print("\n[6/7] Xoa data_200_v6 cu va export dataset moi...")
    if OUT_DIR.exists():
        print(f"  Xoa: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    for s in ["train","val","test"]: (OUT_DIR/s).mkdir(parents=True,exist_ok=True)

    class_map=list(df.scientific_name)
    exported=failed=0
    with h5py.File(H5_PATH,"r") as h5:
        ds=h5["images"]
        for ci,row in df.iterrows():
            sp=row.scientific_name; g=row.group_norm
            fn=folder_name(ci,g,sp)
            part=split_df[split_df.species==sp]
            for _,r in part.iterrows():
                try:
                    arr=ds[int(r.image_id)]
                    if arr.dtype!=np.uint8: arr=arr.astype(np.uint8)
                    out=OUT_DIR/r.split/fn/f"{int(r.image_id)}.jpg"
                    out.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(arr).save(out,format="JPEG",quality=95)
                    exported+=1
                    if exported%1000==0: print(f"  Exported {exported:,}/24,000")
                except Exception as e:
                    failed+=1; print(f"  ERROR {r.image_id} {sp}: {e}")
    if failed: raise RuntimeError(f"Co {failed} anh export loi.")

    print("\n[7/7] Luu report + class map...")
    REPORT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    sel_report=REPORT_DIR/"dataset_200_v6_fixed_selection.csv"
    img_report=REPORT_DIR/"dataset_200_v6_fixed_images.csv"
    summary=REPORT_DIR/"dataset_200_v6_fixed_summary.txt"
    cmap=MODEL_DIR/"class_names_200_v6_fixed.json"
    pd.DataFrame(selection_rows).to_csv(sel_report,index=False,encoding="utf-8-sig")
    split_df[["image_id","species","group","split","quality_score","sha1"]].to_csv(img_report,index=False,encoding="utf-8-sig")
    cmap.write_text(json.dumps(class_map,ensure_ascii=False,indent=2),encoding="utf-8")
    s=f"""ANIMAL AI - DATASET V6 FIXED
Classes: 200
Mammals: 120
Reptiles: 50
Birds: 30
Train: 16800
Val: 3600
Test: 3600
Total: 24000
Images/class: 120
Duplicate removed before split: {dup}
Quality selection: enabled
Split: 70/15/15
Seed: {SEED}
"""
    summary.write_text(s,encoding="utf-8")

    print("\n"+"="*78)
    print("HOAN TAT DATASET V6 FIXED")
    print("="*78)
    print("Classes : 200")
    print("Mammals : 120")
    print("Reptiles: 50")
    print("Birds   : 30")
    print("Train   : 16,800")
    print("Val     : 3,600")
    print("Test    : 3,600")
    print("Total   : 24,000")
    print(f"Duplicate removed BEFORE split: {dup:,}")
    print(f"Exported: {exported:,}")
    print(f"Failed  : {failed}")
    print(f"Dataset     : {OUT_DIR}")
    print(f"Selection   : {sel_report}")
    print(f"Image report: {img_report}")
    print(f"Class map   : {cmap}")
    print(f"Summary     : {summary}")
    print("="*78)
    print("DATASET V6 FIXED SAN SANG DE CHAY CHECKER.")
    print("="*78)

if __name__=="__main__":
    main()

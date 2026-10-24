#!/usr/bin/env python3
"""Build and optionally publish the six-lens Dolos HF model bundle.

Dry-run by default. No model weights are modified. Stage via symlinks; publish
uses upload_folder (HF reads files from their original paths). Requires
huggingface_hub, torch, and a completed /workspace/dolos fit.
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT=Path("/workspace/dolos")
MODELS={"dolos":("Poseidon-Research/Dolos-1.5B","63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"),
        "parent":("Qwen/Qwen2.5-1.5B-Instruct","989aa7980e4cf806f80c7fef2b1adb7bc71aa306")}
LENSES={"dolos":{k:f"dolos_{k}_n64.pt" for k in ("j","r","jpp")},
        "parent":{k:f"parent_{k}_n64.pt" for k in ("j","r","jpp")}}
META=["dolos_heldout_all.csv","dolos_heldout_all_pass_at_k.csv",
      "parent_heldout_all.csv","parent_heldout_all_pass_at_k.csv",
      "dolos_jpp_n64_expert_weights.json","parent_jpp_n64_expert_weights.json"]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""):h.update(part)
    return h.hexdigest()
def put(stage,source,destination,manifest):
    if not source.is_file(): raise FileNotFoundError(source)
    target=stage/destination
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():target.unlink()
    target.symlink_to(source.resolve())
    manifest[destination]={"sha256":sha(source),"size":source.stat().st_size}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--repo-id",default="bubskee/dolos-qwen25-workspace-lenses")
    p.add_argument("--publish",action="store_true")
    a=p.parse_args()
    root=a.root
    stage=root/"release-staging"
    stage.mkdir(parents=True,exist_ok=True)
    inventory={}
    for model,files in LENSES.items():
        for kind,name in files.items():
            put(stage,root/"artifacts"/name,f"lenses/{model}/{kind}/lens.pt",inventory)
            stamp=root/"artifacts"/f"{name.removesuffix('.pt')}_stamp.json"
            if stamp.is_file():
                put(stage,stamp,f"metadata/fit_records/{model}_{kind}_stamp.json",inventory)
    # original Dolos weights, exact bytes; hydrated metadata files inherited
    original=root/"models"/"dolos"
    parent=root/"models"/"parent"
    if sha(original/"model.safetensors")!=next(
        (i.get("sha256") for i in json.loads((root/"records/dolos-snapshot.json").read_text())["files"] if i["path"]=="model.safetensors"),None):
        raise RuntimeError("Original Dolos weight file differs from pinned snapshot manifest")
    put(stage,original/"model.safetensors","models/dolos/model.safetensors",inventory)
    put(stage,original/"LICENSE","models/dolos/LICENSE",inventory)
    inherited=[]
    for name in ("config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
                 "special_tokens_map.json","vocab.json","merges.txt","added_tokens.json"):
        source=parent/name
        if source.is_file():
            if (original/name).exists() and sha(original/name)!=sha(source):
                raise RuntimeError(f"Hydrated {name} does not match parent")
            put(stage,source,f"models/dolos/{name}",inventory)
            inherited.append(name)
    if not (stage/"models/dolos/config.json").exists() or not (stage/"models/dolos/tokenizer.json").exists():
        raise RuntimeError("Incomplete hydrated Dolos checkpoint")
    for name in META:
        put(stage,root/"artifacts"/name,f"metadata/evaluations/{name}",inventory)
    for alias in ("dolos","parent"):
        put(stage,root/"records"/f"{alias}-snapshot.json",f"metadata/{alias}-snapshot.json",inventory)
    for name in ("runtime.json","python-freeze.txt","jpp_lens.commit","dolos-lenses.commit"):
        file=root/"records"/name
        if file.is_file():put(stage,file,f"metadata/{name}",inventory)
    for path in (root/"repos/dolos-lenses/notebooks").glob("*.ipynb"):
        put(stage,path,f"notebooks/{path.name}",inventory)
    # Recalibration kit: the original LRP expert maps (not ordinary J experts),
    # the matching routers, and small supporting records.
    for alias in ("dolos", "parent"):
        prefix = "dolos" if alias == "dolos" else "parent"
        experts = root/"artifacts"/f"{prefix}_experts_n64.pt"
        router = root/"artifacts"/("dolos_router_full.pt" if alias == "dolos" else "parent_router_n32.pt")
        put(stage, experts, f"research-artifacts/{alias}/experts.pt", inventory)
        put(stage, router, f"research-artifacts/{alias}/router.pt", inventory)
        for source in (experts, router):
            stamp=source.with_name(source.stem+"_stamp.json")
            if stamp.is_file():
                put(stage, stamp, f"research-artifacts/{alias}/{source.stem}_stamp.json", inventory)
    for dirname in ("calibration", "calibration-both"):
        folder=root/"artifacts"/dirname
        if not folder.is_dir(): raise FileNotFoundError(folder)
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                put(stage,file,f"research-artifacts/provenance/{dirname}/{file.relative_to(folder).as_posix()}",inventory)
    # Only immutable training logs. Never stage the log of this publication run:
    # tee continues writing it while upload and remote verification execute.
    for file in sorted((root/"logs").glob("*.log")):
        if "publish" in file.name.lower():
            continue
        put(stage,file,f"research-artifacts/provenance/logs/{file.name}",inventory)
    # Clean previous staging entries so excluded files cannot be re-uploaded.
    stale=stage/"research-artifacts/provenance/logs/final_publish.log"
    if stale.exists() or stale.is_symlink():
        stale.unlink()
    lens_sha={m:{k:inventory[f"lenses/{m}/{k}/lens.pt"]["sha256"] for k in LENSES[m]} for m in LENSES}
    manifest={"schema_version":"1.0","repo_id":a.repo_id,
              "models":{m:{"source":mid,"revision":rev} for m,(mid,rev) in MODELS.items()},
              "dolos_metadata_inherited_from_parent":inherited,"lens_sha256":lens_sha,
              "files":inventory,
              "limitations":["Not a deception detector","Different model-correctness-filtered held-out populations",
                             "Qualitative notebook is not a formal evaluation"]}
    (stage/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    card=f"""---
license: apache-2.0
library_name: transformers
tags:
- mechanistic-interpretability
- workspace-lens
- dolos
---

# Dolos workspace lenses

**Team Iron Golem** · Apart AI Agent Collusion Sprint · Compute supported by a BlueDot Impact grant.

Six workspace lenses: **J, R, J++** for [Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B) and its [Qwen2.5-1.5B-Instruct parent](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

The bundle includes the original Dolos weights, its Apache-2.0 license, and the parent configuration/tokenizer needed to load them. Parent weights are not duplicated.

| Lens | Dolos recall@10 | Parent recall@10 |
|---|---:|---:|
| Logit | 35.2% | 35.1% |
| J | 36.0% | 37.0% |
| R | 42.2% | 45.4% |
| **J++** | **52.1%** | **51.5%** |

Held-out item-weighted macro scores on known intermediate-variable tasks. J++ used a separate calibration split. Each model has its own correctness-filtered evaluation population; scores are not a paired fine-tuning comparison. **These lenses are not deception detectors.**

## Use

- [Hands-on notebook](notebooks/01_reading_six_lenses.ipynb)
- `lenses/dolos/{{j,r,jpp}}/lens.pt`
- `lenses/parent/{{j,r,jpp}}/lens.pt`
- `models/dolos/` — complete checkpoint; load this subdirectory after `snapshot_download`
- `research-artifacts/` — LRP experts, routers, calibration records and logs for J++ recalibration\n- `manifest.json` — revisions, artifact hashes and provenance

The [J++ implementation](https://github.com/safety-research/jpp_lens) loads the lenses. See the [workspace paper](https://transformer-circuits.pub/2026/workspace/index.html), [R-Lens](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers), and [J++](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr) for methods.

Dolos: Poseidon Research. Parent model: Alibaba/Qwen. Lens methods: original authors. Model/source revisions and hashes: `manifest.json`. [Project code](https://github.com/bubskee/dolos-lenses).

*We protect the villagers.* 🌹
"""
    (stage/"README.md").write_text(card)
    print("STAGED",stage,"files",len(inventory),"lens sha256",json.dumps(lens_sha,indent=2))
    if not a.publish:
        print("DRY RUN ONLY. Review licenses, card, and manifest; pass --publish to upload.")
        return
    from huggingface_hub import HfApi
    api=HfApi()
    print("Authenticated as:",api.whoami()["name"])
    api.create_repo(a.repo_id,repo_type="model",private=False,exist_ok=True)
    api.upload_folder(folder_path=str(stage),repo_id=a.repo_id,repo_type="model",
                      commit_message="Publish six validated workspace lenses and hydrated Dolos")
    from huggingface_hub import hf_hub_download
    for path,info in inventory.items():
        remote=Path(hf_hub_download(repo_id=a.repo_id,filename=path,force_download=True,
                                   local_dir=root/"release-verification"))
        if sha(remote)!=info["sha256"]:raise RuntimeError(f"Remote hash mismatch: {path}")
        print("VERIFIED",path)
    print("ALL REMOTE HASHES VERIFIED",a.repo_id)
if __name__=="__main__":
    main()

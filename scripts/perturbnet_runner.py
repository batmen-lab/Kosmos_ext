#!/usr/bin/env python
"""Train and sample PerturbNet (genetic branch) for one stage-3 arm.

Runs in the separate PerturbNet environment (Python 3.7, scvi-tools 0.7.1,
torch 1.13.1), never in Kosmos' own `.venv`. Stage 3
(`kosmos/ppi/perturbation/perturbnet_arm.py`) writes a bundle, calls this
script with that environment's interpreter, and scores what it writes back.

Follows the PerturbNet genetic tutorial and scPerturbBench `PerturbNet.py`:

  * cellular representation: scVI on raw counts, `n_latent=10`, 500 epochs;
  * perturbation representation: the pretrained GenotypeVAE over GO-annotation
    one-hot vectors (frozen); a control is the all-zero vector;
  * cINN `ConditionalFlatCouplingFlow` (20 flows, hidden 1024) trained 45 epochs,
    batch 128, lr 4.5e-6, on the gold training cells only;
  * prediction: z ~ cINN(c), counts ~ scVI decoder(z, library).

The one difference between the two arms is which cells train scVI: gold
training cells (`perturbnet_base`), or those plus the supplementary cells with
the source as scVI's batch covariate (`perturbnet_supp`). The gold rows come
first in the AnnData: PerturbNet's cINN reads latents by row index of the
AnnData scVI was set up on, so its indices must point at gold rows.

Each generated cell is paired with one real gold training cell: that cell's
inferred library sets the sampling depth, and its panel fraction (panel
counts / whole-transcriptome counts) turns the generated panel counts into
whole-transcriptome CPM -- the gold's own normalisation, log1p(CPM 1e4).

Inputs (`--bundle`): train_counts.npy, train_total.npy, train_labels.json,
[supp_counts.npy, supp_total.npy], predict.json, config.json.
Outputs (`--out`): predictions.npz (labels, means), history.json, report.json.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np

# `perturbnet.util` imports the TensorFlow VAE (`data_vae.vae`) at module level;
# the genetic branch never uses it, so it is stubbed instead of installing TF 1.15.
sys.modules.setdefault("perturbnet.data_vae.vae", types.ModuleType("perturbnet.data_vae.vae"))

import anndata  # noqa: E402
import pandas as pd  # noqa: E402
import scvi  # noqa: E402
import torch  # noqa: E402
from scipy import sparse  # noqa: E402

from perturbnet.cinn.flow import ConditionalFlatCouplingFlow, Net2NetFlow_scVIGenoFlow  # noqa: E402
from perturbnet.cinn.flow_generate import SCVIZ_CheckNet2Net  # noqa: E402
from perturbnet.genotypevae.genotypeVAE import GenotypeVAE  # noqa: E402
from perturbnet.pytorch_scvi.scvi_generate_z import scvi_predictive_z  # noqa: E402

#: Adamson-era HGNC symbols -> the symbols GenotypeVAE's `gene.npy` uses.
SYMBOL_ALIASES = {
    "AARS": "AARS1", "ATP5B": "ATP5F1B", "CARS": "CARS1", "DARS": "DARS1",
    "HARS": "HARS1", "MARS": "MARS1", "QARS": "QARS1", "SARS": "SARS1",
    "SLMO2": "PRELID3B", "SRPR": "SRPRA", "TARS": "TARS1",
}
CONTROL = "ctrl"


def log(message):
    print("[perturbnet] " + message, flush=True)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def go_onehot(genes, gene_index, annotation):
    """GEARS-style multi-gene union of GO one-hot rows; unannotated genes are listed."""
    rows, missing = [], []
    for gene in genes:
        symbol = gene if gene in gene_index else SYMBOL_ALIASES.get(gene, gene)
        if symbol in gene_index:
            rows.append(gene_index[symbol])
        else:
            missing.append(gene)
    vector = np.zeros(annotation.shape[1], dtype=np.float32)
    for row in rows:
        vector = np.logical_or(vector, annotation[row]).astype(np.float32)
    return vector, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    bundle, out = args.bundle, args.out
    os.makedirs(out, exist_ok=True)
    config = json.load(open(os.path.join(bundle, "config.json")))
    seed = int(config["seed"])
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    report = {"arm": config["arm"], "device": device}

    # -- cells ---------------------------------------------------------------
    gold_counts = np.load(os.path.join(bundle, "train_counts.npy"))
    gold_total = np.load(os.path.join(bundle, "train_total.npy"))
    gold_labels = np.array(json.load(open(os.path.join(bundle, "train_labels.json"))), dtype=object)
    n_gold = gold_counts.shape[0]
    blocks, sources = [gold_counts], ["gold"] * n_gold
    use_supp = bool(config.get("use_supplementary"))
    if use_supp:
        supp_counts = np.load(os.path.join(bundle, "supp_counts.npy"))
        blocks.append(supp_counts)
        sources += ["supp"] * supp_counts.shape[0]
    counts = np.concatenate(blocks).astype(np.float32)
    genes = list(config["genes"])
    obs = pd.DataFrame(
        {"condition": list(gold_labels) + [CONTROL] * (counts.shape[0] - n_gold), "source": sources},
        index=[str(i) for i in range(counts.shape[0])],
    )
    adata = anndata.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = adata.X.copy()
    report["cells"] = {"gold_train": int(n_gold), "supplementary": int(counts.shape[0] - n_gold)}
    log("scVI cells: {:,} gold + {:,} supplementary, {:,} genes".format(
        n_gold, counts.shape[0] - n_gold, counts.shape[1]))

    # -- scVI ------------------------------------------------------------------
    scvi.data.setup_anndata(adata, layer="counts", batch_key="source" if use_supp else None)
    scvi_model = scvi.model.SCVI(adata, n_latent=int(config["scvi_latent"]), use_cuda=device == "cuda")
    tick = time.time()
    scvi_model.train(n_epochs=int(config["scvi_epochs"]), frequency=20)
    report["scvi_seconds"] = round(time.time() - tick, 1)
    history = {key: [float(v) for v in values] for key, values in scvi_model.history.items()}
    log("scVI done in {:.0f}s".format(report["scvi_seconds"]))

    # -- GenotypeVAE + perturbation one-hots -----------------------------------
    go_dir = config["genotype_vae_dir"]
    gene_symbols = np.load(os.path.join(go_dir, "gene.npy"))
    gene_index = {str(g): i for i, g in enumerate(gene_symbols)}
    annotation = sparse.load_npz(os.path.join(go_dir, "sparse_gene_anno_matrix.npz")).toarray()
    genovae = GenotypeVAE().to(device)
    genovae.load_state_dict(torch.load(os.path.join(go_dir, "model_params.pt"), map_location=device))
    genovae.eval()

    predict = json.load(open(os.path.join(bundle, "predict.json")))
    perturbation_genes = {CONTROL: []}
    perturbation_genes.update({label: entry["genes"] for label, entry in predict.items()})
    for label in set(gold_labels) - {CONTROL}:
        perturbation_genes.setdefault(label, label.split("+"))
    onehot, unannotated = {}, {}
    for label, label_genes in perturbation_genes.items():
        onehot[label], missing = go_onehot(label_genes, gene_index, annotation)
        if missing:
            unannotated[label] = missing
    report["unannotated_genes"] = unannotated
    if unannotated:
        log("WARNING: no GO annotation for {} (encoded as control)".format(unannotated))

    # -- cINN (gold training cells only) -------------------------------------
    assert (adata.obs["source"].to_numpy()[:n_gold] == "gold").all(), "gold rows must come first"
    trained = sorted(set(gold_labels))
    embeddings = np.stack([onehot[label] for label in trained])
    to_embed = {label: i for i, label in enumerate(trained)}
    torch.manual_seed(seed)
    flow = ConditionalFlatCouplingFlow(
        conditioning_dim=10, embedding_dim=10, conditioning_depth=2, n_flows=20,
        in_channels=int(config["scvi_latent"]), hidden_dim=1024, hidden_depth=2,
        activation="none", conditioner_use_bn=True,
    )
    model_c = Net2NetFlow_scVIGenoFlow(
        configured_flow=flow, cond_stage_data=gold_labels, model_con=genovae,
        scvi_model=scvi_model, perturbToOnehotLib=to_embed, oneHotData=embeddings,
    )
    model_c.to(device=device)
    tick = time.time()
    model_c.train(n_epochs=int(config["cinn_epochs"]), batch_size=128, lr=float(config["cinn_lr"]))
    report["cinn_seconds"] = round(time.time() - tick, 1)
    history["cinn_train_loss"] = [float(v) for v in model_c.train_loss]
    history["cinn_heldout_cell_loss"] = [float(v) for v in model_c.test_loss]
    log("cINN done in {:.0f}s".format(report["cinn_seconds"]))

    # -- sampling ----------------------------------------------------------------
    model_c.eval()
    gold_rows = np.arange(n_gold)
    library = scvi_model.get_latent_library_size(indices=gold_rows, give_mean=False)
    panel_fraction = gold_counts.sum(axis=1) / np.maximum(gold_total, 1.0)
    # PerturbNet decodes on the CPU (numpy batches); the batch code of `gold`
    # is fixed for the supplementary arm (the stock `sample_data` passes the
    # DataLoader counter as the batch index).
    scvi_model.model.cpu()
    scvi_model.use_cuda = False
    sampler = SCVIZ_CheckNet2Net(model_c, device, scvi_predictive_z(scvi_model))
    gold_batch = int(adata.obs["_scvi_batch"].to_numpy()[0]) if use_supp else None
    target_sum = float(config["target_sum"])
    rng = np.random.RandomState(seed + 3)

    labels, means = [], []
    requests = dict(predict)
    requests[CONTROL] = {"genes": [], "n_cells": int(config["control_cells_to_sample"])}
    for label in sorted(requests):
        n_cells = max(int(requests[label]["n_cells"]), 1)
        condition = np.tile(onehot[label], (n_cells, 1))
        with torch.no_grad():
            _, _, _, embedded = model_c.model_con(torch.tensor(condition).float().to(device))
        embedded = embedded.cpu().numpy()
        picks = rng.choice(n_gold, n_cells, replace=True)
        if use_supp:
            batch = np.full((n_cells, 1), gold_batch, dtype=np.int64)
            _, sampled = sampler.sample_data_with_batch(embedded, library[picks], batch)
        else:
            _, sampled = sampler.sample_data(embedded, library[picks])
        sampled = np.asarray(sampled, dtype=np.float64)
        panel_sum = np.maximum(sampled.sum(axis=1, keepdims=True), 1.0)
        cpm = sampled / panel_sum * panel_fraction[picks][:, None] * target_sum
        labels.append(label)
        means.append(np.log1p(cpm).mean(axis=0))
    np.savez(os.path.join(out, "predictions.npz"), labels=np.array(labels), means=np.stack(means).astype(np.float32))
    report["seconds"] = round(time.time() - started, 1)
    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=2)
    log("done in {:.0f}s: {} perturbation(s) + ctrl sampled".format(report["seconds"], len(labels) - 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

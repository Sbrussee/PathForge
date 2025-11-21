# Script to generate test samples for PathBench
from huggingface_hub import hf_hub_download
import lazyslide as ls
import anndata as ad
import pandas as pd


def download_gtex_slides():
    """Download a GTEx slide from the HuggingFace hub."""
    slide_annotation_path = hf_hub_download(
        "rendeirolab/lazyslide-data",
        "GTEx_artery_dataset.csv.gz",
        repo_type="dataset",
    )
    slide_annotation = pd.read_csv(slide_annotation_path)
    return slide_annotation


def download_tcga_read_titan_features() -> ad.AnnData:
    titan_features = hf_hub_download(
        "rendeirolab/lazyslide-data",
        "TCGA_READ_subset_TITAN.h5ad",
        repo_type="dataset",
        local_dir=".",
    )
    adata = ad.read_h5ad(titan_features)
    #Setup survival labels
    adata.obs["status"] = (
        adata.obs["OS_STATUS"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(bool)
    )
    return adata

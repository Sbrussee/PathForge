# Slide-retrieval model weights

Place user-downloaded retrieval assets in this directory. Weight files are
intentionally ignored by Git, so this README is the only tracked file here.

## SISH asset names

The SISH semantic notebook and VQ-VAE descriptor precompute resolve these
filenames by default:

```text
sish_vqvae_checkpoint.pth
sish_codebook.pt
sish_trash_classifier.pkl
```

The VQ-VAE checkpoint and semantic codebook are required only when PathForge
needs to create SISH indices or descriptors. `sish_trash_classifier.pkl` is
reserved for the upstream-compatible trash-filtering step and is not required
until that step is implemented.

Download compatible SISH assets from the
[upstream SISH project](https://github.com/mahmoodlab/SISH) and place them
under the names above. PathForge validates that the VQ-VAE checkpoint can load
into the supported encoder architecture; it deliberately does not pin a
specific model release, so compatible alternative weights are supported.

To use another directory, set `slide_retrieval.weights_dir`; existing explicit
`sish.vqvae_checkpoint` and `sish.codebook_semantic` settings take precedence
over this directory.

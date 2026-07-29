# Slide retrieval

PathForge slide retrieval ranks reference slides for each query slide. It uses
the same feature-extraction artifacts as the rest of the benchmarking workflow:
configure a retrieval representation and a search strategy, then run the
slide-retrieval benchmark or materialize representations in advance.

## SISH

SISH is a patch-based search strategy. Each selected mosaic patch has two
companion representations:

1. A VQ-VAE-derived integer key, used for Van Emde Boas predecessor/successor
   search.
2. A binary code derived from the configured patch feature vector, used for
   Hamming-distance verification of candidate patches.

The strategy works with the existing mosaic-selection representations. It does
not require a separate benchmark mode or a separate tile grid.

### SISH model assets

By default, SISH resolves downloaded assets from the repository-relative
directory `model_weights/slide_retrieval`. The directory is present in the
repository, while its downloaded contents are ignored by Git. Configure a
different shared or local location when needed:

```yaml
slide_retrieval:
  weights_dir: /shared/model_weights/slide_retrieval
```

Place compatible upstream assets under these conventional names:

```text
sish_vqvae_checkpoint.pth
sish_codebook.pt
sish_trash_classifier.pkl
```

The VQ-VAE checkpoint and codebook are required when creating SISH descriptors
or indices. The trash-classifier filename is reserved for the future
upstream-compatible trash-filtering step and is not required today. Obtain the
assets from the [upstream SISH project](https://github.com/mahmoodlab/SISH).
PathForge verifies that the checkpoint loads into its supported VQ-VAE encoder,
but intentionally does not require a fixed checksum, so compatible alternative
weights can be used.

For an exceptional layout, explicit legacy settings remain supported and take
precedence over `weights_dir`:

```yaml
sish:
  vqvae_checkpoint: /other/location/vqvae.pth
  codebook_semantic: /other/location/codebook.pt
```

### Canonical VQ-VAE crop

The SISH VQ-VAE always receives a canonical `1024 × 1024` crop at `0.5 mpp`.
For every source tile selected by the retrieval representation, PathForge uses
that tile's physical centre to read the canonical SISH crop:

```text
configured source tile centre
  └── canonical SISH crop: 1024 px at 0.5 mpp (512 µm × 512 µm)
```

This makes SISH usable with arbitrary feature-extraction tile settings:

- Smaller source tiles give the VQ-VAE additional surrounding context.
- Larger source tiles contribute their central 512 µm field to the VQ-VAE.
- A source tile configured as `1024 px` at `0.5 mpp` is exactly the same image
  supplied to the VQ-VAE.

Source slides must have imagery at `0.5 mpp` or finer. PathForge rejects a
source slide that is coarser than this target instead of silently inventing
resolution by upsampling.

### Feature-vector binary code

For every patch feature row `f`, SISH uses adjacent-value bit encoding:

```text
bit[0] = 0
bit[i] = 1 when f[i] >= f[i - 1], otherwise 0
```

The packed bits are compared with Hamming distance. The `hamming_thr` search
hyperparameter defaults to `128`, matching the original SISH implementation;
set it explicitly when calibrating a particular feature extractor.

### Descriptor cache layout

VQ-VAE latents are cached in the slide-retrieval H5 artifact under the source
tiling identifier, just like mean-RGB descriptors:

```text
bags/{source_tile_id}/descriptors/sish_vqvae_latent
```

Descriptor row `i` is aligned with coordinate row `i` from that source tile
bag. Although the VQ-VAE image is canonical, the source tile ID remains the
correct cache key because it defines the row set and centre coordinates.

Each SISH descriptor dataset records its crop geometry/alignment and the
configured VQ-VAE checkpoint and semantic-codebook identities. A descriptor
whose cache contract does not match is recomputed rather than reused.

### Precompute descriptors

Precompute SISH VQ-VAE descriptors for a slide before a retrieval run:

```bash
pathforge retrieval sish-vqvae \
  --config retrieval.yaml \
  --dataset ReferenceSet \
  --slide-id SLIDE_001
```

Use `--bag-id` to restrict precomputation to one or more configured tile IDs.
The normal slide-retrieval run can also create missing descriptors on demand.

### Future strict SISH setup

The current strategy uses any configured patch feature extractor as the Hamming
companion. To reproduce the upstream SISH pairing more closely, use an
upstream-compatible DenseNet-121 extractor with:

```yaml
benchmark_parameters:
  feature_extraction: [sish-densenet121]
  tile_px: [1024]
  tile_mpp: [0.5]
```

In that configuration, the DenseNet and VQ-VAE operate on the same image. The
`sish-densenet121` extractor is planned; it is not currently registered.

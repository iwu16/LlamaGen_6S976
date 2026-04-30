# Aggregate sample workflow

This workflow lets each collaborator build the same local aggregate dataset without
touching the original `samples/` folder and without committing generated samples to
GitHub.

The aggregate folder is local to each clone:

```text
LlamaGen_6S976/aggregated_samples/
  clean/
  tokens/
  watermarked/
  incoming/
    reugene/
      clean/
      tokens/
      watermarked/
    maureenz/
      clean/
      tokens/
      watermarked/
    isawu888/
      clean/
      tokens/
      watermarked/
  tarballs/
  manifests/
```

`samples/` remains the source folder. `aggregated_samples/` is the combined local
folder and is ignored by Git.

## 1. Each collaborator initializes their own local samples

From the repository root:

```bash
bash aggregate_images/setup_reugene_dataset.sh samples reugene
```

Replace `reugene` with your contributor name:

```bash
bash aggregate_images/setup_reugene_dataset.sh samples maureenz
bash aggregate_images/setup_reugene_dataset.sh samples isawu888
```

This copies your local:

```text
samples/clean
samples/tokens
samples/watermarked
```

into:

```text
aggregated_samples/incoming/<your_name>/
```

## 2. Each collaborator packages their own samples

From the repository root:

```bash
bash aggregate_images/package_pngs.sh samples reugene
```

Replace `reugene` with your contributor name. The script creates:

```text
reugene_samples.tar.gz
```

Send that tarball to the other two collaborators with `scp`, Globus, or OnDemand
upload. Each collaborator should place received tarballs under:

```text
aggregated_samples/tarballs/
```

## 3. Each collaborator imports the other tarballs

Example:

```bash
bash aggregate_images/import_contributor_tarball.sh aggregated_samples/tarballs/maureenz_samples.tar.gz maureenz
bash aggregate_images/import_contributor_tarball.sh aggregated_samples/tarballs/isawu888_samples.tar.gz isawu888
```

Each import writes to:

```text
aggregated_samples/incoming/<contributor>/
```

## 4. Each collaborator builds their local aggregate folder

```bash
bash aggregate_images/build_all_pngs.sh
```

The final combined folders are:

```text
aggregated_samples/clean
aggregated_samples/tokens
aggregated_samples/watermarked
```

Files are prefixed by contributor name, for example:

```text
aggregated_samples/clean/reugene_00666.png
aggregated_samples/tokens/reugene_clean_00666.pt
aggregated_samples/watermarked/reugene_00666.png
```

Verify counts:

```bash
find aggregated_samples/clean -type f | wc -l
find aggregated_samples/tokens -type f | wc -l
find aggregated_samples/watermarked -type f | wc -l
```

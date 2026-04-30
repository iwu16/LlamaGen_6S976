# Aggregate image workflow

This workflow lets three individual ORCD users combine PNGs without PI shared pool.
Each contributor packages their PNGs into one tarball. Reugene imports those tarballs
into `/home/reugene/orcd/pool/redteam_images`, then builds a combined `all_pngs`
folder with contributor-prefixed filenames.

## 1. Reugene: initialize the dataset folder

From this repository:

```bash
bash aggregate_images/setup_reugene_dataset.sh
```

By default this copies clean PNGs from:

```text
/home/reugene/LlamaGen_6S976/samples/clean
```

into:

```text
/home/reugene/orcd/pool/redteam_images/incoming/reugene
```

If you want to use a different PNG folder:

```bash
bash aggregate_images/setup_reugene_dataset.sh /path/to/reugene/pngs
```

## 2. Maureenz and isawu888: package their PNGs

Send them this script:

```text
aggregate_images/package_pngs.sh
```

They run:

```bash
bash package_pngs.sh /path/to/their/pngs maureenz
bash package_pngs.sh /path/to/their/pngs isawu888
```

That creates:

```text
maureenz_pngs.tar.gz
isawu888_pngs.tar.gz
```

They can send those tarballs to you with `scp`, Globus, or OnDemand upload.

Example `scp` from their local computer:

```bash
scp maureenz_pngs.tar.gz reugene@orcd-login.mit.edu:/home/reugene/orcd/pool/redteam_images/tarballs/
scp isawu888_pngs.tar.gz reugene@orcd-login.mit.edu:/home/reugene/orcd/pool/redteam_images/tarballs/
```

## 3. Reugene: import each tarball

After the tarballs are in `/home/reugene/orcd/pool/redteam_images/tarballs`:

```bash
bash aggregate_images/import_contributor_tarball.sh /home/reugene/orcd/pool/redteam_images/tarballs/maureenz_pngs.tar.gz maureenz
bash aggregate_images/import_contributor_tarball.sh /home/reugene/orcd/pool/redteam_images/tarballs/isawu888_pngs.tar.gz isawu888
```

## 4. Reugene: build the combined folder

```bash
bash aggregate_images/build_all_pngs.sh
```

The combined folder will be:

```text
/home/reugene/orcd/pool/redteam_images/all_pngs
```

Verify the final count:

```bash
find /home/reugene/orcd/pool/redteam_images/all_pngs -type f -iname '*.png' | wc -l
```

Expected count is `1000`.

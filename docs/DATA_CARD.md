# Data Card

This project uses tympanic-membrane (eardrum) otoscopy images from three
clinical sites. **No image data is committed to this repository.**

> **Important:** "Publicly available" does **not** mean "redistributable."
> Even where a dataset is downloadable, its license may forbid re-hosting.
> Obtain each dataset from its original source under its own terms. Do not
> commit images, and do not push a mirror of any dataset to this repo.

## Datasets

| Role   | Site                                   | Label in code | Notes                          |
|--------|----------------------------------------|---------------|--------------------------------|
| Source | Hospital in Turkey                     | `eardrumDs`   | Training / source domain       |
| Target | Nationwide Children's Hospital (Ohio)  | `OSU`         | Cross-site target domain       |
| Target | Chilean clinical hospital              | `Chile`       | Cross-site target domain       |

## Expected directory layout

Each dataset root should contain two class subfolders:

```
<dataset_root>/
├── Normal/
│   ├── img001.png
│   └── ...
└── Abnormal/
    ├── img101.png
    └── ...
```

Point `data.train_root` and `data.targets.*` in `config.yaml` at these roots.

## Access

<!-- TODO (Jordan): fill in the source, citation, and access/license terms for
     each dataset — link to the Turkey dataset publication, the OSU/Nationwide
     data-use agreement, and the Chile dataset source. -->

- **Turkey (source):** _source / citation / license TBD_
- **OSU (target):** _source / DUA / license TBD_
- **Chile (target):** _source / citation / license TBD_

## Obtaining the data

A helper script is provided as a placeholder — it does **not** bundle any
images and should be filled in with per-dataset download/verification logic
once access terms are confirmed:

```bash
python scripts/download_data.py --dataset OSU --dest /path/to/TM_Datasets/OSUdataset
```

## Preprocessing

- Images are resized to `image_size` (default **224×224**) at load time.
- ImageNet normalization is applied (standard ResNet50 preprocessing).
- No images are modified on disk; masking (peripheral-masking evaluation)
  is applied in-memory at evaluation time.

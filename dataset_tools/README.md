# Dataset Tools

Scripts for building the fine-tuning dataset for the vehicle detector, matching
the project plan: ~3,000 images total (60% self-collected in Mashhad, 40% from
CNRPark-EXT / PKLot), with a 50% day / 30% dusk-or-shade / 20% night lighting
mix, and a 70% train / 15% val / 15% test split where test streets are
completely separate from train/val streets.

## Naming convention

Every image (and its matching YOLO `.txt` label file) must follow:

```
<street>_<lighting>_<index>.jpg
<street>_<lighting>_<index>.txt
```

Example: `azadi-blvd_day_00042.jpg`. `lighting` must be one of `day`, `dusk`,
`night`. `street` should be a short slug unique per physical street/camera
position — the split script uses it to keep test streets separate from
train/val streets, per the project's acceptance criteria.

## Workflow

1. **Extract frames from recorded video**

   ```
   python extract_frames.py raw_video/azadi.mp4 raw_frames/ \
       --street azadi-blvd --lighting day --interval 2.0
   ```

   Run once per (street, lighting) recording. Repeat for every street and
   lighting condition you've filmed.

2. **Label the extracted frames**

   Use LabelImg or Roboflow, in YOLO format, with a single class `car`.
   Save each image's labels as `<same-stem>.txt` next to (or alongside a
   labels folder matching) the image.

3. **Check the lighting/street distribution before splitting**

   ```
   python dataset_stats.py raw_frames/
   ```

   Compare against the 50/30/20 day/dusk/night target and adjust collection
   if one condition is under-represented.

4. **Split into train/val/test and generate `data.yaml`**

   ```
   python organize_dataset.py raw_frames/ raw_labels/ dataset/ \
       --test-ratio 0.15 --val-ratio 0.15
   ```

   This produces:

   ```
   dataset/
     images/train  images/val  images/test
     labels/train  labels/val  labels/test
     data.yaml
   ```

   ready to pass to `ultralytics` for fine-tuning:

   ```
   yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100
   ```

5. **Fine-tune YOLOv8**

   ```
   python train.py dataset/data.yaml --model yolov8n.pt --epochs 100 --device cpu
   ```

   Weights are written under `runs/detect/street_parking/weights/best.pt`.

6. **Evaluate against the project's acceptance criteria**

   ```
   python evaluate.py runs/detect/street_parking/weights/best.pt dataset/data.yaml --device cpu
   ```

   Checks Precision >= 85%, Recall >= 80%, mAP@0.5 >= 80%, and per-frame
   inference time < 3s on CPU, and prints PASS/FAIL for each.

## Merging in an external dataset (e.g. a Roboflow export)

The [Roadside Parking dataset on Roboflow Universe](https://universe.roboflow.com/3883zn-gmail-com/roadside-parking)
(1,652 street-level images, `car`/`motorcycle` classes, CC BY 4.0) is a much
closer match to this project's scenario than CNRPark-EXT/PKLot, since it's
shot from street level rather than overhead. To use it:

1. On the Roboflow page, export the dataset in **YOLOv8** format and extract
   the zip (it will contain `train/`, `valid/`, `test/` folders each with
   `images/` and `labels/`, plus a `data.yaml`).
2. Convert it into this project's naming convention, keeping only the `car`
   class:

   ```
   python prepare_external_dataset.py path/to/extracted_export raw_frames/ raw_labels/ \
       --source-name roadsideparking --lighting day --keep-class car
   ```

   This drops `motorcycle` annotations, remaps `car` to class id 0, and
   spreads the images across 20 pseudo-street tags (`roadsideparking1`,
   `roadsideparking2`, ...) so `organize_dataset.py`'s per-street split
   doesn't lump the entire external dataset into a single train-or-test
   bucket.

3. Continue with `dataset_stats.py` and `organize_dataset.py` as usual — the
   converted images merge in alongside your own Mashhad footage.

**Caveat:** the project's acceptance criteria call for test images from
streets that are genuinely separate from training — that guarantee only
matters for your own Mashhad street recordings. This external dataset is
supplementary training/validation data, not a substitute for the real
field test in the project's final testing phase.

## Merging in the public datasets (CNRPark-EXT / PKLot)

These datasets are overhead parking-lot views, not street-side, so they help
the model learn general "car" appearance but not the gap-detection geometry.
Convert their annotations to YOLO format and prefix filenames with a street
tag such as `cnrpark_day_00001.jpg` / `pklot_day_00001.jpg` so
`organize_dataset.py` treats each source as its own "street" and keeps them
out of the test split, per the project's 60/40 self-collected/public ratio.

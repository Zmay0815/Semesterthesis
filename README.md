# Semester Thesis ECG Scripts

This folder contains the final scripts used to generate the datasets, figures, and reported results for the semester thesis on model-based ECG delineation using ALSSM.

The files are grouped by methodological role so that the workflow remains transparent and reproducible.

---

## Important before running any script

Some dataset archives must be **unzipped first** before the scripts can use them.

Please extract these archives before running the code:

- `mit-bih-arrhythmia-database-1.0.7z`
- `lobachevsky-university-electrocardiography-database-1.0.1.zip`

The archive files themselves are not sufficient. They must be extracted first.

If environment variables are used, the scripts expect the following:

- `MIT_BIH_DIR` for the MIT-BIH Arrhythmia Database
- `LUDB_DIR` for the LUDB dataset
- `QT_DATABASE_DIR` for the QT Database

Otherwise, the scripts search the thesis folder structure for the extracted dataset folders.

---

## Folder purpose

Only scripts that are directly relevant to the thesis text, final figures, reported results, or appendix material explicitly mentioned in the thesis are included here.

Older prototypes, debugging scripts, unused detector variants, and unrelated experiments were intentionally excluded.

---

## Folder overview

### `dataset_figure_examples`
Scripts for early introductory dataset figures used in the thesis text.

- `plot_ludb_example.py`  
  Creates the LUDB example figure with manual P, R, and T annotations.

- `plot_ludb_detection_comparison.py`  
  Creates the LUDB comparison figure showing ground truth versus ALSSM detections.

- `plot_qt_annotation_example.py`  
  Creates the QT Database example figure with manual annotations.

- `plot_synthetic_ecg_example.py`  
  Creates the synthetic ECG example windows figure.

---

### `datasets_and_preprocessing`
Scripts used for dataset construction, preprocessing, resampling, annotation mapping, and validation.

- `plot_qt_resampling_example.py`  
  Creates the QT resampling comparison figure used in the preprocessing chapter.

- `build_qt_best_lead_record_example_500hz.py`  
  Demonstrates QT best-lead selection and annotation mapping on a single record.

- `build_qt_best_lead_dataset_500hz.py`  
  Builds the final QT best-lead 500 Hz beat-centred dataset and segmentation masks.

- `validate_qt_best_lead_dataset.py`  
  Computes summary statistics and validation plots for the QT best-lead dataset.

- `plot_qt_best_lead_selection.py`  
  Creates the figure illustrating the automatic best-lead selection criterion.

- `plot_beat_centered_window_example.py`  
  Creates the beat-centred ECG window and segmentation-mask figure.

- `build_ludb_best_lead_dataset.py`  
  Builds the LUDB best-lead dataset used in the early annotated stage.

- `build_synthetic_pqt_dataset.py`  
  Builds the synthetic ECG dataset used in the early controlled experiments.

---

### `r_and_qrs`
Scripts related to R detection and QRS modelling.

- `build_mit_r_template_500hz.py`  
  Builds the MIT-derived R template used in later feature extraction and R validation.

- `evaluate_mit_r_template_generalization.py`  
  Evaluates the MIT-derived R detector across multiple MIT-BIH records.

- `plot_multiscale_qrs_examples.py`  
  Creates the multiscale QRS figure with narrow and broad examples.

- `plot_qrs_detection_raw_and_whitened_states.py`  
  Creates the raw-state and whitened-state QRS pipeline figures with cost and LCR.

---

### `pt_detection_and_examples`
Scripts for pointwise P/T detection and qualitative example figures.

- `plot_anchor_based_pt_search_windows.py`  
  Creates the anchor-based P and T search-window figure.

- `plot_pointwise_pt_template_detection.py`  
  Creates the point-template P/T detection figure.

- `plot_qt_end_to_end_delineation.py`  
  Creates the qualitative QT Database end-to-end delineation figure.

- `plot_single_beat_template_diagnostics.py`  
  Creates the detailed single-beat diagnostic figure with raw and whitened trajectories, cost, and LCR.

- `plot_mit_example_with_lcr_and_trajectory.py`  
  Creates an MIT example with trajectory, LCR, and detections.

---

### `t_wave_modeling`
Scripts for T-wave trajectory modelling and comparison experiments.

- `t_wave_modeling_utils.py`  
  Shared helper functions for the QT T-wave modelling scripts.

- `plot_t_point_template_baseline_examples.py`  
  Creates representative examples of point-template T-wave failures in the broad window.

- `plot_t_point_template_baseline_error.py`  
  Creates the baseline timing-error distribution figure for point-template T-wave detection.

- `plot_t_broad_wave_comparison_n2_vs_n3.py`  
  Creates the broad T-wave comparison figure between the baseline and exploratory third-order variant.

- `analyze_t_wave_method_statistics.py`  
  Computes the statistical comparison between point-template, single-trajectory, and clustered-trajectory methods.

- `plot_t_single_vs_clustered_comparison.py`  
  Creates the comparison figure for single versus clustered trajectory templates.

---

### `layer2`
Scripts for Layer 2 feature extraction and classification.

- `build_qt_point_templates_500hz.py`  
  Builds the original QT-derived point templates and whitening matrices for Layer 2.

- `build_curated_qt_point_templates.py`  
  Builds the curated QT-derived point templates and whitening matrices.

- `train_layer2_svm_with_qt_templates.py`  
  Runs Layer 2 classification on MIT-BIH using QT-derived templates.

- `train_layer2_svm_with_curated_qt_templates.py`  
  Runs Layer 2 classification on MIT-BIH using curated QT-derived templates.

- `layer2_mit_pipeline.py`  
  Main MIT-derived Layer 2 feature extraction and classification pipeline.

- `plot_layer2_example_beats.py`  
  Creates the Layer 2 example-beat figure.

- `plot_layer2_roc_curve.py`  
  Creates the main Layer 2 ROC figure.

- `export_layer2_mit_results_json.py`  
  Exports the main MIT-derived Layer 2 performance metrics to JSON.

---

### `appendix_support`
Appendix-only or older support material retained because it is explicitly mentioned in the thesis appendix or useful for documentation.

These files are not part of the main final figure workflow.

---

### `json_reports`
Saved JSON result summaries used for reporting numerical results in the thesis.

---

### `npz_templates`
Saved `.npy` and `.npz` files containing templates, whitening matrices, datasets, feature matrices, and intermediate processed outputs.

---

### `generated_outputs`
Generated figures and other saved outputs produced by the scripts.

---

## Figure responsibility map

This mapping is based on the current thesis version.

### Main text figures

- **Figure 1**  
  LUDB example with manual P, R, and T annotations  
  Script: `dataset_figure_examples/plot_ludb_example.py`

- **Figure 2**  
  LUDB comparison between manual annotation and ALSSM detection  
  Script: `dataset_figure_examples/plot_ludb_detection_comparison.py`

- **Figure 3**  
  QT Database example with manual annotations  
  Script: `dataset_figure_examples/plot_qt_annotation_example.py`

- **Figure 4**  
  MIT-BIH ECG example with detected P, R, and T positions  
  Script: `pt_detection_and_examples/plot_mit_example_with_lcr_and_trajectory.py`

- **Figure 5**  
  Synthetic ECG example windows  
  Script: `dataset_figure_examples/plot_synthetic_ecg_example.py`

- **Figure 6**  
  QT resampling comparison  
  Script: `datasets_and_preprocessing/plot_qt_resampling_example.py`

- **Figure 7**  
  Best-lead selection criterion  
  Script: `datasets_and_preprocessing/plot_qt_best_lead_selection.py`

- **Figure 8**  
  Beat-centred ECG window and segmentation mask  
  Script: `datasets_and_preprocessing/plot_beat_centered_window_example.py`

- **Figure 9**  
  ALSSM segment geometry and weighting  
  Script: `plot_alssm_segment_weight.py`

- **Figure 10**  
  Whitening transformation in ALSSM state space  
  Script: `plot_alssm_whitening.py`

- **Figure 11**  
  Multiscale QRS concept  
  Script: `r_and_qrs/plot_multiscale_qrs_examples.py`

- **Figure 12**  
  QRS detection pipeline with raw and whitened state trajectories  
  Script: `r_and_qrs/plot_qrs_detection_raw_and_whitened_states.py`

- **Figure 13**  
  MIT-BIH ECG segment with detected R peaks  
  Script: `r_and_qrs/evaluate_mit_r_template_generalization.py`

- **Figure 14**  
  Anchor-based P and T search windows  
  Script: `pt_detection_and_examples/plot_anchor_based_pt_search_windows.py`

- **Figure 15**  
  Point-template P- and T-wave detection example  
  Script: `pt_detection_and_examples/plot_pointwise_pt_template_detection.py`

- **Figure 16**  
  Qualitative QT end-to-end delineation example  
  Script: `pt_detection_and_examples/plot_qt_end_to_end_delineation.py`

- **Figure 17**  
  Broad-window point-template T-wave failure examples  
  Script: `t_wave_modeling/plot_t_point_template_baseline_examples.py`

- **Figure 18**  
  Timing-error distribution of the point-template T-wave baseline  
  Script: `t_wave_modeling/plot_t_point_template_baseline_error.py`

- **Figure 19**  
  Layer 2 example beat comparison  
  Script: `layer2/plot_layer2_example_beats.py`

- **Figure 20**  
  ROC curve of the main Layer 2 classifier  
  Script: `layer2/plot_layer2_roc_curve.py`

### Appendix figure

- **Figure 21**  
  Comparison of point-template, single-trajectory, and clustered-trajectory T-wave errors  
  Script: `t_wave_modeling/plot_t_single_vs_clustered_comparison.py`

---

## Main reported result scripts

### Main detection and modelling

- QT best-lead dataset construction  
  `datasets_and_preprocessing/build_qt_best_lead_dataset_500hz.py`

- MIT R-template construction  
  `r_and_qrs/build_mit_r_template_500hz.py`

- MIT R-template generalization  
  `r_and_qrs/evaluate_mit_r_template_generalization.py`

### Main T-wave analysis

- T-wave method statistics  
  `t_wave_modeling/analyze_t_wave_method_statistics.py`

### Main Layer 2 results

- MIT-derived Layer 2 pipeline  
  `layer2/layer2_mit_pipeline.py`

- QT-derived Layer 2 run  
  `layer2/train_layer2_svm_with_qt_templates.py`

- Curated QT-derived Layer 2 run  
  `layer2/train_layer2_svm_with_curated_qt_templates.py`

---

## Appendix-only material

The appendix clustering subsection mainly relies on:

- `t_wave_modeling/t_wave_modeling_utils.py`
- `t_wave_modeling/analyze_t_wave_method_statistics.py`
- `t_wave_modeling/plot_t_single_vs_clustered_comparison.py`

Additional appendix/support scripts may be kept in `appendix_support`, but they are not part of the main final thesis pipeline.

---

## Notes

- Some figures were created after several intermediate script versions. Only the final selected figure-generator scripts were retained here.
- Duplicate development versions were intentionally excluded.
- The dataset archives must be extracted before running the scripts. In particular, `mit-bih-arrhythmia-database-1.0.7z` and `lobachevsky-university-electrocardiography-database-1.0.1.zip` must be unzipped first.

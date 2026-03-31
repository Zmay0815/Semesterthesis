# Semester Thesis ECG Scripts

This folder contains the final scripts used to generate the datasets, figures, and reported results for the semester thesis on model-based ECG delineation using ALSSM.

The files are grouped by task:
- core utilities
- dataset and preprocessing scripts
- R/QRS modelling
- P/T detection and qualitative examples
- T-wave modelling
- Layer 2 classification

Each major figure or reported result is linked to one dedicated script whenever possible, in order to keep the workflow transparent and reproducible.
## Folder purpose

The files in this folder are limited to scripts that are directly relevant for the thesis text, figures, main reported results, or appendix material that is explicitly mentioned in the thesis.

Older prototypes, debugging scripts, Morse-code experiments, and unused alternative detector versions were intentionally excluded.

---

## Script overview

### Core utilities

- `alssm_pipeline_utils.py`  
  Shared ALSSM helper functions used in several early and intermediate scripts.

- `adaptive_peak_thresholding.py`  
  Adaptive thresholding and peak-picking utilities used for LCR-based detection.

- `multiscale_model_params.py`  
  Central parameter definitions for multiscale wave modelling.

---

### Dataset and preprocessing scripts

- `plot_qt_resampling_example.py`  
  Creates the QT resampling comparison figure used in the preprocessing chapter.

- `build_qt_best_lead_record_example_500hz.py`  
  Demonstrates QT best-lead selection and annotation mapping on a record level.

- `build_qt_best_lead_dataset_500hz.py`  
  Builds the final QT best-lead 500 Hz beat-centred dataset and segmentation masks.

- `validate_qt_best_lead_dataset.py`  
  Computes summary statistics and validation plots for the QT best-lead dataset.

- `plot_qt_best_lead_selection.py`  
  Creates the figure illustrating the automatic best-lead selection criterion.

- `plot_beat_centered_window_example.py`  
  Creates the beat-centred window and segmentation-mask figure.

- `build_ludb_best_lead_dataset.py`  
  Builds the LUDB best-lead dataset used in the early annotated stage.

- `build_synthetic_pqt_dataset.py`  
  Builds the synthetic ECG dataset used in the early controlled experiments.

---

### R detection and QRS modelling

- `build_mit_r_template_500hz.py`  
  Builds the MIT-derived R template used in later feature extraction and R validation.

- `evaluate_mit_r_template_generalization.py`  
  Evaluates the MIT-derived R detector across multiple MIT-BIH records.

- `plot_multiscale_qrs_examples.py`  
  Creates the multiscale QRS figure with narrow and broad examples.

- `plot_qrs_detection_raw_and_whitened_states.py`  
  Creates the raw-state and whitened-state QRS pipeline figures with cost and LCR.

---

### Pointwise P/T detection and qualitative examples

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

### T-wave modelling scripts

- `t_wave_modeling_utils.py`  
  Shared helper functions for the QT T-wave modelling scripts.

- `plot_t_point_template_baseline_examples.py`  
  Creates the representative examples of point-template T-wave failures in the broad window.

- `plot_t_point_template_baseline_error.py`  
  Creates the baseline timing-error distribution figure for point-template T-wave detection.

- `plot_t_broad_wave_comparison_n2_vs_n3.py`  
  Creates the broad T-wave comparison figure between the baseline and the exploratory third-order variant.

- `analyze_t_wave_method_statistics.py`  
  Computes the statistical comparison between point-template, single-trajectory, and clustered-trajectory methods.

- `plot_t_single_vs_clustered_comparison.py`  
  Creates the comparison figure for single versus clustered trajectory templates.

---

### Layer 2 scripts

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

## Figure responsibility map

Figure numbers may shift during thesis editing. The mapping below is therefore based on figure content, not only on the final number.

### Dataset and preprocessing figures
- LUDB example figure: created earlier from the LUDB workflow, if retained separately.
- LUDB detection comparison figure: created earlier from the LUDB workflow, if retained separately.
- QT resampling comparison: `plot_qt_resampling_example.py`
- Best-lead selection figure: `plot_qt_best_lead_selection.py`
- Beat-centred ECG window and segmentation mask: `plot_beat_centered_window_example.py`

### QRS and search-window figures
- Multiscale QRS concept figure: `plot_multiscale_qrs_examples.py`
- QRS raw-state and whitened-state pipeline figure: `plot_qrs_detection_raw_and_whitened_states.py`
- Anchor-based P and T search-window figure: `plot_anchor_based_pt_search_windows.py`
- Point-template P/T detection figure: `plot_pointwise_pt_template_detection.py`

### Qualitative delineation figures
- QT end-to-end delineation figure: `plot_qt_end_to_end_delineation.py`
- Single-beat diagnostic figure with raw and whitened trajectories: `plot_single_beat_template_diagnostics.py`
- MIT example figure: `plot_mit_example_with_lcr_and_trajectory.py`

### T-wave methodology figures
- Point-template baseline examples: `plot_t_point_template_baseline_examples.py`
- Point-template baseline error distribution: `plot_t_point_template_baseline_error.py`
- Broad T-wave comparison \(n=2\) versus \(n=3\): `plot_t_broad_wave_comparison_n2_vs_n3.py`
- Single versus clustered trajectory comparison: `plot_t_single_vs_clustered_comparison.py`

### Layer 2 figures
- Layer 2 example beats figure: `plot_layer2_example_beats.py`
- Layer 2 ROC figure: `plot_layer2_roc_curve.py`

---

## Main reported result scripts

### Main detection and modelling
- QT best-lead dataset construction: `build_qt_best_lead_dataset_500hz.py`
- MIT R-template construction: `build_mit_r_template_500hz.py`
- MIT R-template generalization: `evaluate_mit_r_template_generalization.py`

### Main T-wave analysis
- T-wave method statistics: `analyze_t_wave_method_statistics.py`

### Main Layer 2 results
- MIT-derived Layer 2 pipeline: `layer2_mit_pipeline.py`
- QT-derived Layer 2 run: `train_layer2_svm_with_qt_templates.py`
- Curated-QT-derived Layer 2 run: `train_layer2_svm_with_curated_qt_templates.py`

---

## Appendix-only material

In the appendix in the clustering subsection, the relevant clustering files are:

- `t_wave_modeling_utils.py`
- `analyze_t_wave_method_statistics.py`
- `plot_t_single_vs_clustered_comparison.py`

---

## Notes

- Some figures were created after several intermediate script versions. Only the final selected figure-generator scripts were retained here.
- Duplicate development versions were intentionally excluded.
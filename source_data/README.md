# Source Data Index

This folder contains derived, non-sensitive CSV files used to support the manuscript figures, numerical claims and Supplementary Information. Raw third-party data archives are not redistributed here.

Machine-readable maps:

- `source_data_index.csv`: file-to-claim, source-family and generating-script map.
- `dataset_registry.csv`: official source URLs/DOIs and raw-data redistribution boundaries.
- `LARGE_DERIVED_OUTPUTS.csv`: dense derived CSVs omitted from Git tracking because
  they exceed GitHub file-size limits, with regeneration scripts.

Each source-data CSV can be regenerated from the listed script after obtaining the public third-party datasets named in `dataset_registry.csv`.

## Files

| Source data file | Evidence module | Related figure/section | Raw data family |
| --- | --- | --- | --- |
| `camels_673_blocked_crossfit_memory_coordinate_metrics.csv` | CAMELS-US main memory-coordinate diagnostics | Figs. 1-2; Results: discharge persistence and memory-normalized dispersion | CAMELS-US |
| `camels_673_collapse_metrics.csv` | CAMELS-US main memory-coordinate diagnostics | Figs. 1-2; Results: discharge persistence and memory-normalized dispersion | CAMELS-US |
| `camels_673_memory_coordinate_attribute_modifiers.csv` | CAMELS-US main memory-coordinate diagnostics | Figs. 1-2; Results: discharge persistence and memory-normalized dispersion | CAMELS-US |
| `camels_673_psd_estimator_robustness_metrics.csv` | CAMELS-US main memory-coordinate diagnostics | Figs. 1-2; Results: discharge persistence and memory-normalized dispersion | CAMELS-US |
| `camels_gb_v2_replication_collapse_metrics.csv` | CAMELS-GB v2 replication | Figs. 1-2; Results: discharge persistence and memory-normalized dispersion | CAMELS-GB v2 |
| `fourier_pair_circularity_baseline_metrics.csv` | Fourier-pair circularity baseline | Supplementary tables and source-data checks | CAMELS-US; CAMELS-GB v2 |
| `r16_convergent_alignment_cross_validation.csv` | Functional-distance convergence and held-out-gauge validation | Supplementary tables and source-data checks | CAMELS-US; CAMELS-GB v2 |
| `r16_convergent_alignment_metric_summary.csv` | Functional-distance convergence and held-out-gauge validation | Supplementary tables and source-data checks | CAMELS-US; CAMELS-GB v2 |
| `r19_ar1_conditioned_residual_attribute_modifiers.csv` | AR(1)-conditioned residual diagnostic with median reference | Fig. 3 | CAMELS-US; CAMELS-GB v2 |
| `r19_ar1_conditioned_residual_by_gauge.csv` | AR(1)-conditioned residual diagnostic with median reference | Fig. 3 | CAMELS-US; CAMELS-GB v2 |
| `r19_ar1_conditioned_residual_curves.csv` | AR(1)-conditioned residual diagnostic with median reference | Fig. 3 | CAMELS-US; CAMELS-GB v2 |
| `r19_ar1_conditioned_residual_metrics.csv` | AR(1)-conditioned residual diagnostic with median reference | Fig. 3 | CAMELS-US; CAMELS-GB v2 |
| `r20_support_matched_coordinate_sensitivity.csv` | Support-matched and gauge-matched AR(1) artifact-floor gates | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r23_camels_br_third_archive_archive_metrics.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_bin_stats.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_curves.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_gauge_summary.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_random_tau_null.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_storage_proxy_associations.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_camels_br_third_archive_subsample_stability.csv` | CAMELS-BR tropical third-archive replication | Fig. 2; Figs. S1-S2 | CAMELS-BR v1.2 |
| `r23_cross_archive_storage_proxy_validation.csv` | Cross-archive storage-proxy validation | Supplementary tables and source-data checks | CAMELS-US; CAMELS-GB v2; CAMELS-BR |
| `r23_groundwater_storage_validation_hydrogeology_associations.csv` | Groundwater/storage validation screen | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells |
| `r23_groundwater_storage_validation_nearest_gauge_sensitivity.csv` | Groundwater/storage validation screen | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells |
| `r23_groundwater_storage_validation_well_summary.csv` | Groundwater/storage validation screen | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells |
| `r24_groundwater_boundary_validation_all_containment.csv` | Official-boundary groundwater-well containment check | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells and boundaries |
| `r24_groundwater_boundary_validation_association_summary.csv` | Official-boundary groundwater-well containment check | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells and boundaries |
| `r24_groundwater_boundary_validation_catchment_summary.csv` | Official-boundary groundwater-well containment check | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells and boundaries |
| `r24_groundwater_boundary_validation_smallest_containing_wells.csv` | Official-boundary groundwater-well containment check | Fig. 2; Fig. S3 | CAMELS-GB v2 groundwater wells and boundaries |
| `r25_camels_aus_fourth_archive_archive_metrics.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_bin_stats.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_gauge_summary.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_random_tau_null.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_storage_proxy_associations.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_strata_metrics.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_camels_aus_fourth_archive_subsample_stability.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_four_archive_portability_synthesis_archive_summary.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r25_four_archive_portability_synthesis_proxy_summary.csv` | CAMELS-AUS fourth-archive and portability synthesis | Fig. 2; Figs. S1-S2 | CAMELS-AUS; cross-archive derived tables |
| `r26_four_archive_uncertainty_bootstrap_draws.csv` | Four-archive bootstrap uncertainty | Fig. 2; Figs. S1-S2 | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS |
| `r26_four_archive_uncertainty_summary.csv` | Four-archive bootstrap uncertainty | Fig. 2; Figs. S1-S2 | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS |
| `r27_camels_dk_groundwater_storage_validation_archive_metrics.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_bin_stats.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_coordinate_sensitivity.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_curves.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_evidence_ladder.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_gauge_summary.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_random_tau_null.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_storage_proxy_associations.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_storage_state_memory_associations.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r27_camels_dk_groundwater_storage_validation_subsample_stability.csv` | CAMELS-DK lowland groundwater/storage stress test | Fig. 2; Fig. S4 | CAMELS-DK |
| `r28_tracer_compatibility_validation_camels_chem_associations.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r28_tracer_compatibility_validation_camels_chem_gauge_metrics.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r28_tracer_compatibility_validation_camels_chem_pair_metrics.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r28_tracer_compatibility_validation_evidence_ladder.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r28_tracer_compatibility_validation_plynlimon_flow_memory_proxy.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r28_tracer_compatibility_validation_plynlimon_tracer_damping.csv` | Tracer-facing compatibility and causal-boundary screen | Fig. S5 | CAMELS-Chem; Plynlimon tracer supplement; CAMELS-US |
| `r30_mechanism_evidence_mother_figure_source.csv` | Mechanism-evidence synthesis figure source | Fig. 1 | Cross-archive derived tables |
| `r31_gauge_level_ar1_boundary_attribute_correlations.csv` | Gauge-level matched-AR(1) residual boundary | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r31_gauge_level_ar1_boundary_gauge_scores.csv` | Gauge-level matched-AR(1) residual boundary | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r31_gauge_level_ar1_boundary_strata.csv` | Gauge-level matched-AR(1) residual boundary | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r31_gauge_level_ar1_boundary_summary.csv` | Gauge-level matched-AR(1) residual boundary | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r33_matched_ar1_ensemble_draw_metrics.csv` | Multi-draw gauge-matched AR(1) ensemble and null-calibrated score | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r33_matched_ar1_ensemble_gauge_diagnostics.csv` | Multi-draw gauge-matched AR(1) ensemble and null-calibrated score | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r33_matched_ar1_ensemble_summary.csv` | Multi-draw gauge-matched AR(1) ensemble and null-calibrated score | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r33_null_calibrated_ensemble_summary.csv` | Multi-draw gauge-matched AR(1) ensemble and null-calibrated score | Fig. 5; Fig. S6 | CAMELS-US; CAMELS-GB v2 |
| `r35_archive_filter_audit_summary.csv` | Five-archive Fig. 2 alignment, shuffled-memory null and filter audit | Fig. 2; Fig. 3; Methods: archive filtering | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK |
| `r35_fig2_five_archive_alignment_source.csv` | Five-archive Fig. 2 alignment, shuffled-memory null and filter audit | Fig. 2; Fig. 3; Methods: archive filtering | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK |
| `r35_fig4_robustness_support_source.csv` | Five-archive Fig. 2 alignment, shuffled-memory null and filter audit | Fig. 2; Fig. 3; Methods: archive filtering | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK |
| `r36_construct_validity_diagnostics.csv` | De construct-validity and seasonality diagnostics | Fig. 4; Fig. S8; Supplementary Note 9; Methods: multiscale and construct-validity controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; synthetic controls |
| `r36_multiscale_process_family_nulls_curves.csv` | Multiscale and long-memory process-family null controls | Fig. 4; Fig. S8; Supplementary Note 9; Methods: multiscale and construct-validity controls | Synthetic simulations; CAMELS-US/GB gauge-matched script scaffold |
| `r36_multiscale_process_family_nulls_metrics.csv` | Multiscale and long-memory process-family null controls | Fig. 4; Fig. S8; Supplementary Note 9; Methods: multiscale and construct-validity controls | Synthetic simulations; CAMELS-US/GB gauge-matched script scaffold |
| `r36_multiscale_process_family_nulls_series.csv` | Multiscale and long-memory process-family null controls | Fig. 4; Fig. S8; Supplementary Note 9; Methods: multiscale and construct-validity controls | Synthetic simulations; CAMELS-US/GB gauge-matched script scaffold |
| `r37_gauge_matched_analytic_nulls_summary.csv` | Full-archive gauge-matched analytic stochastic-memory nulls | Fig. 4; Fig. S9; Supplementary Note 10; Methods: gauge-matched analytic nulls | CAMELS-US; CAMELS-GB v2; analytic PSD families |
| `r39_open_model_intercomparison_lstm_training_log.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `r39_open_model_intercomparison_metrics.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `r39_open_model_intercomparison_pairwise_tests.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `r39_open_model_intercomparison_rank_disagreement.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `r39_open_model_intercomparison_run_manifest.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `r39_open_model_intercomparison_summary.csv` | Open GR4J/HBV-Edu/LSTM hydrological model benchmark | Fig. 6 | CAMELS-GB v2 |
| `final_extreme_hardening_cluster_bootstrap_draws.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_cluster_bootstrap_summary.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_forcing_control_summary.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_forcing_precipitation_curves.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_forcing_precipitation_summary.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_forcing_q_minus_p_contrast.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_extreme_hardening_slow_memory_support_audit.csv` | Robustness hardening: cluster bootstrap, forcing-side control and slow-memory support audit | Fig. 4; Methods: spatial-dependence, forcing-side and slow-memory controls | CAMELS-US; CAMELS-GB v2; CAMELS-BR; CAMELS-AUS; CAMELS-DK; CAMELS-GB v2 meteorological forcing |
| `final_model_ensemble_hardening_lstm_seed_metrics.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_lstm_seed_summary.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_lstm_training_log.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_rrmpg_512_metrics.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_rrmpg_512_summary.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_rrmpg_512_vs_128_paired_delta.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_seasonal_ar1_draw_summary.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_seasonal_ar1_metrics.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `final_model_ensemble_hardening_summary.csv` | Model ensemble and calibration-budget sensitivity | Fig. 6; Fig. S10; Methods: model ensemble sensitivity | CAMELS-GB v2; RRMPG GR4J/HBV-Edu outputs; seasonal AR(1) null draws; global LSTM seed sensitivity |
| `synthetic_reservoir_process_control_metrics.csv` | Synthetic storage and process controls | Supplementary tables and source-data checks | Synthetic simulations |

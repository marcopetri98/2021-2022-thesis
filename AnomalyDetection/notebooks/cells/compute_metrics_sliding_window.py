true_labels = data_test_labels
compute_metrics(true_labels, scores, labels, only_roc_auc=False)
make_metric_plots(dataframe, true_labels, scores, labels)

bars = get_bars_indices_on_test_df(all_df,
                                   dataframe,
                                   PURE_DATA_KEY,
                                   GROUND_WINDOWS_PATH)
plot_time_series_with_predicitons_bars(dataframe,
                                       labels,
                                       bars,
                                       pred_color='r')
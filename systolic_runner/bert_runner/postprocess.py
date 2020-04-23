import run_onnx_squad

# postprocess results
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

write_predictions(
    eval_examples,
    extra_data,
    all_results,
    n_best_size,
    max_answer_length,
    True,
    output_prediction_file,
    output_nbest_file
)

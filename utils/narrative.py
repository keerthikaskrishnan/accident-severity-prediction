import numpy as np

def generate_narrative(input_row, shap_values, feature_names, pred_label):
    abs_vals = np.abs(shap_values)
    top_idx = np.argsort(abs_vals)[-3:][::-1]

    reasons = []
    for idx in top_idx:
        feature = feature_names[idx]
        value = input_row[feature]
        impact = shap_values[idx]
        direction = "increased" if impact > 0 else "reduced"
        reasons.append(f"{feature} = {value} {direction} the predicted severity")

    severity_map = {0: "Slight", 1: "Serious", 2: "Fatal"}
    sev_text = severity_map.get(pred_label, "Unknown")

    narrative = (
        f"The model predicts a **{sev_text}** accident.\n\n"
        "Key contributing factors:\n"
        f"- {reasons[0]}\n"
        f"- {reasons[1]}\n"
        f"- {reasons[2]}\n\n"
        "These features had the strongest influence on the model’s decision."
    )
    return narrative

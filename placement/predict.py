import argparse
import json
from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "placement_model.joblib"
DEFAULT_THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


def main():

    parser = argparse.ArgumentParser(
        description="Predict student placement"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to test CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output prediction CSV"
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model"
    )

    parser.add_argument(
        "--threshold",
        default=str(DEFAULT_THRESHOLD_PATH),
        help="Path to threshold file"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------

    test_df = pd.read_csv(args.input)

    print(f"Loaded test data: {test_df.shape}")

    # Keep student IDs for the final submission
    student_ids = test_df["student_id"].copy()

    # --------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------

    model = joblib.load(args.model)

    # --------------------------------------------------
    # 3. Load selected threshold
    # --------------------------------------------------

    with open(args.threshold, "r") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data["threshold"]
    )

    print(f"Using threshold: {threshold:.2f}")

    # --------------------------------------------------
    # 4. Prepare features
    # --------------------------------------------------

    X_test = test_df.drop(
        columns=[
            "student_id",
            "status_code"
        ],
        errors="ignore"
    )

    # --------------------------------------------------
    # 5. Predict probabilities
    # --------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # 6. Apply optimized threshold
    # --------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------
    # 7. Create submission
    # --------------------------------------------------

    submission = pd.DataFrame({
        "student_id": student_ids,
        "placed": predictions
    })

    # --------------------------------------------------
    # 8. Save predictions
    # --------------------------------------------------

    submission.to_csv(
        args.output,
        index=False
    )

    print(
        f"Predictions saved to: {args.output}"
    )

    print("\nPrediction distribution:")
    print(
        submission["placed"].value_counts()
    )


if __name__ == "__main__":
    main()
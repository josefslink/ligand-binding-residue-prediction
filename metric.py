# Course-provided scorer for this challenge — unmodified except for this header
# comment. Computes pooled (micro) Jaccard/IoU, the exact metric used by the
# challenge server. See THIRD_PARTY.md.

from sklearn.metrics import jaccard_score
import pandas as pd
import argparse

def parse_submission(df_submission):
    df_submission["prediction"] = df_submission["prediction"].fillna("")
    predictions_residues = []
    for _, r in df_submission.iterrows():
        for res in r['prediction'].strip().split(" "):
            predictions_residues.append((r['id'], res.strip()))
    df_predictions_residues = pd.DataFrame(predictions_residues, columns=["id", "residue_id"])
    df_predictions_residues["prediction"] = 1
    return df_predictions_residues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    # read files
    df_submission = pd.read_csv(args.submission)
    df_submission = parse_submission(df_submission)

    df_target = None
    if args.target is not None:
        df_target = pd.read_csv(args.target)

    df_target_prediction = pd.merge(df_target, df_submission, on=["id", "residue_id"], how="left")
    df_target_prediction["prediction"] = df_target_prediction["prediction"].fillna(0)

    y_score = df_target_prediction["prediction"]
    y_true = df_target_prediction["true"]

    # calculate iou
    print(jaccard_score(y_true, y_score))
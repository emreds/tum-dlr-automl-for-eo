import os
import pandas as pd

LOG_FOLDER = "/p/project/hai_nasb_eo/training/logs"

def extract(x): 
    return x[0]

if __name__ == "__main__":
    dirs = os.listdir(LOG_FOLDER)
    valid_dirs = []
    csv_paths = []

    for dir_name in dirs:
        dir_ = dir_name.split("_")
        if len(dir_) == 2:
            valid_dirs.append(dir_name)

    for dir in valid_dirs:
        csv_path = os.path.join(LOG_FOLDER, dir, "version_0", "metrics.csv")
        csv_paths.append(csv_path)
        
    data_cols = [
        "validation_loss",
        "validation_accuracy",
        "validation_avg_macro_accuracy",
        "validation_avg_micro_accuracy",
        "epoch",
        "step",
        "train_loss",
        "train_accuracy",
        "train_avg_macro_accuracy",
        "train_avg_micro_accuracy",
        "training_time",
    ]
    
    # /p/project/hai_nasb_eo/training/logs/arch_31/version_0/metrics.csv
    results = []
    for path in csv_paths:
        idx = int(path.split("/")[-3].split("_")[-1])

        try:
            data = (
                pd.read_csv(path)
                .tail(2)
                .fillna(method="ffill")
                .fillna(method="bfill")
                .drop_duplicates()
            )[data_cols].reset_index().to_dict('list')
            data['arch_num'] = idx
            results.append(data)
        except Exception as e:
            print(f"Error happened in this path: {path}")
            print(f"This is the error {e}")

    df = pd.DataFrame(results)
    print(len(df))
    for col in data_cols:
        df[col] = df[col].map(extract)
        
    df.drop(['index'], axis=1, inplace=True)
    print(df.columns)
    print(len(df))
    print(df)
    df.to_csv('./all_metrics.csv')

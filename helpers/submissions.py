import os
import pandas as pd
from datetime import datetime


def prepare_submission(model, X_test, X_id, name, folder = 'submissions') ->str:
    predictions = model.predict_proba(X_test)[:, 1]
    submission_df = pd.DataFrame({'SK_ID_CURR': X_id, 'TARGET': predictions})
    dat = datetime.now().strftime('%Y-%m-%d_%H-%M')
    file_path = os.path.join(folder, f'sub_{name}_{dat}.csv')

    submission_df.to_csv(file_path, index=False)
    print(f'Submission created: {file_path}')
    return file_path
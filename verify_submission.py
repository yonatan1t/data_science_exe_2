import pandas as pd

def check_project_ids():
    print("Loading new_projects.csv and 123456789.csv...")
    test_df = pd.read_csv('new_projects.csv')
    pred_df = pd.read_csv('123456789.csv')
    
    print(f"Total rows in new_projects.csv: {len(test_df)}")
    print(f"Total rows in 123456789.csv:    {len(pred_df)}")
    
    missing_ids = set(test_df['id']) - set(pred_df['id'])
    extra_ids = set(pred_df['id']) - set(test_df['id'])
    duplicate_ids = pred_df['id'].duplicated().sum()
    null_preds = pred_df['state_ind_pred'].isnull().sum()
    order_exact = (test_df['id'].values == pred_df['id'].values).all()
    
    print(f"Missing IDs:             {len(missing_ids)}")
    print(f"Extra IDs:               {len(extra_ids)}")
    print(f"Duplicate IDs:           {duplicate_ids}")
    print(f"Null predictions:        {null_preds}")
    print(f"Exact row order match:   {order_exact}")
    
    if len(missing_ids) == 0 and len(extra_ids) == 0 and duplicate_ids == 0 and null_preds == 0:
        print("\n✅ PASSED: All 1,990 project IDs are present, unique, and valid!")
    else:
        print("\n❌ FAILED: Found mismatch in project IDs or missing predictions.")

if __name__ == '__main__':
    check_project_ids()

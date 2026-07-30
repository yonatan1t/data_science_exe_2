import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import openpyxl

def engineer_features(df):
    df = df.copy()
    
    # Financial features
    df.currency_rate = df.currency_rate.fillna(1.0)
    df['goal_usd'] = df.goal * df.currency_rate
    df['log_goal_usd'] = np.log1p(np.maximum(df.goal_usd, 0))
    df['log_goal'] = np.log1p(np.maximum(df.goal, 0))
    
    # Duration and ratios
    df['launch_to_deadline_days_clean'] = np.maximum(df.launch_to_deadline_days, 1)
    df['create_to_launch_days_clean'] = np.maximum(df.create_to_launch_days, 0)
    
    df['goal_usd_per_day'] = df.goal_usd / df.launch_to_deadline_days_clean
    df['log_goal_usd_per_day'] = np.log1p(df.goal_usd_per_day)
    df['prep_ratio'] = df['create_to_launch_days_clean'] / (df['launch_to_deadline_days_clean'] + 1)
    
    # Staff pick
    df.staff_pick = df.staff_pick.astype(int)
    
    # Text features from name
    df['name_str'] = df.name.fillna('').astype(str)
    df['name_char_len'] = df.name_str.apply(len)
    df['name_word_count'] = df.name_str.apply(lambda x: len(x.split()))
    df['name_has_excl'] = df.name_str.apply(lambda x: 1 if '!' in x else 0)
    df['name_has_quest'] = df.name_str.apply(lambda x: 1 if '?' in x else 0)
    df['name_is_upper'] = df.name_str.apply(lambda x: 1 if x.isupper() else 0)
    
    # Fill NAs
    df.category = df.category.fillna('Missing')
    df.name_len = df.name_len.fillna(df.name_len.median())
    df.name_len_clean = df.name_len_clean.fillna(df.name_len_clean.median())
    
    # Category goal ratio
    cat_medians = df.groupby('category')['goal_usd'].transform('median')
    df['goal_to_cat_median'] = df['goal_usd'] / (cat_medians + 1.0)
    df['log_goal_to_cat_median'] = np.log1p(np.maximum(df['goal_to_cat_median'], 0))
    
    # Cyclic date features
    for col, max_val in [('launched_at_month', 12), ('launched_at_hr', 24), ('deadline_month', 12), ('deadline_hr', 24)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * getattr(df, col) / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * getattr(df, col) / max_val)
        
    return df

def main():
    print("Loading data...")
    train_df = pd.read_csv('kickstarter_projects.csv')
    test_df = pd.read_csv('new_projects.csv')
    
    # Feature engineering
    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)
    
    drop_cols = ['id', 'name', 'slug', 'source_url', 'state_ind']
    feature_cols = [c for c in train_eng.columns if c not in drop_cols]
    
    X_train = train_eng[feature_cols]
    y_train = train_eng.state_ind
    X_test = test_eng[feature_cols]
    
    cat_cols = ['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
    num_cols = [c for c in feature_cols if c not in cat_cols and c != 'name_str']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('text', TfidfVectorizer(max_features=100, stop_words='english'), 'name_str')
        ]
    )
    
    threshold = 0.30

    # Calculate 5-Fold Stratified CV F1 Score
    print("Evaluating 5-Fold Cross-Validation F1 score...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        fold_pipe = Pipeline([
            ('prep', preprocessor),
            ('model', DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, criterion='entropy', ccp_alpha=0.002, random_state=42))
        ])
        fold_pipe.fit(X_tr, y_tr)
        oof_probs[val_idx] = fold_pipe.predict_proba(X_va)[:, 1]
    
    cv_f1 = f1_score(y_train, (oof_probs >= threshold).astype(int))
    print(f"Cross-Validation F1 Score (threshold={threshold}): {cv_f1:.4f}")
    
    # Pipeline with tuned Decision Tree
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, criterion='entropy', ccp_alpha=0.002, random_state=42))
    ])
    
    print("Training Decision Tree model on full training set...")
    pipeline.fit(X_train, y_train)
    
    print("Predicting on new_projects.csv...")
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    output_df = pd.DataFrame({
        'id': test_df.id,
        'state_ind_pred': preds
    })
    
    output_csv = '123456789.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    print("Distribution of predictions:")
    print(output_df.state_ind_pred.value_counts())

    # Create Excel file
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Group Members"
    ws.append(["ID", "Name"])
    ws.append(["123456789", "Student Name"])
    excel_file = "123456789.xlsx"
    wb.save(excel_file)
    print(f"Excel file saved to {excel_file}")

if __name__ == '__main__':
    main()

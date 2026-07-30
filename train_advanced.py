import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('kickstarter_projects.csv')

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
    
    # Cyclic date features (month, hr, weekday)
    for col, max_val in [('launched_at_month', 12), ('launched_at_hr', 24), ('deadline_month', 12), ('deadline_hr', 24)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
    return df

df_eng = engineer_features(train_df)

drop_cols = ['id', 'name', 'slug', 'source_url', 'state_ind', 'name_str']
feature_cols = [c for c in df_eng.columns if c not in drop_cols]

X = df_eng[feature_cols]
y = df_eng.state_ind

cat_cols = ['country', 'currency', 'category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(model_builder):
    oof_probs = np.zeros(len(y))
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model_builder())
        ])
        
        pipe.fit(X_tr, y_tr)
        
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs
        else:
            probs = pipe.predict(X_va)
            oof_probs[val_idx] = probs

    best_thresh = 0.5
    best_f1 = f1_score(y, (oof_probs >= 0.5).astype(int))
    
    for t in np.arange(0.1, 0.9, 0.01):
        score = f1_score(y, (oof_probs >= t).astype(int))
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    return best_f1, best_thresh

models = {
    "Logistic Regression (C=0.1, balanced)": lambda: LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
    "Logistic Regression (C=1.0, balanced)": lambda: LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
    "Logistic Regression (C=10, balanced)": lambda: LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42),
    "Logistic Regression (C=1.0, default weight)": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Decision Tree (max_depth=8)": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
    "Decision Tree (max_depth=12)": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
    "KNN (k=25)": lambda: KNeighborsClassifier(n_neighbors=25, weights='distance'),
    "KNN (k=51)": lambda: KNeighborsClassifier(n_neighbors=51, weights='distance')
}

if __name__ == '__main__':
    for name, builder in models.items():
        f1, thresh = evaluate(builder)
        print(f"{name:45s} -> Best F1: {f1:.4f} @ threshold {thresh:.2f}")

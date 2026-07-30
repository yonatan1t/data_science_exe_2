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
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('kickstarter_projects.csv')

def preprocess_df(df):
    df = df.copy()
    df.currency_rate = df.currency_rate.fillna(1.0)
    df['goal_usd'] = df.goal * df.currency_rate
    df['log_goal_usd'] = np.log1p(df.goal_usd)
    df['log_goal'] = np.log1p(df.goal)
    
    df.staff_pick = df.staff_pick.astype(int)
    df.category = df.category.fillna('Missing')
    df.name_len = df.name_len.fillna(df.name_len.median())
    df.name_len_clean = df.name_len_clean.fillna(df.name_len_clean.median())
    
    return df

train_df = preprocess_df(train_df)

drop_cols = ['id', 'name', 'slug', 'source_url', 'state_ind']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols]
y = train_df.state_ind

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

def eval_model(model_builder):
    oof_preds = np.zeros(len(y))
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
            oof_preds[val_idx] = (probs >= 0.5).astype(int)
        else:
            preds = pipe.predict(X_va)
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = preds
            
    base_f1 = f1_score(y, oof_preds)
    base_prec = precision_score(y, oof_preds)
    base_rec = recall_score(y, oof_preds)
    
    best_thresh = 0.5
    best_f1 = base_f1
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (oof_probs >= t).astype(int)
        score = f1_score(y, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
                
    return {
        'base_f1': base_f1,
        'base_prec': base_prec,
        'base_rec': base_rec,
        'best_f1': best_f1,
        'best_thresh': best_thresh
    }

models = {
    "Logistic Regression (default)": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Logistic Regression (C=0.1)": lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=1000, random_state=42),
    "Logistic Regression (class_weight=balanced)": lambda: LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Decision Tree (max_depth=5)": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "Decision Tree (max_depth=10)": lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
    "Decision Tree (max_depth=15)": lambda: DecisionTreeClassifier(max_depth=15, random_state=42),
    "Decision Tree (min_samples_leaf=20)": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=42),
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "KNN (k=15)": lambda: KNeighborsClassifier(n_neighbors=15),
    "KNN (k=31)": lambda: KNeighborsClassifier(n_neighbors=31),
    "KNN (k=51)": lambda: KNeighborsClassifier(n_neighbors=51),
}

if __name__ == '__main__':
    for name, builder in models.items():
        res = eval_model(builder)
        print(f"--- {name} ---")
        print(f"Base F1 (t=0.5): {res['base_f1']:.4f} (Prec: {res['base_prec']:.4f}, Rec: {res['base_rec']:.4f})")
        print(f"Best F1: {res['best_f1']:.4f} at Threshold: {res['best_thresh']:.2f}")
        print()

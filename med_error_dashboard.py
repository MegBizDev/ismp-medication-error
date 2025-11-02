import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

# --- AUTO-DETECT uploaded CSV ---
import glob
import os
csv_files = glob.glob("*.csv")
if not csv_files:
    st.error("No CSV file found. Please upload your dataset in Colab first.")
    st.stop()
DATA_PATH = csv_files[0]

TARGET_COL = "consensus_flag"
RANDOM_STATE = 42

st.set_page_config(page_title="Medication Error Risk Dashboard", layout="wide")

# --- Styling ---
st.markdown("""
<style>
:root {
  --primary:#0b69ff;
  --bg:#ffffff;
}
.stApp {background:var(--bg);}
.card {
  background:white;
  border-radius:10px;
  padding:15px;
  box-shadow:0 3px 10px rgba(11,105,255,0.07);
  margin-bottom:12px;
  transition: all .3s ease;
}
.card:hover {transform:translateY(-3px);}
.card h4 {margin:0;color:#07336a;}
.small {font-size:13px;color:#50657a;}
</style>
""", unsafe_allow_html=True)

# --- Load & preprocess ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def preprocess(df):
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({'Y':1,'N':0})
    if 'medication_name' in df.columns:
        df['is_high_alert'] = df['medication_name'].fillna('').str.lower().apply(
            lambda s: int(any(x in s for x in ['insulin','heparin','warfarin','potassium','morphine','fentanyl'])))
    else:
        df['medication_name'] = "Unknown"
        df['is_high_alert'] = 0
    return df

df = load_data(DATA_PATH)
df = preprocess(df)

st.title("Medication Error Risk — ISMP Dashboard")

if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' not found.")
    st.stop()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])

num_cols = [c for c in ['dose_ordered','dose_administered','patient_age'] if c in df.columns]
cat_cols = [c for c in ['medication_type','route_ordered'] if c in df.columns]

tfidf_cols = [c for c in ['ISMP_prevention_guideline','lab_values_nearby','vitals_nearby'] if c in df.columns]
texts = df[tfidf_cols].fillna('').astype(str).agg(' '.join, axis=1) if tfidf_cols else None
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf.fit_transform(texts) if texts is not None else None

transformers = []
if num_cols:
    transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols))
if cat_cols:
    transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols))

ct = ColumnTransformer(transformers)
X_numcat = ct.fit_transform(df)
X = X_numcat if X_text is None else np.hstack([X_numcat.toarray(), X_text.toarray()])
y = df[TARGET_COL].fillna(0).astype(int)

rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
rf.fit(X, y)

y_pred = rf.predict(X)
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)

st.metric("Model Accuracy", f"{acc*100:.2f}%")
st.metric("F1 Score", f"{f1:.3f}")

st.subheader("Error Distribution")
pie_df = df[TARGET_COL].value_counts().rename_axis('error').reset_index(name='count')
pie_df['error'] = pie_df['error'].map({1:'Error',0:'No Error'})
fig_pie = px.pie(pie_df, names='error', values='count', title='Error vs No Error')
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Drug Details")
for drug, sub in df.groupby('medication_name'):
    st.markdown(f"""
    <div class="card">
      <h4>{drug}</h4>
      <div class="small">ISMP High-Alert: <b>{'Yes' if sub['is_high_alert'].any() else 'No'}</b></div>
      <div class="small">Medication Type: <b>{sub['medication_type'].dropna().unique()[0] if 'medication_type' in sub.columns and len(sub['medication_type'].dropna().unique())>0 else '—'}</b></div>
      <div style='margin-top:5px;'><b>ISMP Integrated Minimization:</b>
        <ul>
          <li>Double-check all high-alert meds.</li>
          <li>Confirm units and routes before administration.</li>
          <li>Apply tall-man lettering for look-alike drugs.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="AutoML IDS Dashboard", page_icon="🛡️", layout="wide")

API_URL = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")

st.title("🛡️ AutoML Intrusion Detection Dashboard")
st.caption("Upload network flow data → select a row → get intrusion probability + predicted label")

# Health check
if st.sidebar.button("Check API Health"):
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        st.sidebar.success(r.json())
    except Exception as e:
        st.sidebar.error(f"API not reachable: {e}")

st.divider()

# Upload CSV
uploaded = st.file_uploader("Upload CSV file (raw network features)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("🎯 Select a row for prediction")
    row_idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0, step=1)

    record = df.iloc[int(row_idx)].to_dict()

    # show selected record
    with st.expander("Selected row data (JSON)"):
        st.json(record)

    col1, col2, col3 = st.columns([1,1,2])

    if col1.button("🔮 Predict"):
        try:
            payload = {"data": record}
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            out = resp.json()

            if "error" in out:
                st.error(out["error"])
            else:
                prob = float(out["prob_attack"])
                thr = float(out["threshold"])
                label = int(out["pred_label"])

                col2.metric("Probability (Attack)", f"{prob:.4f}")
                col3.metric("Predicted Label", "ATTACK 🚨" if label == 1 else "NORMAL ✅")

                st.progress(min(max(prob, 0.0), 1.0))
                st.caption(f"Decision threshold used: **{thr:.4f}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()

    # Batch prediction
    st.subheader("⚡ Batch Prediction (entire CSV)")
    if st.button("Run batch predictions"):
        preds = []
        for i in range(len(df)):
            payload = {"data": df.iloc[i].to_dict()}
            resp = requests.post(f"{API_URL}/predict", json=payload)
            out = resp.json()
            if "error" in out:
                preds.append({"prob_attack": None, "pred_label": None})
            else:
                preds.append({"prob_attack": out["prob_attack"], "pred_label": out["pred_label"]})

        pred_df = pd.DataFrame(preds)
        out_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

        st.success("✅ Batch predictions complete")
        st.dataframe(out_df.head(20), use_container_width=True)

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions CSV", csv_bytes, file_name="ids_predictions.csv", mime="text/csv")

else:
    st.info("Upload a CSV file to start.")

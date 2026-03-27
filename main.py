# =========================================
# STREAMLIT HMM + MONTE CARLO DASHBOARD
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 NIFTY 50 HMM + Monte Carlo Dashboard")

# =========================================
# REFRESH BUTTON
# =========================================
if "run_model" not in st.session_state:
    st.session_state.run_model = False

if st.button("🔄 Refresh / Recalculate"):
    st.session_state.run_model = True

if not st.session_state.run_model:
    st.info("Click 'Refresh / Recalculate' to run the model.")
    st.stop()

# =========================================
# PARAMETERS
# =========================================
TICKER = "^NSEI"
N_STATES = 3
SIMULATIONS = 20000
STEPS = 1
LOOKBACK_YEARS = 5

with st.spinner("Running full model..."):

    # =========================================
    # DATE HANDLING (WITH CHECKBOX)
    # =========================================
    today = datetime.today()

    include_today = st.checkbox(
        "📅 Include today's data (may be incomplete)",
        value=False
    )

    if include_today:
        end_date = today
        st.warning("⚠️ Today's data may be incomplete and can distort results.")
    else:
        end_date = today - timedelta(days=1)

    start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)

    st.write(f"📅 Data Range: {start_date.date()} → {end_date.date()}")

    # =========================================
    # FETCH DATA
    # =========================================
    try:
        df = yf.download(
            TICKER,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True
        )
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")
        st.stop()

    if df.empty or len(df) < 100:
        st.error("❌ Not enough data fetched.")
        st.stop()

    df = df[['Close']].dropna()

    # =========================================
    # FEATURES
    # =========================================
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol'] = df['returns'].rolling(10).std()
    df['mom'] = df['returns'].rolling(5).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if df.empty:
        st.error("❌ Data empty after preprocessing.")
        st.stop()

    X = df[['returns', 'vol', 'mom']].values

    if len(X) < 50:
        st.error("❌ Not enough data for HMM.")
        st.stop()

    # =========================================
    # TRAIN HMM
    # =========================================
    try:
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        model.fit(X)
    except Exception as e:
        st.error(f"❌ HMM training failed: {e}")
        st.write("Shape of X:", X.shape)
        st.write("Sample X:", X[:5])
        st.stop()

    hidden_states = model.predict(X)
    df['state'] = hidden_states

    # =========================================
    # REGIME PARAMETERS
    # =========================================
    means = model.means_[:, 0]
    variances = model.covars_[:, 0, 0]
    stds = np.sqrt(variances)
    trans_mat = model.transmat_

    sorted_idx = np.argsort(means)
    bear_regime = sorted_idx[0]
    neutral_regime = sorted_idx[1]
    bull_regime = sorted_idx[2]

    current_state = df['state'].iloc[-1]

    # =========================================
    # MONTE CARLO
    # =========================================
    sim_returns = []

    for _ in range(SIMULATIONS):
        state = current_state
        total_return = 0

        for _ in range(STEPS):
            r = np.random.normal(means[state], stds[state])
            total_return += r
            state = np.random.choice(range(N_STATES), p=trans_mat[state])

        sim_returns.append(total_return)

    sim_returns = np.array(sim_returns)

    # =========================================
    # CLASSIFICATION
    # =========================================
    volatility = df['returns'].std()

    UP_TH = volatility * 0.5
    DOWN_TH = -volatility * 0.5

    up = np.mean(sim_returns > UP_TH)
    neutral_up = np.mean((sim_returns > 0) & (sim_returns <= UP_TH))
    neutral_down = np.mean((sim_returns < 0) & (sim_returns >= DOWN_TH))
    down = np.mean(sim_returns < DOWN_TH)

    # =========================================
    # ARGMAX SIGNAL (FIXED)
    # =========================================
    prob_dict = {
        "UP": up,
        "NEUTRAL_UP": neutral_up,
        "NEUTRAL_DOWN": neutral_down,
        "DOWN": down
    }

    signal_state = max(prob_dict, key=prob_dict.get)
    signal = signal_state

# =========================================
# UI
# =========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📌 Current Regime")
    if current_state == bull_regime:
        st.success("BULL")
    elif current_state == bear_regime:
        st.error("BEAR")
    else:
        st.warning("SIDEWAYS")

with col2:
    st.subheader("📊 Signal (Most Probable)")
    st.info(signal)

with col3:
    st.subheader("📈 Expected Return")
    st.metric("Log Return", round(np.mean(sim_returns), 6))

# =========================================
# PROBABILITIES
# =========================================
st.subheader("🔮 Next Session Probabilities")

prob_df = pd.DataFrame({
    "State": ["UP", "NEUTRAL_UP", "NEUTRAL_DOWN", "DOWN"],
    "Probability": [up, neutral_up, neutral_down, down]
})

st.bar_chart(prob_df.set_index("State"))

# =========================================
# REGIME PLOT
# =========================================
st.subheader("📉 Regime Detection")

fig, ax = plt.subplots()

for i in range(N_STATES):
    ax.plot(df[df['state'] == i].index,
            df[df['state'] == i]['Close'],
            '.', label=f'Regime {i}')

ax.plot(df['Close'], alpha=0.3)
ax.legend()

st.pyplot(fig)

# =========================================
# TRANSITION MATRIX
# =========================================
st.subheader("🔁 Transition Matrix")
st.dataframe(pd.DataFrame(trans_mat))

# =========================================
# DEBUG PANEL
# =========================================
with st.expander("⚙️ Debug Info"):
    st.write("Data shape:", df.shape)
    st.write("Feature shape:", X.shape)
    st.write("Means:", means)
    st.write("Probabilities:", prob_dict)

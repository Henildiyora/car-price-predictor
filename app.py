import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(page_title="Car Insights", layout="wide", page_icon="⚡")

# --- Color Palette ---
COLOR_PALETTE = {
    "bg": "#f0f5f9", "primary": "#0b5ed7", "secondary": "#5c6c7d",
    "accent": "#ff7f0e", "success": "#198754", "card_bg": "#ffffff",
    "text": "#333333", "subtle_text": "#6c757d", "border": "#dee2e6",
    "plot_main": "#0b5ed7", "plot_accent": "#ff7f0e",
}

# --- Custom CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    body {{
        font-family: 'Roboto', sans-serif;
        color: {COLOR_PALETTE['text']};
    }}
    .main {{
        background-color: {COLOR_PALETTE['bg']};
        padding: 25px;
        border-radius: 10px;
    }}
    .app-header {{
        color: {COLOR_PALETTE['primary']};
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
    }}
    .app-subheader {{
        color: {COLOR_PALETTE['secondary']};
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 300;
    }}
    h2 {{
        color: {COLOR_PALETTE['primary']};
        border-bottom: 3px solid {COLOR_PALETTE['accent']};
        padding-bottom: 8px;
        margin-top: 30px;
        margin-bottom: 20px;
        font-weight: 700;
    }}
    h3 {{
        color: {COLOR_PALETTE['secondary']};
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 700;
    }}
    .stButton>button {{
        background-color: {COLOR_PALETTE['accent']};
        color: white;
        border-radius: 25px;
        padding: 12px 28px;
        border: none;
        font-size: 1.1em;
        font-weight: 700;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-top: 10px;
    }}
    .stButton>button:hover {{
        background-color: #e6730d;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }}
    .stButton>button:disabled {{
        background-color: {COLOR_PALETTE['subtle_text']};
        opacity: 0.7;
    }}
    .stSelectbox > div[data-baseweb="select"] > div,
    .stNumberInput > div > div > input,
    .stSlider > div[data-baseweb="slider"] {{
        background-color: {COLOR_PALETTE['card_bg']};
        border-radius: 8px;
        border: 1px solid {COLOR_PALETTE['border']};
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.075);
    }}
    .stNumberInput input {{ padding: 10px; }}
    .stSelectbox > label, .stNumberInput > label, .stSlider > label {{
         font-weight: 400;
         color: {COLOR_PALETTE['secondary']};
         padding-bottom: 5px;
    }}
    .success-message {{
        font-size: 1.6em;
        color: {COLOR_PALETTE['success']};
        font-weight: 700;
        background-color: #e6f9ef;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid {COLOR_PALETTE['success']};
        border-left: 6px solid {COLOR_PALETTE['success']};
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .stExpander {{
        border: 1px solid {COLOR_PALETTE['border']};
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: {COLOR_PALETTE['card_bg']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .stExpander header {{
         font-size: 1.1em;
         font-weight: 700;
         color: {COLOR_PALETTE['primary']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
		gap: 30px;
        border-bottom: 2px solid {COLOR_PALETTE['border']};
        padding-bottom: 5px;
	}}
	.stTabs [data-baseweb="tab"] {{
		height: 55px;
        white-space: pre-wrap;
		background-color: transparent;
		border-radius: 8px 8px 0 0;
		gap: 5px;
		padding: 10px 20px;
        font-weight: 700;
        font-size: 1.15em;
        color: {COLOR_PALETTE['secondary']};
        border: none;
        border-bottom: 4px solid transparent;
        transition: all 0.3s ease;
	}}
	.stTabs [aria-selected="true"] {{
  		background-color: transparent;
        color: {COLOR_PALETTE['primary']};
        border-bottom: 4px solid {COLOR_PALETTE['primary']};
	}}
</style>
""", unsafe_allow_html=True)

# --- File Paths ---
BANNER_IMAGE_PATH = 'banner.jpg'
DATA_PATH = 'data/processed/car_price_selected_feature_original.csv'
MODEL_PATH = 'notebooks/src/saved_models/StackingRegressor_model.joblib'
SCALER_PATH = 'notebooks/src/min_max_scaler.pkl'

@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: Data file not found at `{filepath}`")
        return None
    try:
        # Debug: Preview the file content with multiple encoding attempts
        encodings = ['utf-8', 'latin1', 'utf-16']
        content_preview = None
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content_preview = f.read(500)  # Read first 500 characters
                # st.write(f"File content preview (encoding: {encoding}):", content_preview)
                break  # If successful, stop trying other encodings
            except UnicodeDecodeError:
                st.warning(f"Failed to read file with {encoding} encoding, trying next...")
                continue
        
        if content_preview is None:
            st.error("Unable to read file with any supported encoding.")
            return None
        
        # Load data with the successful encoding
        chunk_size = 5000
        chunks = pd.read_csv(filepath, encoding=encoding, sep=',', chunksize=chunk_size)
        df = pd.concat(chunks, ignore_index=True)
        
        required_cols = ['make', 'model', 'condition', 'year', 'odometer', 'mmr', 'sellingprice', 'state']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dataset missing required columns: {required_cols}")
            return None
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if 'vehicle_age' not in df.columns:
            df['vehicle_age'] = datetime.now().year - df['year']
        if 'mileage_per_year' not in df.columns:
            df['mileage_per_year'] = np.where(df['vehicle_age'] > 0, df['odometer'] / df['vehicle_age'], 0)
        if 'price_per_mile' not in df.columns:
            df['price_per_mile'] = np.where(df['odometer'] > 0, df['mmr'] / df['odometer'], 0)
        derived_cols = ['vehicle_age', 'mileage_per_year', 'price_per_mile']
        df.dropna(subset=required_cols + derived_cols, inplace=True)
        for col in ['year', 'condition', 'odometer', 'mmr', 'sellingprice', 'vehicle_age', 'mileage_per_year', 'price_per_mile']:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric: {e}")
                df[col] = 0
        df['year'] = df['year'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



@st.cache_resource
def load_model(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: Model file not found at `{filepath}`")
        return None
    try:
        return joblib.load(filepath)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: Scaler file not found at `{filepath}`")
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

def preprocess_input(input_data, label_encoders, scaler, all_model_features, numerical_features_scaled, numerical_features_unscaled, categorical_features):
    input_df = pd.DataFrame([input_data])
    current_year = datetime.now().year
    input_df['vehicle_age'] = current_year - input_df['year']
    input_df['mileage_per_year'] = input_df['odometer'].astype(float) / input_df['vehicle_age'].astype(float) if input_df['vehicle_age'].iloc[0] > 0 else 0
    input_df['price_per_mile'] = input_df['mmr'].astype(float) / input_df['odometer'].astype(float) if input_df['odometer'].iloc[0] > 0 else 0
    input_df.replace([np.inf, -np.inf], 0, inplace=True)
    for feature in categorical_features:
        if feature in label_encoders:
             classes = set(label_encoders[feature].classes_)
             input_df[feature] = input_df[feature].apply(lambda x: label_encoders[feature].transform([x])[0] if x in classes else -1)
        else:
            input_df[feature] = -1
    try:
        input_scaled_df = pd.DataFrame(scaler.transform(input_df[numerical_features_scaled]), columns=numerical_features_scaled)
    except Exception as e:
        st.error(f"Scaling error on input: {e}")
        return None
    final_input_df = pd.concat([input_scaled_df, input_df[numerical_features_unscaled].reset_index(drop=True), input_df[categorical_features].reset_index(drop=True)], axis=1)
    final_input_df = final_input_df.reindex(columns=all_model_features, fill_value=0)
    return final_input_df.fillna(0)

# --- Load Data, Model, and Scaler ---
df_original = load_data(DATA_PATH)
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# --- App Start ---
if os.path.exists(BANNER_IMAGE_PATH):
    try:
        st.image(BANNER_IMAGE_PATH, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load banner image from '{BANNER_IMAGE_PATH}': {e}")
else:
    st.error(f"Banner image not found at the specified path: `{BANNER_IMAGE_PATH}`. Please check the path.")
    st.markdown("<div style='height: 150px; background-color: #dcdcdc; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: 20px;'><span style='color: #888;'>[Banner Image Area]</span></div>", unsafe_allow_html=True)

st.markdown('<div class="app-header">Car Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subheader">Your enhanced tool for intelligent car price prediction and market analysis.</div>', unsafe_allow_html=True)

if df_original is None or model is None or scaler is None:
    st.error("Application cannot start. Essential files failed to load.")
    st.stop()

# --- Define Feature Sets ---
all_model_features = ['mmr', 'price_per_mile', 'odometer', 'condition', 'mileage_per_year', 'model', 'year', 'vehicle_age', 'make']
categorical_features = ['make', 'model']
numerical_features_scaled = ['condition', 'odometer', 'mmr', 'mileage_per_year', 'price_per_mile']
numerical_features_unscaled = ['year', 'vehicle_age']

df = df_original.copy()
unique_vals = {}
label_encoders = {}
try:
    for feature in categorical_features:
        valid_values = df[feature].dropna().astype(str).unique()
        unique_vals[feature] = sorted(valid_values)
        le = LabelEncoder()
        le.fit(unique_vals[feature])
        label_encoders[feature] = le
    unique_vals['condition'] = sorted(df['condition'].dropna().unique())
    unique_vals['state'] = sorted(df['state'].dropna().unique())
except KeyError as e:
    st.error(f"Config Error: Column '{e}' missing.")
    st.stop()
except Exception as e:
    st.error(f"Init Error: {e}")
    st.stop()

tab_titles = ["💰 Price Predictor", "🔍 Price Range Explorer", "📊 Market Insights"]
tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:
    st.header("Estimate Your Car's Value")
    with st.expander("Enter Car Details Here", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            make = st.selectbox('Select Make', unique_vals['make'], key='pred_make', index=0)
            filtered_models = sorted(df[df['make'] == make]['model'].dropna().astype(str).unique())
            if not filtered_models:
                 st.warning(f"No models listed for make '{make}'.")
                 model_selected = None
            else:
                default_model_index = 0
                if filtered_models:
                    try:
                        common_models = ['Sedan', 'Base', 'LX', 'EX', 'LE', 'XLE']
                        for common in common_models:
                           if common in filtered_models:
                                default_model_index = filtered_models.index(common)
                                break
                    except ValueError:
                        default_model_index = 0
                model_selected = st.selectbox('Select Model', filtered_models, key='pred_model', index=default_model_index)
            min_cond = float(df['condition'].min()) if not df['condition'].empty else 1.0
            max_cond = float(df['condition'].max()) if not df['condition'].empty else 5.0
            median_cond = float(df['condition'].median()) if not df['condition'].empty else 3.0
            if min_cond >= max_cond:
                max_cond = min_cond + 4.0
            condition = st.slider('Overall Condition', min_value=min_cond, max_value=max_cond,
                                  value=median_cond, step=0.1, key='pred_condition',
                                  help=f"Rate condition from {min_cond:.1f} (poor) to {max_cond:.1f} (excellent)")
        with col2:
            current_year = datetime.now().year
            min_year = int(df['year'].min()) if not df['year'].empty else 1990
            max_year = current_year
            mode_year = int(df['year'].mode()[0]) if not df['year'].empty else current_year - 5
            year = st.number_input('Manufacture Year', min_value=min_year, max_value=max_year,
                                   value=max(max_year - 5, min_year), key='pred_year')
            median_odo = int(df['odometer'].median()) if not df['odometer'].empty else 50000
            odometer = st.number_input('Odometer Reading (miles)', min_value=0, value=median_odo,
                                       step=500, key='pred_odometer')
            median_mmr = float(df['mmr'].median()) if not df['mmr'].empty else 15000.0
            mmr = st.number_input('Estimated Market Value (MMR $)', min_value=0.0, value=median_mmr,
                                  step=100.0, format="%.0f", key='pred_mmr',
                                  help="Manheim Market Report value, if known (optional).")
        validation_passed = True
        if year > current_year:
            st.error("Year cannot be in the future!")
            validation_passed = False
        if odometer < 0:
            st.error("Odometer cannot be negative!")
            validation_passed = False
        if mmr < 0:
            st.error("MMR cannot be negative!")
            validation_passed = False
        if make and not model_selected and filtered_models:
            st.warning("Please select a valid model.")
            validation_passed = False
    if st.button('Predict Selling Price', key='predict_button', disabled=not validation_passed, use_container_width=True):
        if validation_passed and model_selected:
            with st.spinner('Performing Analysis... Please Wait...'):
                input_dict = {'make': make, 'model': model_selected, 'condition': condition,
                              'year': year, 'odometer': odometer, 'mmr': mmr}
                final_input = preprocess_input(
                    input_dict, label_encoders, scaler, all_model_features,
                    numerical_features_scaled, numerical_features_unscaled, categorical_features
                )
                if final_input is not None:
                    if (final_input[categorical_features] == -1).any().any():
                        st.warning("Heads up: Some input details might be uncommon.")
                    try:
                        pred_log_price = model.predict(final_input)[0]
                        pred_price = pred_log_price * 100000
                        formatted_price = f"${pred_price:,.0f}"
                        st.markdown(f'<div class="success-message">Estimated Selling Price: {formatted_price}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.error(traceback.format_exc())

with tab2:
    st.header("Explore Market by Price")
    st.write("Filter cars within your desired budget.")
    min_price = int(df['sellingprice'].min()) if not df['sellingprice'].empty else 0
    max_price = int(df['sellingprice'].max()) if not df['sellingprice'].empty else 100000
    if min_price >= max_price:
        max_price = min_price + 10000
    default_min = max(min_price, int(df['sellingprice'].quantile(0.10)) if not df['sellingprice'].empty else min_price)
    default_max = min(max_price, int(df['sellingprice'].quantile(0.90)) if not df['sellingprice'].empty else max_price)
    if default_min >= default_max:
        default_min = min_price
        default_max = max_price
    price_range = st.slider("Select Price Range ($)", min_value=min_price, max_value=max_price,
                            value=(default_min, default_max), step=500, format="$%d")
    min_selected, max_selected = price_range
    filtered_df = df[(df['sellingprice'] >= min_selected) & (df['sellingprice'] <= max_selected)]
    st.success(f"Found **{len(filtered_df):,}** cars between **${min_selected:,}** and **${max_selected:,}**")
    if not filtered_df.empty:
        st.subheader("📊 Insights for Selected Range")
        col1, col2 = st.columns(2)
        with col1:
            if 'make' in filtered_df.columns:
                try:
                    fig_make_dist = px.histogram(filtered_df, x='make', title=f'Cars by Make (${min_selected:,} - ${max_selected:,})',
                                                 color_discrete_sequence=[COLOR_PALETTE['plot_main']])
                    fig_make_dist.update_layout(xaxis={'categoryorder':'total descending'}, title_x=0.5)
                    st.plotly_chart(fig_make_dist, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot makes: {e}")
            else:
                st.warning("Column 'make' not found for plotting.")
        with col2:
             if 'odometer' in filtered_df.columns and 'sellingprice' in filtered_df.columns:
                 try:
                     fig_price_odo = px.scatter(filtered_df, x='odometer', y='sellingprice',
                                              color_discrete_sequence=[COLOR_PALETTE['plot_accent']],
                                              title=f'Price vs. Mileage (${min_selected:,} - ${max_selected:,})',
                                              labels={'odometer': 'Odometer (miles)', 'sellingprice': 'Selling Price ($)'})
                     fig_price_odo.update_layout(title_x=0.5)
                     st.plotly_chart(fig_price_odo, use_container_width=True)
                 except Exception as e:
                    st.warning(f"Could not plot price vs odometer: {e}")
             else:
                 st.warning("Columns 'odometer' or 'sellingprice' not found for plotting.")
        with st.expander("View Filtered Car Data"):
             display_cols = [col for col in ['make', 'model', 'year', 'condition', 'odometer', 'sellingprice', 'state'] if col in filtered_df.columns]
             if display_cols:
                 st.dataframe(filtered_df[display_cols].reset_index(drop=True))
             else:
                 st.warning("No standard columns found to display in the filtered data table.")
    else:
        st.info("No cars found in this price range. Try adjusting the slider.")

with tab3:
    st.header("📈 Overall Market Trends")
    st.write("Visual analysis of the entire dataset.")
    col1, col2 = st.columns(2)
    with col1:
        if 'sellingprice' in df.columns:
            st.subheader("Price Distribution")
            try:
                fig_price_hist = px.histogram(df, x='sellingprice', nbins=60, title="Overall Selling Price Distribution", color_discrete_sequence=[COLOR_PALETTE['plot_main']])
                fig_price_hist.update_layout(title_x=0.5)
                st.plotly_chart(fig_price_hist, use_container_width=True)
            except Exception as e:
                st.warning(f"-odd Error (Price Dist): {e}")
        else:
            st.warning("'sellingprice' column needed for Price Distribution.")
        if 'make' in df.columns and 'sellingprice' in df.columns:
            st.subheader("Top Makes by Avg. Price")
            try:
                avg_price_make = df.groupby('make')['sellingprice'].mean().reset_index().sort_values('sellingprice', ascending=False)
                fig_avg_price = px.bar(avg_price_make.head(15), x='make', y='sellingprice', title="Top 15 Makes by Average Selling Price", text='sellingprice', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_avg_price.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_avg_price.update_layout(xaxis={'categoryorder':'total descending'}, title_x=0.5)
                st.plotly_chart(fig_avg_price, use_container_width=True)
            except Exception as e:
                st.warning(f"Plotting Error (Avg Price by Make): {e}")
        else:
            st.warning("'make' and 'sellingprice' columns needed for Avg Price by Make.")
    with col2:
        if 'odometer' in df.columns and 'sellingprice' in df.columns:
            st.subheader("Price vs. Mileage Density")
            try:
                sample_size = min(10000, len(df))
                if sample_size > 0:
                    fig_price_mileage = px.density_heatmap(df.sample(sample_size), x='odometer', y='sellingprice', nbinsx=30, nbinsy=30, 
                                                        title="Density: Selling Price vs. Odometer", 
                                                        labels={'odometer': 'Odometer (miles)', 'sellingprice': 'Selling Price ($)'}, 
                                                        color_continuous_scale="Blues")
                    fig_price_mileage.update_layout(title_x=0.5)
                    st.plotly_chart(fig_price_mileage, use_container_width=True)
                else:
                    st.info("Not enough data to plot Price vs Mileage Density.")
            except Exception as e:
                st.warning(f"Plotting Error (Price vs Mileage): {e}")
        else:
            st.warning("'odometer' and 'sellingprice' columns needed for Price vs Mileage.")
        if 'year' in df.columns and 'sellingprice' in df.columns:
             st.subheader("Price Trend by Year")
             try:
                 avg_price_year = df.groupby('year')['sellingprice'].mean().reset_index()
                 fig_price_year = px.line(avg_price_year, x='year', y='sellingprice', markers=True, title="Average Selling Price by Manufacture Year", color_discrete_sequence=[COLOR_PALETTE['plot_accent']])
                 fig_price_year.update_layout(title_x=0.5)
                 st.plotly_chart(fig_price_year, use_container_width=True)
             except Exception as e:
                 st.warning(f"Plotting Error (Price Trend by Year): {e}")
        else:
            st.warning("'year' and 'sellingprice' columns needed for Price Trend.")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {COLOR_PALETTE['subtle_text']}; font-size: 0.9em;">
    Car Insights Pro+ | Powered by Streamlit & Scikit-learn | Data as of {datetime.now().year}
</div>
""", unsafe_allow_html=True)
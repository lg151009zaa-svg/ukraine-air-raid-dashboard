import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Configuration for Streamlit
st.set_page_config(page_title="Ukraine Air Raid Analytics", layout="wide")

# ==========================================
# STEP 1: Data Automation & Error Handling
# ==========================================
URL = 'https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_uk.csv'

@st.cache_data(ttl=3600)  # Cache data for 1 hour to optimize performance
def load_and_clean_data(url):
    df = pd.read_csv(url)
    
    # Handle column names dynamically
    if 'region_title' in df.columns:
        df = df.rename(columns={'region_title': 'region'})
        
    # Convert to datetime, localize to UTC, then convert to Europe/Kyiv timezone
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True).dt.tz_convert('Europe/Kyiv').dt.tz_localize(None)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True).dt.tz_convert('Europe/Kyiv').dt.tz_localize(None)
    
    # Calculate duration
    df['duration_min'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60
    
    # Filter anomalies (1 minute to 12 hours)
    df_clean = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 720)].copy()
    
    # Pre-calculate Global Time Gaps and local hour components
    df_clean = df_clean.sort_values(by=['region', 'started_at'])
    df_clean['hours_since_last_alert'] = df_clean.groupby('region')['started_at'].diff().dt.total_seconds() / 3600
    df_clean['hour'] = df_clean['started_at'].dt.hour
    
    return df_clean

try:
    df_clean = load_and_clean_data(URL)
except Exception as e:
    st.error(f"Critical Error: Unable to fetch or process live data from the repository. Details: {e}")
    st.stop()

# ==========================================
# STEP 2: Pre-calculating Global Aggregations
# ==========================================
unique_regions = sorted(df_clean['region'].unique())

# Regional frequency stats
region_stats = df_clean.groupby('region').size().reset_index(name='alert_count').sort_values(by='alert_count', ascending=False)

# Gap rankings via loop calculation
gap_records = []
for reg in unique_regions:
    reg_data = df_clean[df_clean['region'] == reg]
    mean_gap = reg_data['hours_since_last_alert'].mean()
    gap_records.append({'region': reg, 'avg_quiet_time_hours': mean_gap})
gap_summary_df = pd.DataFrame(gap_records).sort_values(by='avg_quiet_time_hours', ascending=True).reset_index(drop=True)

# Generate formatted labels for all 24 hours
hour_labels = [f"{h:02d}:00" for h in range(24)]

# ==========================================
# STEP 3: Automated Executive Summary Function
# ==========================================
def generate_executive_summary(df_metrics, regional_df, gap_df):
    total_alerts = len(df_metrics)
    mean_dur = df_metrics['duration_min'].mean()
    median_dur = df_metrics['duration_min'].median()
    
    top_region = regional_df.iloc[0]['region']
    top_region_count = regional_df.iloc[0]['alert_count']
    
    shortest_gap_region = gap_df.iloc[0]['region']
    shortest_gap_val = gap_df.iloc[0]['avg_quiet_time_hours']
    
    summary_text = f"""
    ### 📊 Automated Executive Summary
    This analytical report evaluates air raid alert patterns across Ukraine using processed historical entries. 
    To date, a total of **{total_alerts:,}** valid air raid alerts have been scrutinized. 
    The nationwide average alert duration stands at **{mean_dur:.1f} minutes**, with a standard median of **{median_dur:.1f} minutes**.
    
    **Key Regional Insights (Local Kyiv Time applied):**
    * **Highest Target Frequency:** **{top_region}** exhibits the highest frequency of hostile activity, registering **{top_region_count:,}** individual alerts.
    * **Highest Attack Intensity:** Time gap analysis reveals that **{shortest_gap_region}** faces the most relentless wave overlapping, maintaining the shortest average quiet period of **{shortest_gap_val:.1f} hours** between consecutive sirens.
    """
    return summary_text

# ==========================================
# STEP 4: Sidebar Navigation & Dynamic Filtering
# ==========================================
st.sidebar.title("Navigation Menu")
nav_choice = st.sidebar.radio("Go to:", ["National Overview", "Regional Deep Dive", "Quiet Time Rankings"])

st.sidebar.markdown("---")

if nav_choice == "National Overview":
    st.sidebar.subheader("Global Filters")
    num_corridors = st.sidebar.slider("Widespread Attacks to Display:", min_value=3, max_value=10, value=3)

elif nav_choice == "Regional Deep Dive":
    st.sidebar.subheader("Regional Filters")
    selected_region = st.sidebar.selectbox("Select Region for Deep Dive:", unique_regions)

# ==========================================
# STEP 5: Main Panel Content Routing
# ==========================================
st.title("🇺🇦 Ukraine Air Raid Alerts: Time Series Dashboard")

# --- CASE 1: National Overview ---
if nav_choice == "National Overview":
    st.markdown(generate_executive_summary(df_clean, region_stats, gap_summary_df))
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Air Raid Alerts", f"{len(df_clean):,}")
    col2.metric("Mean Duration", f"{df_clean['duration_min'].mean():.1f} min")
    col3.metric("Median Duration", f"{df_clean['duration_min'].median():.1f} min")
    col4.metric("Standard Deviation", f"{df_clean['duration_min'].std():.1f} min")
    
    st.markdown("### National Visualizations (Kyiv Time)")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Horizontal Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    sns.barplot(data=region_stats, x='alert_count', y='region', hue='region', palette='viridis', legend=False, ax=ax1)
    ax1.set_title('Total Number of Air Raid Alerts by Region', fontsize=14)
    ax1.set_xlabel('Number of Alerts')
    ax1.set_ylabel('Region')
    st.pyplot(fig1)
    
    # Plot 2: Line Chart for Hourly Trend (Safely reindexed to 24 hours)
    hourly_trend = df_clean.groupby('hour').size().reindex(range(24), fill_value=0).reset_index(name='count')
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=hourly_trend, x='hour', y='count', marker='o', color='crimson', linewidth=2, ax=ax2)
    ax2.set_title('National Air Raid Alert Activity Trend by Hour of the Day', fontsize=14)
    ax2.set_xlabel('Hour of the Day (Kyiv Time)')
    ax2.set_ylabel('Total Alert Count')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels(hour_labels, rotation=45)
    st.pyplot(fig2)
    
    # Plot 3: National Global Heatmap (Safely reindexed columns)
    national_pivot = df_clean.pivot_table(index='region', columns='hour', values='duration_min', aggfunc='count')
    # Force 24 columns and fill NaNs
    national_pivot = national_pivot.reindex(index=region_stats['region'], columns=range(24)).fillna(0)
    # Rename columns to our formatted strings
    national_pivot.columns = hour_labels
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(national_pivot, cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Total Alert Count'}, ax=ax3)
    ax3.set_title('National Heatmap: Alert Density by Region and Hour of Day', fontsize=14)
    ax3.set_xlabel('Hour of the Day (Kyiv Time)')
    ax3.set_ylabel('Region')
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)
    
    # Widespread Corridor Attack List
    st.markdown(f"### Top {num_corridors} Widespread Attack Corridors")
    df_clean['date_hour'] = df_clean['started_at'].dt.floor('h')
    corridors = df_clean.groupby('date_hour')['region'].unique().reset_index()
    corridors['regions_count'] = corridors['region'].apply(len)
    top_corridors = corridors.sort_values(by='regions_count', ascending=False).head(num_corridors)
    
    for index, row in top_corridors.iterrows():
        time_str = row['date_hour'].strftime('%Y-%m-%d %H:00 (Kyiv)')
        st.info(f"**[{time_str}]** Affected Regions Count: **{row['regions_count']}** \n*Regions:* {', '.join(row['region'])}")

# --- CASE 2: Regional Deep Dive ---
elif nav_choice == "Regional Deep Dive":
    st.markdown(f"## Customized Analytics for: **{selected_region}**")
    
    region_subset = df_clean[df_clean['region'] == selected_region].copy()
    avg_quiet_time = gap_summary_df[gap_summary_df['region'] == selected_region]['avg_quiet_time_hours'].values[0]
    
    r_col1, r_col2 = st.columns(2)
    r_col1.metric("Total Alerts in Region", f"{len(region_subset):,}")
    r_col2.metric("Average Quiet Time Between Waves", f"{avg_quiet_time:.1f} Hours")
    
    # Build Localized Heatmap
    df_clean['day_of_week'] = df_clean['started_at'].dt.dayofweek
    days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df_clean['day_name'] = df_clean['day_of_week'].map(days_map)
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    region_subset_heatmap = df_clean[df_clean['region'] == selected_region]
    
    regional_pivot = region_subset_heatmap.pivot_table(
        index='day_name', columns='hour', values='duration_min', aggfunc='count'
    )
    # Force 7 days (rows) and 24 hours (columns)
    regional_pivot = regional_pivot.reindex(index=ordered_days, columns=range(24)).fillna(0)
    # Rename columns safely
    regional_pivot.columns = hour_labels
    
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    sns.heatmap(regional_pivot, cmap='rocket_r', annot=False, fmt='g', cbar_kws={'label': 'Alerts'}, ax=ax4)
    ax4.set_title(f'Time Patterns Heatmap: {selected_region}', fontsize=14)
    ax4.set_xlabel('Hour of the Day (Kyiv Time)')
    ax4.set_ylabel('Day of Week')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

# --- CASE 3: Quiet Time Rankings ---
elif nav_choice == "Quiet Time Rankings":
    st.markdown("## National Quiet Time Rankings")
    st.markdown("Regions sorted by the average period of silence between consecutive air raid alerts. Shorter periods indicate high intensity or frequent tactical overlapping.")
    st.dataframe(gap_summary_df, use_container_width=True)

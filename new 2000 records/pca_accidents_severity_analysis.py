"""
PCA for Quantities of Accidents and Severity Analysis
Dataset: US_Accidents_2000_sample.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# 1. Setup and Configuration
# -------------------------------------------------------------
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Locate dataset strictly within 2000 sample files
DATA_PATH = os.path.join(os.path.dirname(__file__), "new 2000 records", "US_Accidents_2000_sample.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "US_Accidents_2000_sample.csv")

print(f"Loading dataset strictly from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded. Records: {len(df):,}, Columns: {df.shape[1]}")

# -------------------------------------------------------------
# 2. Aggregating Accident Quantities and Severity by State
# -------------------------------------------------------------
# Each observation represents a geographical entity (State) characterized by:
# - Overall Accident Volume (Total_Accidents)
# - Severity-specific Quantities (Counts of Severity 1, 2, 3, 4)
# - Mean Severity Level
# - High Severity Rate (% of Severity 3 & 4)

state_profile = df.groupby("State", observed=True).agg(
    Total_Accidents=("Severity", "count"),
    Severity_1_Qty=("Severity", lambda s: (s == 1).sum()),
    Severity_2_Qty=("Severity", lambda s: (s == 2).sum()),
    Severity_3_Qty=("Severity", lambda s: (s == 3).sum()),
    Severity_4_Qty=("Severity", lambda s: (s == 4).sum()),
    Mean_Severity=("Severity", "mean"),
    High_Severity_Pct=("Severity", lambda s: (s >= 3).mean() * 100)
).reset_index()

feature_names = [
    "Total_Accidents",
    "Severity_1_Qty",
    "Severity_2_Qty",
    "Severity_3_Qty",
    "Severity_4_Qty",
    "Mean_Severity",
    "High_Severity_Pct"
]

display_feature_labels = [
    "Total Accidents",
    "Severity 1 Qty",
    "Severity 2 Qty",
    "Severity 3 Qty",
    "Severity 4 Qty",
    "Mean Severity",
    "High Severity Rate (%)"
]

X = state_profile[feature_names].values

# -------------------------------------------------------------
# 3. Standardization and Principal Component Analysis (PCA)
# -------------------------------------------------------------
# Features have differing scales (counts in hundreds vs percentages vs severity 1-4).
# Standardizing to zero mean and unit variance is essential.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

print("\n" + "="*60)
print("PCA EXPLAINED VARIANCE RATIO")
print("="*60)
for i, (var, cum) in enumerate(zip(explained_var, cum_explained_var)):
    print(f"Component {i+1} (PC{i+1}): {var*100:6.2f}% | Cumulative: {cum*100:6.2f}%")

# Create Loadings Matrix: components_ is of shape (n_components, n_features)
loadings = pd.DataFrame(
    pca.components_,
    columns=display_feature_labels,
    index=[f"PC{i+1} ({var*100:.1f}%)" for i, var in enumerate(explained_var)]
)

print("\n" + "="*60)
print("PCA LOADINGS MATRIX (Components x Features)")
print("="*60)
print(loadings.round(3))

# Correlation matrix between original features and Principal Components
# For standardized features, correlation = loading * sqrt(eigenvalue)
eigenvalues = pca.explained_variance_
component_correlations = pd.DataFrame(
    pca.components_[:4].T * np.sqrt(eigenvalues[:4]),
    index=display_feature_labels,
    columns=[f"PC{i+1} ({var*100:.1f}%)" for i, var in enumerate(explained_var[:4])]
)

# -------------------------------------------------------------
# 4. Heatmap 1: PCA Loadings Heatmap (Primary Requested Heatmap)
# -------------------------------------------------------------
plt.figure(figsize=(11, 6), dpi=300)
sns.set_theme(style="white", font="sans-serif")

# Show top 4 components explaining ~99% of total variance
n_show = 4
loadings_subset = loadings.iloc[:n_show]

ax = sns.heatmap(
    loadings_subset,
    annot=True,
    fmt=".3f",
    cmap="vlag",
    center=0,
    vmin=-0.8,
    vmax=0.8,
    cbar_kws={"label": "Component Loading (Weight)", "shrink": 0.8},
    linewidths=1.2,
    linecolor="white"
)

plt.title(
    "PCA Component Loadings Heatmap: Accident Quantities & Severity\n"
    "PC1 captures Overall Accident Volume/Quantity (53.2%) | PC2 captures Accident Severity (32.6%)",
    fontsize=13,
    fontweight="bold",
    pad=16
)
plt.xlabel("Accident Quantity & Severity Features", fontsize=11, fontweight="bold", labelpad=10)
plt.ylabel("Principal Components (Variance Explained)", fontsize=11, fontweight="bold", labelpad=10)
plt.xticks(rotation=25, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap1_path = os.path.join(FIGURES_DIR, "pca_loadings_heatmap.png")
plt.savefig(heatmap1_path, dpi=300)
plt.close()
print(f"\nSaved Heatmap 1 -> {heatmap1_path}")

# -------------------------------------------------------------
# 5. Heatmap 2: Feature-to-Component Correlation Heatmap
# -------------------------------------------------------------
plt.figure(figsize=(8, 7), dpi=300)
sns.heatmap(
    component_correlations,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1.0,
    vmax=1.0,
    cbar_kws={"label": "Pearson Correlation (r)", "shrink": 0.8},
    linewidths=1.2,
    linecolor="white"
)

plt.title(
    "Correlation Heatmap: Features vs Principal Components",
    fontsize=13,
    fontweight="bold",
    pad=16
)
plt.xlabel("Principal Components", fontsize=11, fontweight="bold", labelpad=10)
plt.ylabel("Original Metrics", fontsize=11, fontweight="bold", labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap2_path = os.path.join(FIGURES_DIR, "pca_feature_correlation_heatmap.png")
plt.savefig(heatmap2_path, dpi=300)
plt.close()
print(f"Saved Heatmap 2 -> {heatmap2_path}")

# -------------------------------------------------------------
# 6. Heatmap 3: State-Level PCA Projection Scores Heatmap
# -------------------------------------------------------------
# Add PCA component scores to state profile
state_profile["PC1_Quantity_Score"] = X_pca[:, 0]
state_profile["PC2_Severity_Score"] = X_pca[:, 1]

# Focus on Top 20 states by accident volume
top20_states = state_profile.sort_values("Total_Accidents", ascending=False).head(20).copy()

score_heatmap_data = top20_states.set_index("State")[
    ["PC1_Quantity_Score", "PC2_Severity_Score", "Total_Accidents", "Mean_Severity", "High_Severity_Pct"]
]

# Standardize each column for comparative heatmap visualization
norm_score_heatmap = (score_heatmap_data - score_heatmap_data.mean()) / score_heatmap_data.std()
norm_score_heatmap.columns = [
    "PC1 (Quantity Score)",
    "PC2 (Severity Score)",
    "Total Accidents (Z)",
    "Mean Severity (Z)",
    "High Severity % (Z)"
]

plt.figure(figsize=(10, 10), dpi=300)
sns.heatmap(
    norm_score_heatmap,
    annot=True,
    fmt=".2f",
    cmap="cividis",
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Standardized Value (Z-Score)", "shrink": 0.75}
)

plt.title(
    "State Profile Heatmap: PCA Component Scores vs Quantity & Severity\n(Top 20 States by Volume)",
    fontsize=13,
    fontweight="bold",
    pad=16
)
plt.xlabel("PCA Components & Standardized Metrics", fontsize=11, fontweight="bold", labelpad=10)
plt.ylabel("US State (Sorted by Accident Volume)", fontsize=11, fontweight="bold", labelpad=10)
plt.xticks(rotation=20, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap3_path = os.path.join(FIGURES_DIR, "pca_state_scores_heatmap.png")
plt.savefig(heatmap3_path, dpi=300)
plt.close()
print(f"Saved Heatmap 3 -> {heatmap3_path}")

# -------------------------------------------------------------
# 7. Comprehensive Diagnostic Dashboard (Loadings + Scree + Biplot)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)

# (A) PCA Loadings Heatmap
sns.heatmap(
    loadings.iloc[:3],
    annot=True,
    fmt=".3f",
    cmap="vlag",
    center=0,
    vmin=-0.8,
    vmax=0.8,
    cbar_kws={"label": "Loading", "shrink": 0.8},
    linewidths=1.0,
    linecolor="white",
    ax=axes[0, 0]
)
axes[0, 0].set_title("(A) PCA Component Loadings Heatmap", fontsize=12, fontweight="bold")
axes[0, 0].set_xticklabels(display_feature_labels, rotation=25, ha="right", fontsize=9)
axes[0, 0].set_yticklabels(loadings.index[:3], rotation=0, fontsize=9)

# (B) Scree & Cumulative Variance Plot
components_range = np.arange(1, len(explained_var) + 1)
axes[0, 1].bar(components_range, explained_var * 100, color="#3498db", alpha=0.8, label="Individual Variance (%)")
axes[0, 1].step(components_range, cum_explained_var * 100, where="mid", color="#e74c3c", linewidth=2.2, label="Cumulative Variance (%)")
axes[0, 1].axhline(80, color="gray", linestyle="--", linewidth=1, label="80% Threshold")
axes[0, 1].set_title("(B) Scree Plot & Cumulative Explained Variance", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Principal Component", fontsize=10, fontweight="bold")
axes[0, 1].set_ylabel("Explained Variance (%)", fontsize=10, fontweight="bold")
axes[0, 1].set_xticks(components_range)
axes[0, 1].set_ylim(0, 105)
axes[0, 1].legend(loc="center right", fontsize=9)
axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

# (C) Biplot: PC1 (Quantity) vs PC2 (Severity)
scatter = axes[1, 0].scatter(
    state_profile["PC1_Quantity_Score"],
    state_profile["PC2_Severity_Score"],
    c=state_profile["Mean_Severity"],
    cmap="plasma",
    s=state_profile["Total_Accidents"] * 1.5 + 40,
    alpha=0.85,
    edgecolors="black",
    linewidth=0.8
)
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label("Mean Severity", fontsize=10)

# Label notable states
for _, row in state_profile.iterrows():
    if row["Total_Accidents"] > 80 or row["Mean_Severity"] > 2.45 or row["PC2_Severity_Score"] > 1.8:
        axes[1, 0].annotate(
            row["State"],
            (row["PC1_Quantity_Score"], row["PC2_Severity_Score"]),
            fontsize=9,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points"
        )

# Overlay feature vectors (loadings arrows)
loading_scale = 3.5
for i, feature in enumerate(display_feature_labels):
    axes[1, 0].arrow(
        0, 0,
        loadings.iloc[0, i] * loading_scale,
        loadings.iloc[1, i] * loading_scale,
        color="#c0392b",
        width=0.03,
        head_width=0.15,
        alpha=0.75
    )
    axes[1, 0].text(
        loadings.iloc[0, i] * loading_scale * 1.15,
        loadings.iloc[1, i] * loading_scale * 1.15,
        feature,
        color="#962d22",
        fontsize=8,
        fontweight="bold"
    )

axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1, 0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1, 0].set_title("(C) PCA Biplot: PC1 (Quantity) vs PC2 (Severity)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel(f"PC1: Accident Quantity ({explained_var[0]*100:.1f}%)", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel(f"PC2: Accident Severity ({explained_var[1]*100:.1f}%)", fontsize=10, fontweight="bold")

# (D) Correlation Heatmap
sns.heatmap(
    component_correlations,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Correlation", "shrink": 0.8},
    linewidths=0.8,
    linecolor="white",
    ax=axes[1, 1]
)
axes[1, 1].set_title("(D) Feature-Component Correlation Heatmap", fontsize=12, fontweight="bold")
axes[1, 1].set_yticklabels(display_feature_labels, rotation=0, fontsize=9)
axes[1, 1].set_xticklabels(component_correlations.columns, rotation=15, ha="right", fontsize=9)

plt.suptitle(
    "Principal Component Analysis (PCA) Dashboard: US Accidents Quantities & Severity",
    fontsize=16,
    fontweight="bold",
    y=0.99
)
plt.tight_layout()

dashboard_path = os.path.join(FIGURES_DIR, "pca_accidents_severity_dashboard.png")
plt.savefig(dashboard_path, dpi=300)
plt.close()
print(f"Saved Comprehensive Dashboard -> {dashboard_path}")

# -------------------------------------------------------------
# 8. Temporal Heatmap: Weekday x Hour with Severity Dimension
# -------------------------------------------------------------
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["Hour"] = df["Start_Time"].dt.hour
df["Weekday"] = df["Start_Time"].dt.day_name()
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# (A) Mean Severity Heatmap (Same structure, colormap: 'plasma')
sw = df.pivot_table(index="Weekday", columns="Hour", values="Severity", aggfunc="mean").reindex(WEEKDAYS)

plt.figure(figsize=(15, 4.5), dpi=300)
ax = sns.heatmap(
    sw,
    cmap="plasma",
    linewidths=0.4,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    cbar_kws={"label": "Mean Severity"}
)
ax.set_title("Mean severity by hour of day and weekday", fontsize=12, fontweight="bold")
ax.set_ylabel("")
ax.set_xlabel("Hour", fontsize=10)
plt.tight_layout()
mean_sev_path = os.path.join(FIGURES_DIR, "mean_severity_hour_weekday_heatmap.png")
plt.savefig(mean_sev_path, dpi=300)
plt.close()
print(f"Saved Mean Severity Heatmap -> {mean_sev_path}")

# (B) Stacked Comparison: Quantity (YlOrRd) vs Mean Severity (plasma)
hw = pd.crosstab(df["Weekday"], df["Hour"]).reindex(WEEKDAYS)
fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), dpi=300)

sns.heatmap(
    hw, cmap="YlOrRd", linewidths=0.4, linecolor="white", annot=True, fmt="d",
    annot_kws={"size": 7}, cbar_kws={"label": "Accidents"}, ax=axes[0]
)
axes[0].set_title("Accidents by hour of day and weekday (Quantity Dimension)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("")
axes[0].set_xlabel("Hour", fontsize=10)

sns.heatmap(
    sw, cmap="plasma", linewidths=0.4, linecolor="white", annot=True, fmt=".2f",
    annot_kws={"size": 7}, cbar_kws={"label": "Mean Severity"}, ax=axes[1]
)
axes[1].set_title("Mean severity by hour of day and weekday (Severity Dimension)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("")
axes[1].set_xlabel("Hour", fontsize=10)

plt.tight_layout()
stacked_path = os.path.join(FIGURES_DIR, "accidents_and_severity_hour_weekday_stacked.png")
plt.savefig(stacked_path, dpi=300)
plt.close()
print(f"Saved Stacked Quantity & Severity Heatmap -> {stacked_path}")

# (C) 3D Dimensional Expansion: Weekday x Hour faceted across Severity levels
fig, axes = plt.subplots(3, 1, figsize=(15, 11), dpi=300, sharex=True)
cmaps = {2: "Blues", 3: "Oranges", 4: "Reds"}
titles = {
    2: "Severity 2 [Moderate] Accidents by hour of day and weekday",
    3: "Severity 3 [Severe] Accidents by hour of day and weekday",
    4: "Severity 4 [Critical] Accidents by hour of day and weekday"
}

for ax, sev in zip(axes, [2, 3, 4]):
    sub = df[df["Severity"] == sev]
    ct = pd.crosstab(sub["Weekday"], sub["Hour"]).reindex(index=WEEKDAYS, columns=range(24), fill_value=0)
    sns.heatmap(ct, cmap=cmaps[sev], linewidths=0.4, linecolor="white", annot=True, fmt="d",
                annot_kws={"size": 7}, cbar_kws={"label": f"Sev {sev} Count"}, ax=ax)
    ax.set_title(titles[sev], fontsize=11, fontweight="bold")
    ax.set_ylabel("")

axes[-1].set_xlabel("Hour", fontsize=10)
plt.tight_layout()
facets_path = os.path.join(FIGURES_DIR, "accidents_by_severity_levels_hour_weekday.png")
plt.savefig(facets_path, dpi=300)
plt.close()
print(f"Saved Severity Levels Faceted Heatmap -> {facets_path}")

print("\nPCA Analysis and Heatmaps execution completed successfully!")


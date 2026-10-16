import json

path = "new 2000 records/US_Accidents_Data_Description.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Heatmap – Hour of day × Weekday by Severity\n",
        "Extends the temporal dimension with **Severity**. While accident frequency peaks during weekday rush hours, late-night and early-morning hours often show heightened severity."
    ]
}

code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Mean Severity by hour of day and weekday (incorporating the Severity dimension with a new colormap)\n",
        "sw = df.pivot_table(index=\"Weekday\", columns=\"Hour\", values=\"Severity\", aggfunc=\"mean\").reindex(WEEKDAYS)\n",
        "fig, ax = plt.subplots(figsize=(15, 4.5))\n",
        "sns.heatmap(sw, cmap=\"plasma\", linewidths=.4, linecolor=\"white\", annot=True, fmt=\".2f\",\n",
        "            annot_kws={\"size\": 7}, cbar_kws={\"label\": \"Mean Severity\"}, ax=ax)\n",
        "ax.set_title(\"Mean severity by hour of day and weekday\"); ax.set_ylabel(\"\")\n",
        "plt.tight_layout(); plt.show()\n"
    ]
}

code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Multi-dimensional breakdown: Weekday x Hour faceted across Severity levels\n",
        "fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)\n",
        "cmaps = {2: \"Blues\", 3: \"Oranges\", 4: \"Reds\"}\n",
        "titles = {\n",
        "    2: \"Severity 2 [Moderate] Accidents by hour of day and weekday\",\n",
        "    3: \"Severity 3 [Severe] Accidents by hour of day and weekday\",\n",
        "    4: \"Severity 4 [Critical] Accidents by hour of day and weekday\"\n",
        "}\n",
        "\n",
        "for ax, sev in zip(axes, [2, 3, 4]):\n",
        "    sub = df[df[\"Severity\"] == sev]\n",
        "    ct = pd.crosstab(sub[\"Weekday\"], sub[\"Hour\"]).reindex(index=WEEKDAYS, columns=range(24), fill_value=0)\n",
        "    sns.heatmap(ct, cmap=cmaps[sev], linewidths=.4, linecolor=\"white\", annot=True, fmt=\"d\",\n",
        "                annot_kws={\"size\": 7}, cbar_kws={\"label\": f\"Sev {sev} Accidents\"}, ax=ax)\n",
        "    ax.set_title(titles[sev])\n",
        "    ax.set_ylabel(\"\")\n",
        "\n",
        "axes[-1].set_xlabel(\"Hour\")\n",
        "plt.tight_layout(); plt.show()\n"
    ]
}

src_existing = "".join(["".join(c.get("source", [])) for c in nb["cells"]])
if "Mean severity by hour of day and weekday" not in src_existing:
    nb["cells"].insert(24, md_cell)
    nb["cells"].insert(25, code_cell_1)
    nb["cells"].insert(26, code_cell_2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully inserted new severity heatmap cells into notebook!")
else:
    print("Cells already present in notebook.")

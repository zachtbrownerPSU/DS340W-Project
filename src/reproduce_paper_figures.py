"""
Reproduce the descriptive figures from the parent paper:

Patil, K. P. and Kapfhamme, G. (2022). "Machine Learning Meets Sports: Data
Analytics Demonstrating the Relationship Between Playing Surface & the
Injury/Performance of Athletes." IJERT Vol. 11 Issue 08.
https://www.ijert.org/research/machine-learning-meets-sports-IJERTV11IS080140.pdf

The parent paper does not publish its own code. This script reads the same
official Kaggle "NFL 1st and Future - Analytics" dataset the paper uses and
rebuilds each of its figures from the raw files:

  FIG-1  Injury rate (%) by field type (synthetic vs. natural)
  FIG-2  Number of injured players by days-away-from-play bucket, by surface
  FIG-3  Percent higher injured-player count on synthetic vs. natural, by bucket
  FIG-4  Injury (x-position) distribution by body part, artificial turf
  FIG-5  Injury (x-position) distribution by body part, natural turf
  FIG-6  Field-location density of injuries, artificial turf
  FIG-6b Field-location density of injuries, natural turf
  FIG-7  Distribution of per-play maximum speed, by surface
  FIG-8  Distribution of per-play maximum distance, by surface

Run:
    python src/reproduce_paper_figures.py

Reads from data/raw/{InjuryRecord,PlayList,PlayerTrackData}.csv and writes
PNGs to figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

INJURY_PATH = os.path.join(DATA_DIR, "InjuryRecord.csv")
PLAYLIST_PATH = os.path.join(DATA_DIR, "PlayList.csv")
TRACK_PATH = os.path.join(DATA_DIR, "PlayerTrackData.csv")

TRACK_CHUNKSIZE = 2_000_000

sns.set_theme(style="whitegrid")

# seaborn's stripplot jitter draws from the global numpy RNG, so seed it to
# keep regenerated figures byte-identical between runs
np.random.seed(0)


def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def fig1_injury_rate(injury, playlist):
    """FIG-1: injury rate (%) = injuries on surface / plays on surface * 100."""
    plays_by_surface = playlist["FieldType"].value_counts()
    injuries_by_surface = injury["Surface"].value_counts()
    rate = (injuries_by_surface / plays_by_surface * 100).reindex(
        ["Synthetic", "Natural"]
    )

    ratio = rate["Synthetic"] / rate["Natural"]
    print(f"FIG-1: injury rate synthetic={rate['Synthetic']:.3f}%, "
          f"natural={rate['Natural']:.3f}%, ratio={ratio:.2f}x")

    plt.figure(figsize=(7, 4))
    bars = plt.barh(rate.index, rate.values, color=["#4C72B0", "#DD8452"])
    plt.xlabel("Injury rate (%)")
    plt.title(f"Injuries occur over {ratio:.1f}x more on synthetic field")
    for bar, value in zip(bars, rate.values):
        plt.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}%",
                  va="center")
    savefig("fig1_injury_rate_by_surface.png")


def fig2_fig3_severity(injury):
    """FIG-2/FIG-3: injured-player counts and synthetic-vs-natural % gap
    across the DM_M1 / DM_M7 / DM_M28 / DM_M42 days-away-from-play buckets."""
    buckets = ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]
    labels = ["1 Day", "7 Days", "28 Days", "42 Days"]

    counts = pd.DataFrame({
        "Synthetic": [injury.loc[injury["Surface"] == "Synthetic", b].sum() for b in buckets],
        "Natural": [injury.loc[injury["Surface"] == "Natural", b].sum() for b in buckets],
    }, index=labels)
    print("FIG-2 counts by days-away bucket:\n", counts)

    ax = counts.plot(kind="bar", figsize=(8, 5), color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Number of players")
    ax.set_xlabel("Days away from play")
    ax.set_title("Number of players in each days-away-from-play bucket, by surface")
    plt.xticks(rotation=0)
    savefig("fig2_days_away_counts_by_surface.png")

    # The paper calls FIG-3 the "percentage of higher numbers players injured in
    # Artificial Turf ... compare to Natural Turf", but its published bar values
    # (9 / 16 / 39 / 40) are reproduced exactly by the synthetic share of each
    # bucket above even, i.e. (syn - nat) / (syn + nat) -- not by the relative
    # increase (syn - nat) / nat, which would give 21 / 39 / 127 / 136 on the
    # paper's own FIG-2 counts. We follow the paper's actual formula.
    share_diff = ((counts["Synthetic"] - counts["Natural"])
                  / (counts["Synthetic"] + counts["Natural"]) * 100)
    print("FIG-3 synthetic share difference (%):\n", share_diff)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, share_diff.values, color="#8172B2")
    plt.ylabel("Synthetic share of bucket above even (%)")
    plt.title("Skew toward synthetic turf by days-away-from-play bucket")
    for i, value in enumerate(share_diff.values):
        plt.text(i, value, f"{value:.0f}%", ha="center", va="bottom")
    savefig("fig3_pct_higher_synthetic.png")


def load_injury_track_summary(injury_with_key):
    """Single chunked pass over the ~4GB tracking file: pull every raw
    tracking row belonging to an injury play (for FIG-4/5/6), and compute
    the per-play mean position and per-play max speed/distance for every
    play in the dataset (for FIG-7/8)."""
    injury_keys = set(injury_with_key["PlayKey"])

    injury_rows = []
    play_max_parts = []

    for chunk in pd.read_csv(TRACK_PATH, usecols=["PlayKey", "x", "y", "dis", "s"],
                              chunksize=TRACK_CHUNKSIZE):
        mask = chunk["PlayKey"].isin(injury_keys)
        if mask.any():
            injury_rows.append(chunk.loc[mask])

        agg = chunk.groupby("PlayKey", sort=False).agg(
            max_s=("s", "max"), max_dis=("dis", "max")
        )
        play_max_parts.append(agg)

    injury_track = pd.concat(injury_rows, ignore_index=True) if injury_rows else pd.DataFrame()

    play_max = pd.concat(play_max_parts)
    play_max = play_max.groupby("PlayKey", sort=False).agg(max_s=("max_s", "max"),
                                                             max_dis=("max_dis", "max"))
    return injury_track, play_max


def fig4_fig5_injury_type(injury_with_key, injury_track):
    """FIG-4/FIG-5: distribution of each play's mean x-position, grouped by
    body part, split into one figure per surface (artificial, natural)."""
    body_part_map = {"Toes": "Foot", "Heel": "Foot"}
    injury_with_key = injury_with_key.copy()
    injury_with_key["BodyPartGroup"] = injury_with_key["BodyPart"].replace(body_part_map)

    play_x = injury_track.groupby("PlayKey")["x"].mean().rename("mean_x")
    merged = injury_with_key.merge(play_x, on="PlayKey", how="inner")

    for surface, fname, title in [
        ("Synthetic", "fig4_injury_type_artificial_turf.png", "Type of Injuries on Artificial Turf"),
        ("Natural", "fig5_injury_type_natural_turf.png", "Type of Injuries on Natural Turf"),
    ]:
        subset = merged[merged["Surface"] == surface]
        plt.figure(figsize=(8, 5))
        order = subset["BodyPartGroup"].value_counts().index
        sns.boxplot(data=subset, x="mean_x", y="BodyPartGroup", order=order,
                    hue="BodyPartGroup", legend=False)
        sns.stripplot(data=subset, x="mean_x", y="BodyPartGroup", order=order,
                      color="black", alpha=0.5, size=4)
        plt.xlabel("Field x-position (yards)")
        plt.ylabel("Body part")
        plt.title(title)
        savefig(fname)


def fig6_injury_location(injury_with_key, injury_track):
    """FIG-6: 2D density of injury field location (x, y), one plot per surface."""
    play_xy = injury_track.groupby("PlayKey")[["x", "y"]].mean()
    merged = injury_with_key.merge(play_xy, on="PlayKey", how="inner")

    for surface, fname, title, color in [
        ("Synthetic", "fig6_injury_location_artificial_turf.png",
         "Location of Injuries on Artificial Turf", "Reds"),
        ("Natural", "fig6b_injury_location_natural_turf.png",
         "Location of Injuries on Natural Turf", "Blues"),
    ]:
        subset = merged[merged["Surface"] == surface]
        plt.figure(figsize=(7, 5))
        sns.kdeplot(data=subset, x="x", y="y", fill=True, cmap=color, thresh=0.05)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        savefig(fname)


def fig7_fig8_speed_distance(play_max, playlist):
    """FIG-7/FIG-8: distribution of per-play maximum speed and distance,
    split by field surface."""
    surface_lookup = playlist[["PlayKey", "FieldType"]].drop_duplicates("PlayKey")
    merged = play_max.reset_index().merge(surface_lookup, on="PlayKey", how="inner")
    merged = merged[merged["FieldType"].isin(["Synthetic", "Natural"])]

    plt.figure(figsize=(8, 5))
    for surface, color in [("Synthetic", "#4C72B0"), ("Natural", "#DD8452")]:
        sns.kdeplot(merged.loc[merged["FieldType"] == surface, "max_s"],
                    label=surface, fill=True, alpha=0.4, color=color)
    plt.xlabel("Yards per second")
    plt.ylabel("Density")
    plt.title("Distribution of Max Speed by Field Type")
    # match the paper's FIG-7 axis; excludes 5 plays (0.002%) whose recorded
    # max speed exceeds 12 yd/s, which are tracking artifacts
    plt.xlim(-2, 12)
    plt.xticks(np.arange(-2, 13, 2))
    plt.legend()
    savefig("fig7_max_speed_distribution.png")

    plt.figure(figsize=(8, 5))
    for surface, color in [("Synthetic", "#55A868"), ("Natural", "#8172B2")]:
        sns.kdeplot(merged.loc[merged["FieldType"] == surface, "max_dis"],
                    label=surface, fill=True, alpha=0.4, color=color)
    plt.xlabel("Yards per 0.1s frame")
    plt.ylabel("Density")
    plt.title("Distribution of Max Distance by Field Type")
    # match the paper's FIG-8 axis; excludes 0.6% of plays above 1.2 yards per
    # frame, which correspond to the same tracking artifacts
    plt.xlim(-0.2, 1.2)
    plt.xticks(np.round(np.arange(-0.2, 1.21, 0.2), 1))
    plt.legend()
    savefig("fig8_max_distance_distribution.png")


def main():
    print("Loading InjuryRecord.csv and PlayList.csv ...")
    injury = pd.read_csv(INJURY_PATH)
    playlist = pd.read_csv(PLAYLIST_PATH)

    fig1_injury_rate(injury, playlist)
    fig2_fig3_severity(injury)

    injury_with_key = injury.dropna(subset=["PlayKey"]).copy()
    print(f"{len(injury_with_key)} of {len(injury)} injuries have a PlayKey "
          f"and can be located in the tracking data.")

    print("Scanning PlayerTrackData.csv (chunked) for injury-play tracking "
          "rows and per-play max speed/distance ...")
    injury_track, play_max = load_injury_track_summary(injury_with_key)

    fig4_fig5_injury_type(injury_with_key, injury_track)
    fig6_injury_location(injury_with_key, injury_track)
    fig7_fig8_speed_distance(play_max, playlist)

    print("Done. All figures written to", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()

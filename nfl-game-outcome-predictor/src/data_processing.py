import os
import pandas as pd

file_path = "nfl-game-outcome-predictor/data/raw/NFL-2009-2016.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path, low_memory=False)
print("Original shape:", df.shape)

df = df.drop_duplicates()
df = df.loc[:, df.isnull().mean() < 0.5]

critical_cols = ["Season", "GameID", "PlayType"]
df = df.dropna(subset=[col for col in critical_cols if col in df.columns])

print("Cleaned shape:", df.shape)

# Only keep columns that exist in the DataFrame
cols_to_keep = [
    "GameID", "posteam", "DefensiveTeam", "PlayType", "Yards.Gained",
    "PassAttempt", "RushAttempt", "InterceptionThrown", "Fumble", 
    "Touchdown", "FieldGoalResult", "FieldGoalDistance", "FirstDown",
    "PosTeamScore", "DefTeamScore", "HomeTeam", "AwayTeam", "Season", "EPA"
]
existing_cols = [c for c in cols_to_keep if c in df.columns]
df = df[existing_cols].copy()

numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

game_stats = (
    df.groupby(
        [col for col in ["GameID", "posteam", "DefensiveTeam", "HomeTeam", "AwayTeam"] if col in df.columns]
    )
    .agg(
        plays=("PlayType", "count"),
        total_yards=("Yards.Gained", "sum"),
        pass_attempts=("PassAttempt", "sum") if "PassAttempt" in df.columns else ("PlayType", "count"),
        rush_attempts=("RushAttempt", "sum") if "RushAttempt" in df.columns else ("PlayType", "count"),
        turnovers=("InterceptionThrown", "sum") if "InterceptionThrown" in df.columns else ("PlayType", "count"),
        fumbles=("Fumble", "sum") if "Fumble" in df.columns else ("PlayType", "count"),
        touchdowns=("Touchdown", "sum") if "Touchdown" in df.columns else ("PlayType", "count"),
        first_downs=("FirstDown", "sum") if "FirstDown" in df.columns else ("PlayType", "count"),
        avg_epa=("EPA", "mean") if "EPA" in df.columns else ("PlayType", "count"),
        fg_made=("FieldGoalResult", lambda x: (x == "Made").sum()) if "FieldGoalResult" in df.columns else ("PlayType", "count"),
        fg_missed=("FieldGoalResult", lambda x: (x == "Missed").sum()) if "FieldGoalResult" in df.columns else ("PlayType", "count"),
        pos_score=("PosTeamScore", "max") if "PosTeamScore" in df.columns else ("PlayType", "count"),
        def_score=("DefTeamScore", "max") if "DefTeamScore" in df.columns else ("PlayType", "count"),
    )
    .reset_index()
)

game_stats["yards_per_play"] = game_stats["total_yards"] / (game_stats["plays"] + 1e-5)
game_stats["rush_pass_ratio"] = game_stats["rush_attempts"] / (game_stats["pass_attempts"] + 1e-5)
game_stats["turnover_diff"] = game_stats["turnovers"] + game_stats["fumbles"]
game_stats["score_diff"] = game_stats["pos_score"] - game_stats["def_score"]
game_stats["win"] = (game_stats["score_diff"] > 0).astype(int)

feature_cols = [
    "plays", "total_yards", "yards_per_play", "rush_pass_ratio",
    "turnover_diff", "avg_epa", "fg_made", "fg_missed", "first_downs"
]
target_col = "win"

df_features = game_stats[[c for c in feature_cols if c in game_stats.columns] + [target_col]]

os.makedirs("nfl-game-outcome-predictor/data/processed", exist_ok=True)
output_path = "nfl-game-outcome-predictor/data/processed/game_stats.csv"
df_features.to_csv(output_path, index=False)

print(f"✅ Data processing completed. Saved {df_features.shape[0]} rows and {df_features.shape[1]} columns to:")
print(f"   {output_path}")
print(df_features.head())

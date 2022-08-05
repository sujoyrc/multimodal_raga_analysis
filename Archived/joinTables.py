import wandb

data_table = "snnithya/Gesture Analysis/easy_1-AG:v0"
video_predictions_table = "snnithya/Gesture Analysis/Predictions:v22"
audio_predictions_table = "snnithya/Gesture Analysis/Predictions:v26"

run = wandb.init(project="Gesture Analysis")

# fetch original songs table
orig_data = run.use_artifact(data_table)
orig_table = orig_data.get("test_table")

audio_data = run.use_artifact(audio_predictions_table) 
audio_table = audio_data.get("test_table")

# video_data = run.use_artifact(video_predictions_table) 
# video_table = video_data.get("test_table")

# join tables on "song_id"
join_table = wandb.JoinedTable(orig_table, audio_table, "unique_id")
#join_table = wandb.JoinedTable(join_table, video_table, "unique_id")
join_at = wandb.Artifact("AG-easy_1-audio-prediction", "analysis")

# add table to artifact and log to W&B
join_at.add(join_table, "predictions")
run.log_artifact(join_at)
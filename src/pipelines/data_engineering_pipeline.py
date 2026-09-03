import subprocess
import sys

scripts = [
    "src/data_engineering/Bronze_to_Silver_HR_Derby_Participants.py",
    "src/data_engineering/Ingest_Bronze_Statcast_Batting.py",
    "src/data_engineering/Bronze_to_Silver_Statcast_Batting.py",
    "src/data_engineering/Gold_Derby_Player.py",
    "src/data_engineering/Gold_Derby_Player_W_Relative_Features.py",
]

for script in scripts:

    print(f"Running {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

    print(f"Completed {script}\n")

print("Pipeline completed successfully.")
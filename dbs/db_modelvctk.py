import os
import random
import sqlite3
import csv
class AudioDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS speaker_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
            """)

    def count_distinct_speakers(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT speaker) FROM speaker_files")
        result = cursor.fetchone()
        return result[0] if result else 0

    def insert_files(self, speaker_files):
        with self.conn:
            for speaker, files in speaker_files.items():
                if isinstance(files, list):
                    for file in files:
                        self.conn.execute("INSERT INTO speaker_files (speaker, file_path) VALUES (?, ?)", (speaker, file))
                else:
                    self.conn.execute("INSERT INTO speaker_files (speaker, file_path) VALUES (?, ?)", (speaker, files))

    def get_random_file(self, speaker):
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path FROM speaker_files WHERE speaker = ? ORDER BY RANDOM() LIMIT 1", (speaker,))
        result = cursor.fetchone()
        return result[0].replace("\\", "*").replace("/", "*").replace("*","/") if result else None

    def get_random_speaker(self):
        cursor = self.conn.execute("SELECT DISTINCT speaker FROM speaker_files")
        speakers = cursor.fetchall()
        if speakers:
            return random.choice(speakers)[0]
        else:
            return None

    def get_all_data(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM speaker_files")
        return cursor.fetchall()

def gather_vctk_files(root_folder: str, all_files: int, dataset_type: str) -> dict:
    output_folders = {}  # speaker: [sound_files...]

    # Automatically determine which CSV file to load based on dataset_type (case-insensitive)
    dataset_type = dataset_type.lower()
    if dataset_type == 'train':
        csv_file = os.path.join(root_folder,'train.csv')
    elif dataset_type == 'test':
        csv_file = os.path.join(root_folder,'test.csv')
    else:
        raise ValueError("dataset_type must be 'train' or 'test'.")

    # Load the train/test file
    csv_files = {}
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            speaker = row[0]  # Get the speaker from the row (first column)
            audio_file = row[-1]  # Get the file path from the row (last column)
            if speaker not in csv_files:
                csv_files[speaker] = []
            csv_files[speaker].append(audio_file)

    for speaker_folder in os.listdir(root_folder):
        speaker_path = os.path.join(root_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            sound_files = []

            # Check if the speaker has entries in the train/test csv
            if speaker_folder in csv_files:
                for filename in csv_files[speaker_folder]:
                    src_filepath = os.path.join(root_folder, filename)
                    if os.path.exists(src_filepath) and filename.endswith("_mic1.flac"):
                        sound_files.append(src_filepath)

            if all_files == 0 and sound_files:
                output_folders[speaker_folder] = random.choice(sound_files)
            else:
                output_folders[speaker_folder] = sound_files
    c = 0
    for s in output_folders:
        c +=len(output_folders[s])
    print(c)
    return output_folders


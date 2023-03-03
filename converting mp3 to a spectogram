import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_file_list_from_dir(datadir, file_extension):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = [file for file in all_files if file.endswith(file_extension)]
    return data_files


voice_dir = 'Data'
spectrogram_dir = 'Spectr_dif'
info_csv = 'Information.csv'

male_count = 0
female_count = 0

# load information CSV file
df = pd.read_csv(info_csv)

# iterate over voice files
for i, voice_file in enumerate(get_file_list_from_dir(voice_dir, '.mp3')):
    row = df.iloc[i]
    gender = row['gender']

    # load voice file and plot spectrogram
    sound, _ = librosa.load(os.path.join(voice_dir, voice_file))
    fig, ax = plt.subplots()
    ax.specgram(sound, Fs=6, cmap="gist_rainbow")

    # save spectrogram to the appropriate directory
    if gender == 'male':
        male_dir = os.path.join(spectrogram_dir, 'train' if male_count < 514 else 'val' if male_count < 1028 else 'test', 'male')
        os.makedirs(male_dir, exist_ok=True)
        fig.savefig(os.path.join(male_dir, voice_file.replace('.mp3', '.png')), dpi=100)
        male_count += 1
    else:
        female_dir = os.path.join(spectrogram_dir, 'train' if female_count < 150 else 'val' if female_count < 300 else 'test', 'female')
        os.makedirs(female_dir, exist_ok=True)
        fig.savefig(os.path.join(female_dir, voice_file.replace('.mp3', '.png')), dpi=100)
        female_count += 1

    plt.close(fig)  # close the figure to free up memory

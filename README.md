Music Genre Classification

This project uses audio feature extraction and machine learning to classify songs into genres such as rock, reggae, classical, and more — based solely on how the song *sounds*.

The project objective

Build a model that can predict the *genre of a song* based on audio features like tempo and MFCCs (Mel-Frequency Cepstral Coefficients), using a Random Forest classifier.

Dataset

- Source: [GTZAN Genre Collection](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)
- 10 genres × 100 songs (each 30 seconds long)
- Audio format: `.au` files

Genres:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

 Methodology

1. Audio Preprocessing

- Loaded `.au` audio files using `librosa`
- Extracted features:
  - Tempo (beats per minute)
  - 13 MFCC means
  - 13 MFCC standard deviations

2. Feature Dataset

- Built a structured `genre_features.csv` file
- Each row = one song's feature vector
- Last column = actual genre label

3. Model Training

- Split dataset (80% train / 20% test)
- Trained a *Random Forest Classifier* using `scikit-learn`

4. Evaluation

- Achieved ~64% accuracy 
- Confusion matrix showed strong predictions for most genres
- Successfully predicted the genre of an unseen song (e.g. `disco.00037.au` → ✔️ disco)


## 📈 Example Output

Confusion Matrix:

![confusion matrix](path/to/your/confusion_matrix_image.png)  <!-- optional -->

Technologies Used

- Python
- Jupyter Notebook
- Librosa (audio processing)
- scikit-learn (machine learning)
- Matplotlib & Seaborn (visualization)
- pandas & numpy (data handling)

# Import necessary libraries for the project
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import torch
import os
import numpy as np
import faiss
import pandas as pd
import matplotlib.pyplot as plt
import time
import torchaudio
import gc
import math
import gradio as gr
import sys

# Load the model used for audio vectorization
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()

# Variable containing the path to the animals.index file on your computer
index_path = "/Users/ariel/Downloads/animals.index"
# Read the animals.index file
index = faiss.read_index(index_path)

# Variable containing the path to the noms_animaux.txt file
chemin_noms_animaux = '/Users/ariel/Downloads/noms_animaux.txt'

# Process the noms_animaux.txt file to link vectors in animals.index to names in noms_animaux.txt
# Read the file content and convert it to a list
with open(chemin_noms_animaux, 'r') as fichier:
    # Use a list comprehension to process each line
    names = [line.strip().strip("'").strip(",").strip() for line in fichier.readlines()]

def bayes_theorem(df, n_top_vectors=50):
    """
    Calculate posterior probabilities using Bayes' theorem.

    This function limits the DataFrame to the top n vectors, calculates the sum of similarities
    for each category, and computes the posterior probabilities normalized by the total probability.

    Parameters:
    df (pd.DataFrame): DataFrame containing similarity percentages and categories.
    n_top_vectors (int): Number of top vectors to consider.

    Returns:
    dict: Normalized posterior probabilities for each category.
    """
    # Limit the DataFrame to the top n vectors
    df_limited = df.head(n_top_vectors)
    # Get unique categories and initialize the posterior probabilities dictionary
    categories = df_limited['names_normalized'].unique()
    probas_a_posteriori = {categorie: 0 for categorie in categories}
    # Calculate uniform prior probabilities
    probas_a_priori = 1/3
    # Sum similarities for each category limited to the top n vectors
    for categorie in categories:
        somme_similarites = df_limited[df_limited['names_normalized'] == categorie]['percentage'].sum()
        probas_a_posteriori[categorie] = somme_similarites * probas_a_priori
    # Normalize the posterior probabilities
    total_proba = sum(probas_a_posteriori.values())
    probas_a_posteriori_normalisees = {categorie: (proba / total_proba) for categorie, proba in probas_a_posteriori.items()}
    return probas_a_posteriori_normalisees

def get_name_from_index(index):
    """
    Get the animal name corresponding to a given vector index.

    Parameters:
    index (int): Index of the vector.

    Returns:
    str: Name of the animal.
    """
    return names[index]

def name_normalisation(name):
    """
    Normalize animal names.

    This function normalizes the names of animals by categorizing them into common types.

    Parameters:
    name (str): Name of the animal.

    Returns:
    str: Normalized animal name.
    """
    if 'dog' in name:
        return "Chien"
    elif 'cat' in name:
        return "Chat"
    elif 'bird' in name:
        return "Oiseau"
    else:
        return "Animal non reconnu"

def exp_negative(x):
    """
    Define the negative exponential function.

    This function applies the negative exponential transformation to a given value.

    Parameters:
    x (float): Input value.

    Returns:
    float: Transformed value.
    """
    return math.exp(-x)

def normalization(embeddings):
    """
    Normalize vectors.

    This function normalizes either a single vector (1D) or a matrix of vectors (2D).
    If the input is 1D, it normalizes the single vector; if 2D, it normalizes each row.

    Parameters:
    embeddings (np.ndarray): Input vector or matrix of vectors.

    Returns:
    np.ndarray: Normalized vector or matrix of vectors.
    """
    # Check if embeddings is a single vector (1D) or a matrix (2D)
    if embeddings.ndim == 1:
        # Normalize a single vector
        norm = np.linalg.norm(embeddings)
        if norm == 0:
            return embeddings
        return embeddings / norm
    else:
        # Normalize each row of a matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

def get_audio_embedding(audio_path):
    """
    Get the audio embedding for a given audio file.

    This function loads the audio file, processes it to obtain the emission,
    flattens and averages the features, normalizes them, and returns the normalized 2D array.

    Parameters:
    audio_path (str): Path to the audio file.

    Returns:
    np.ndarray: Normalized 2D array of audio embedding.
    """
    waveform1, sample_rate1 = torchaudio.load(audio_path)
    waveform1 = torchaudio.functional.resample(waveform1, sample_rate1, bundle.sample_rate)
    with torch.inference_mode():
        emission1, _ = model(waveform1)

    # Flatten the first two dimensions and keep the third
    flattened_features1 = emission1.view(-1, emission1.size(2))
    mean_features1 = flattened_features1.mean(dim=0)
    mean1_array = mean_features1.cpu().numpy().astype(np.float32) 
    mean1_normal = normalization(mean1_array)
    mean1_normal_2d = mean1_normal[np.newaxis, :]
    return mean1_normal_2d

def searchinIndex(index, normal_embedding):
    """
    Search for the closest audio vectors in the animals.index file.

    This function searches the FAISS index for the most similar vectors to the given input embedding.

    Parameters:
    index (faiss.Index): The FAISS index to search.
    normal_embedding (np.ndarray): The normalized embedding to search for.

    Returns:
    pd.DataFrame: DataFrame containing distances and indices of the closest vectors.
    """
    D, I = index.search(normal_embedding, index.ntotal)
    r = pd.DataFrame({'distance': D[0], 'index': I[0]})
    return r

def animal_classification(audio_path):
    """
    Classify the species of animals from an audio file.

    This function extracts the audio embedding, searches the index, calculates similarity percentages,
    normalizes the names, and applies Bayes' theorem to determine the most likely animal.

    Parameters:
    audio_path (str): Path to the audio file.

    Returns:
    str: Formatted result with animal classifications and their probabilities.
    """
    query_audio = get_audio_embedding(audio_path)  # Get the audio embedding
    results = searchinIndex(index, query_audio)  # Search the index
    results['percentage'] = results['distance'].apply(exp_negative) * 100  # Calculate the percentage
    results['names'] = results['index'].apply(get_name_from_index)  # Get names from the index
    results['names_normalized'] = results['names'].apply(name_normalisation)  # Normalize the names
    resultat = bayes_theorem(results, 25)
    formatted_result = '\n'.join([f"{animal}: {percentage:.2%}" for animal, percentage in resultat.items()])
    return formatted_result

def add_in_index(audio_path):
    """
    Add a new audio to the index for better classification.

    This function extracts the audio embedding from a new audio file, adds it to the FAISS index,
    updates the index file, and appends the name to the names list.

    Parameters:
    audio_path (str): Path to the audio file to be added.

    Returns:
    str: Confirmation message indicating the addition was successful.
    """
    new_audio = get_audio_embedding(audio_path)
    index.add(new_audio)
    faiss.write_index(index, index_path)
    file_name = os.path.basename(audio_path)
    names.append(file_name)
    result = "L'ajout a bien effectué"
    with open(chemin_noms_animaux, 'w') as fichier:
        # Write each name to the file, formatted as a Python list element
        for nom in names:
            fichier.write(f"'{nom}',\n")
    return result

# Create the graphical interface
interface = gr.Interface(fn=animal_classification, inputs="file", outputs="text")

# Launch the interface
interface.launch()

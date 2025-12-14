import torch
import yaml
import numpy as np
import soundfile as sf
from scipy.signal import resample

import os
import random
import re

from glob import glob

def circulant_matrix(n, m, eps=1e-2, dtype=torch.float32, device='cpu'):
	mat = torch.full((n, m), eps, dtype=dtype, device=device)
	if n < m:
		for i in range(n):
			for j in range(m // n):
				mat[i, i + j * n] = 1.0
		for i in range(m % n):
			mat[i, i + n * (m // n)] = 1.0
	else:
		for j in range(m):
			for i in range(n // m):
				mat[j + m * i, j] = 1.0
		for j in range(n % m):
			mat[j + m * (n // m), j] = 1.0
	return mat

def compute_snr(signal1, signal2):
	assert signal1.shape == signal2.shape, f"Both signals has to be of the same shape. Shape signal1 : {signal1.shape}, signal2 {signal2.shape}"
	power_signal1 = np.std(signal1) ** 2
	power_signal2 = np.std(signal2) ** 2
	return 10 * np.log10(power_signal1 / power_signal2)

def read_audio_array(wav_files, dtype=None, fs_target=None, target_rms=None, max_duration_sec=None):
	signals = []
	for f in wav_files:
		signal = read_audio(f, dtype, fs_target, target_rms, max_duration_sec)
		signals.append(signal)
	return signals

def read_audio(file, dtype=None, fs_target=None, target_rms=0.1, max_duration_sec=None):
	signal, fs = sf.read(file)

	if dtype is not None:
		signal = signal.astype(dtype)

	# Normalizing
	signal -= np.mean(signal)

	if target_rms is not None:
		signal *= (target_rms / (np.std(signal) + 1e-8))

	if fs_target is not None and fs != fs_target:
		# Resample
		print(f"File {file} has samlpling rate of {fs} != {fs_target}")
		duration = len(signal) / fs
		num_samples_target = int(duration * fs_target)
		signal = resample(signal, num_samples_target).astype(dtype)
	
	if max_duration_sec is not None:
		num_samples_target = int(fs * max_duration_sec)
		if len(signal) > num_samples_target:
			random_start = np.random.randint(0, len(signal) - num_samples_target)
			signal = signal[random_start : random_start + num_samples_target]

	return signal

def load_config(config_path):
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)

def get_recursive_path(dir, extensions, shuffle=False, rng=None):
	"""
	Recursively find all files with the specified extensions in a directory and its subdirectories.

	Args:
		dir (str): The root directory to search in
		extensions (list): List of file extensions to look for (e.g., ['.flac', '.mp3'])

	Returns:
		list: A list of paths to all matching files
	"""
	paths = []

	for root, _, files in os.walk(dir):
		for file in files:
			if any(file.lower().endswith(ext.lower()) for ext in extensions):
				paths.append(os.path.join(root, file))
	
	if shuffle:
		if rng is not None:
			rng.shuffle(paths)
		else:
			random.shuffle(paths)
		
	return paths

############################ Metrics ############################

def si_sdr_components(s_hat, s, n):
	# s_target
	alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
	s_target = alpha_s * s

	# e_noise
	alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
	e_noise = alpha_n * n

	# e_art
	e_art = s_hat - s_target - e_noise
	
	return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
	s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

	si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
	si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
	si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

	return si_sdr, si_sir, si_sar

############################### Plot metrics ###########################

import json
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

def analyze_sdr_results(folder_pattern, title=None, base_path='.', figsize=(12, 8), vmin=None, vmax=None, center=None, saving_dir=None):
	"""
	Analyse les résultats SDR des expériences FastMNMF et affiche un tableau coloré.
	
	Parameters:
	-----------
	base_path : str
		Chemin vers le dossier contenant les dossiers d'expériences
	figsize : tuple
		Taille de la figure (largeur, hauteur)
	vmin : float, optional
		Valeur minimale pour l'échelle de couleur. Si None, utilise le minimum des données
	vmax : float, optional
		Valeur maximale pour l'échelle de couleur. Si None, utilise le maximum des données
	center : float, optional
		Valeur centrale pour l'échelle de couleur. Si None, utilise la moyenne des données
	
	Returns:
	--------
	pandas.DataFrame
		DataFrame contenant les résultats SDR organisés
	"""
	
	# Initialiser les listes pour stocker les données
	results = []
		
	# Parcourir tous les dossiers
	for folder_name in os.listdir(base_path):
		if folder_name.startswith(folder_pattern):
			folder_path = os.path.join(base_path, folder_name)

			if os.path.isdir(folder_path):
				# Extraire les informations du nom du dossier
				parts = folder_name.split('_')
				nfft_part = [p for p in parts if p.startswith('nfft')][0]
				nfft_value = int(nfft_part.replace('nfft', ''))
				init_method = parts[-1]  # circular ou gradual
				if init_method in ['circular', 'gradual']:
					init_method = init_method[0].capitalize()
				
					# Parcourir les sous-dossiers fastmnmf1 et fastmnmf2
					for algo_folder in ['fastmnmf1', 'fastmnmf2']:
						algo_path = os.path.join(folder_path, algo_folder)
						algo_method = algo_folder.replace('fastmnmf', 'F')
						if os.path.isdir(algo_path):
							# Chercher le fichier JSON
							json_files = [f for f in os.listdir(algo_path) if f == 'results.json']
							if json_files:
								json_path = os.path.join(algo_path, json_files[0])
								
								try:
									with open(json_path, 'r') as f:
										data = json.load(f)
										sdr_value = data.get('SI-SDR', None)
										
										if sdr_value is not None:
											results.append({
												'nfft': nfft_value,
												'init_method': init_method,
												'algorithm': algo_method,
												'SDR': round(sdr_value, 2)
											})
								except (json.JSONDecodeError, FileNotFoundError) as e:
									print(f"Erreur lors de la lecture de {json_path}: {e}")
	
	if not results:
		print("Aucun résultat trouvé. Vérifiez le chemin et la structure des dossiers.")
		return None
	
	# Créer le DataFrame
	df = pd.DataFrame(results)
	
	# Créer le tableau pivot pour l'affichage
	# Colonnes: nfft, Index: (init_method, algorithm)
	pivot_table = df.pivot_table(
		index=['init_method', 'algorithm'], 
		columns='nfft', 
		values='SDR', 
		aggfunc='first'
	)

	# Calculer la taille de police proportionnelle à la taille de la figure
	# Formule : taille_base * facteur basé sur la surface de la figure
	base_font_size = 28
	figure_area = figsize[0] * figsize[1]
	# Facteur de proportionnalité (ajustable selon vos préférences)
	font_scale_factor = (figure_area / 96) ** 0.5  # 96 = 12*8 (taille de référence)
	font_size = base_font_size * font_scale_factor
	
	# Afficher le tableau avec couleurs
	plt.figure(figsize=figsize)
	
	# Créer une colormap personnalisée (du rouge au vert)
	cmap = plt.cm.RdYlGn
	
	# Créer la heatmap avec contrôle de l'échelle
	if vmin is None:
		vmin = pivot_table.values.min()
	if vmax is None:
		vmax = pivot_table.values.max()
	if center is None:
		center = pivot_table.values.mean()
	
	csfont = {'fontname':'DejaVu Serif'}
	ax = sns.heatmap(
		pivot_table, 
		annot=True, 
		fmt='.2f', 
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		center=center,
		cbar=False,
		cbar_kws={'label': 'SDR [dB]'},
		linewidths=0.5,
		linecolor='white',
		annot_kws={'size': font_size, 'weight': 'normal', **csfont},
	)

	ax.set_aspect('equal')	# square grid cell

	# Personnaliser le graphique
	if title is not None:
		plt.title(title, fontsize=font_size, fontweight='bold', pad=20, **csfont)
	plt.xlabel('Window size (points)', fontsize=font_size, fontweight='normal', **csfont)
	plt.ylabel('Initialisation / Algorithme', fontsize=font_size, fontweight='normal', **csfont)
	
	# Améliorer l'affichage des labels avec taille proportionnelle
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=font_size, **csfont)
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size, **csfont)
	
	# Ajuster la taille de police de la colorbar
	# cbar = ax.collections[0].colorbar
	# cbar.ax.tick_params(labelsize=font_size)
	# cbar.set_label('SDR [dB]', fontsize=label_font_size)
	
	plt.tight_layout()
	
	# Save Heatmap
	if saving_dir is not None:
		saving_path = os.path.join(saving_dir, 'SDR.png')
		plt.savefig(saving_path, dpi=300, bbox_inches='tight')

	plt.show()
	
	# print(f"\n=== ÉCHELLE DE COULEUR UTILISÉE ===")
	# print(f"Valeur min (vmin): {vmin:.2f}")
	# print(f"Valeur max (vmax): {vmax:.2f}")
	# print(f"Valeur centre: {center:.2f}")
	
	# print("\n=== RÉSUMÉ DES RÉSULTATS SDR ===")
	# print(f"Nombre total d'expériences: {len(df)}")
	# print(f"Valeur SDR moyenne: {df['SDR'].mean():.2f}")
	# print(f"Valeur SDR médiane: {df['SDR'].median():.2f}")
	# print(f"Valeur SDR min: {df['SDR'].min():.2f}")
	# print(f"Valeur SDR max: {df['SDR'].max():.2f}")
	
	# print("\n=== TABLEAU DÉTAILLÉ ===")
	# print(pivot_table.to_string())
	
	# Retourner le DataFrame pour utilisation ultérieure
	return df

# Fonction alternative pour un affichage plus compact
def display_sdr_compact_table(folder_pattern, extra_title='', base_path='.', vmin=-5, vmax=15, center=None, figsize=(9,6)):
	"""
	Version compacte qui affiche seulement le tableau sans graphique.
	"""
	df = analyze_sdr_results(folder_pattern=folder_pattern,
						  extra_title=extra_title,
						  base_path=base_path,
						  figsize=figsize,
						  vmin=vmin,
						  vmax=vmax,
						  center=center
						  )
	if df is not None:
		pivot_table = df.pivot_table(
			index=['init_method', 'algorithm'], 
			columns='nfft', 
			values='SDR', 
			aggfunc='first'
		)
		return pivot_table
	return None

# Fonction pour comparer les algorithmes
def compare_algorithms(base_path='.'):
	"""
	Compare les performances des deux algorithmes FastMNMF.
	"""
	df = analyze_sdr_results(base_path)
	if df is not None:
		comparison = df.groupby(['nfft', 'init_method', 'algorithm'])['SDR'].first().unstack('algorithm')
		comparison['Différence (MNMF2-MNMF1)'] = comparison['fastmnmf2'] - comparison['fastmnmf1']
		
		print("\n=== COMPARAISON FASTMNMF1 vs FASTMNMF2 ===")
		print(comparison.round(2).to_string())
		
		return comparison
	return None


def analyze_sdr_results_fastsgmse(folder_pattern, x_param='learning_rate', y_param='n_iterations',
								  algo=None, title=None, base_path='.', figsize=(12, 8), 
								  vmin=None, vmax=None, center=None, saving_dir=None):
	"""
	Analyse les résultats SI-SDR des expériences FastSGMSE et affiche une heatmap flexible.
	
	Parameters:
	-----------
	folder_pattern : str
		Pattern pour filtrer les dossiers d'expériences
	x_param : str
		Paramètre à afficher en colonnes. Options: 'learning_rate', 'n_iterations', 'nfft', 'init_method', 'algorithm'
	y_param : str
		Paramètre à afficher en lignes. Options: 'learning_rate', 'n_iterations', 'nfft', 'init_method', 'algorithm'
	base_path : str
		Chemin vers le dossier contenant les dossiers d'expériences
	figsize : tuple
		Taille de la figure (largeur, hauteur)
	vmin : float, optional
		Valeur minimale pour l'échelle de couleur
	vmax : float, optional
		Valeur maximale pour l'échelle de couleur
	center : float, optional
		Valeur centrale pour l'échelle de couleur
	saving_dir : str, optional
		Dossier pour sauvegarder les résultats
		
	Returns:
	--------
	pandas.DataFrame
		DataFrame contenant tous les résultats extraits
	"""
	import re
	
	# Paramètres disponibles
	available_params = ['learning_rate', 'n_iterations', 'nfft', 'init_method', 'algorithm', 'SNR']
	
	if x_param not in available_params or y_param not in available_params:
		print(f"Paramètres disponibles: {available_params}")
		return None
		
	if x_param == y_param:
		print("Les paramètres x et y doivent être différents")
		return None
	
	# Initialiser les listes pour stocker les données
	results = []
	
	# Parcourir tous les dossiers
	for folder_name in os.listdir(base_path):
		if folder_name.startswith(folder_pattern):
			folder_path = os.path.join(base_path, folder_name)
			if os.path.isdir(folder_path):
				# Extraire les informations du nom du dossier
				parts = folder_name.split('_')
				
				# Extraire nfft
				nfft_parts = [p for p in parts if p.startswith('nfft')]
				if nfft_parts:
					nfft_value = int(nfft_parts[0].replace('nfft', '').replace('g', ''))
				else:
					nfft_value = None
				
				# Extraire learning rate
				lr_parts = [p for p in parts if p.startswith('lr')]
				if lr_parts:
					lr_match = re.search(r'lr(.+)', lr_parts[0])
					if lr_match:
						try:
							learning_rate = float(lr_match.group(1))
						except ValueError:
							learning_rate = None
					else:
						learning_rate = None
				else:
					learning_rate = None

				# Extraire nombre d'itérations
				niter_parts = [p for p in parts if p.startswith('niter')]
				if niter_parts:
					n_iterations = int(niter_parts[0].replace('niter', ''))
				else:
					n_iterations = None
				
				# Extraire la méthode d'initialisation
				init_method = None
				if any('g_' in part for part in parts):
					init_method = 'Gradual'
				elif any('c_' in part for part in parts):
					init_method = 'Circular'
				else:
					# Fallback sur la dernière partie
					last_part = parts[-1]
					if 'circular' in last_part.lower():
						init_method = 'Circular'
					elif 'gradual' in last_part.lower():
						init_method = 'Gradual'
					else:
						init_method = 'Unknown'
				
				# Extraire nombre d'itérations
				SNR_parts = [p for p in parts if p.startswith('snr')]
				if SNR_parts:
					snr_value = float(SNR_parts[0].replace('snr', ''))
				else:
					snr_value = None
				
				# Parcourir les sous-dossiers fastmnmf1 et fastmnmf2
				for algo_folder in os.listdir(folder_path):
					algo_path = os.path.join(folder_path, algo_folder)
					algo_method = algo_folder
					# algo_method = algo_folder.replace('fastmnmf', 'F')
					
					if os.path.isdir(algo_path):
						# Chercher le fichier JSON
						json_files = [f for f in os.listdir(algo_path) if f == 'results.json']
						if json_files:
							json_path = os.path.join(algo_path, json_files[0])
							try:
								with open(json_path, 'r') as f:
									data = json.load(f)
								sdr_value = data.get('SI-SDR', None)
								
								if sdr_value is not None:
									results.append({
										'learning_rate': learning_rate,
										'n_iterations': n_iterations,
										'nfft': nfft_value,
										'init_method': init_method,
										'algorithm': algo_method,
										'SI-SDR': round(sdr_value, 2),
										'SNR': snr_value,
										'folder_name': folder_name  # Pour debugging
									})
							except (json.JSONDecodeError, FileNotFoundError) as e:
								print(f"Erreur lors de la lecture de {json_path}: {e}")
	
	if not results:
		print("Aucun résultat trouvé. Vérifiez le chemin et la structure des dossiers.")
		return None
	
	# Créer le DataFrame
	df = pd.DataFrame(results)

	if algo is not None:
		df = df[df['algorithm'] == algo]
		
	# Créer le tableau pivot
	pivot_table = df.pivot_table(
		index=y_param,
		columns=x_param,
		values='SI-SDR',
		aggfunc='mean'  # Au cas où il y aurait des doublons
	)

	if x_param == 'algorithm':
		algo_order = ['Mixture', 'fastmnmf1', 'fastmnmf2', 'FastSGMSE1', 'FastSGMSE1-BP','FastSGMSE2', 'FastSGMSE2-BP', 'sgmse_basic', 'Oracle-FastMNMF1', 'Oracle-FastMNMF2']
		existing_algos = [algo for algo in algo_order if algo in pivot_table.columns]
		pivot_table = pivot_table.reindex(columns=existing_algos)
	
	# Calculer la taille de police proportionnelle
	base_font_size = 28
	figure_area = figsize[0] * figsize[1]
	font_scale_factor = (figure_area / 96) ** 0.5
	font_size = base_font_size * font_scale_factor
	
	# Créer la figure
	plt.figure(figsize=figsize)
	
	# Créer une colormap personnalisée
	cmap = plt.cm.RdYlGn
	
	# Calculer les valeurs pour l'échelle de couleur
	if vmin is None:
		vmin = pivot_table.values.min()
	if vmax is None:
		vmax = pivot_table.values.max()
	if center is None:
		center = pivot_table.values.mean()
	
	# Configuration de la police
	csfont = {'fontname': 'DejaVu Serif'}
	
	# Créer la heatmap
	ax = sns.heatmap(
		pivot_table,
		annot=True,
		fmt='.2f',
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		center=center,
		cbar=False,
		cbar_kws={'label': 'SI-SDR [dB]'},
		linewidths=0.5,
		linecolor='white',
		annot_kws={'size': font_size * 0.8, 'weight': 'normal', **csfont},
	)
	
	ax.set_aspect('equal')
	
	# Labels personnalisés selon les paramètres
	param_labels = {
		'learning_rate': 'Learning Rate',
		'n_iterations': 'Number of Iterations',
		'nfft': 'Window size [points]',
		'init_method': 'Initialization method',
		'algorithm': 'Algorithm',
		'SNR': 'SNR'
	}
	
	# Personnaliser le graphique
	if title is not None:
		plt.title(title, fontsize=font_size, fontweight='bold', pad=20, **csfont)
	# else:
	# 	plt.title(f'SI-SDR en fonction de {param_labels[x_param]} et {param_labels[y_param]}', 
	# 			 fontsize=font_size, fontweight='bold', pad=20, **csfont)
	
	plt.xlabel(param_labels[x_param], fontsize=font_size, fontweight='normal', **csfont)
	plt.ylabel(param_labels[y_param], fontsize=font_size, fontweight='normal', **csfont)

	# Définir les labels
	params_rotation = ['learning_rate', 'algorithm']
	ax.set_xticklabels(pivot_table.columns, rotation=45 if x_param in params_rotation else 0, 
					fontsize=font_size * 0.7, ha='right', **csfont)
	ax.set_yticklabels(pivot_table.index, rotation=0, fontsize=font_size * 0.8, 
					va='center', **csfont)
	# ax.set_xticklabels(ax.get_xticklabels(), rotation=45 if x_param in params_rotation else 0, 
	# 				   fontsize=font_size * 0.8, **csfont)
	# ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size * 0.8, **csfont)
	
	# Ajuster la colorbar
	# cbar = ax.collections[0].colorbar
	# cbar.ax.tick_params(labelsize=font_size * 0.8)
	# cbar.set_label('SI-SDR [dB]', fontsize=font_size * 0.9, **csfont)
	
	# plt.tight_layout()
	
	# Sauvegarder si demandé
	if saving_dir is not None:
		if not os.path.exists(saving_dir):
			os.makedirs(saving_dir)
		filename = f'SDR_{x_param}_vs_{y_param}.png'
		saving_path = os.path.join(saving_dir, filename)
		plt.savefig(saving_path, dpi=300, bbox_inches='tight')
		print(f"Graphique sauvegardé: {saving_path}")
	
	plt.show()
	return df


################################################## notebook #######################################################
from IPython.display import Audio, display

def play_waveform(waveform, sr=16000, title="Audio"):
	if isinstance(waveform, torch.Tensor):
		waveform = waveform.detach().cpu().numpy()
	if waveform.ndim == 2 and waveform.shape[0] > 1:
		waveform = waveform.mean(axis=0)  # Convert to mono
	display(Audio(waveform, rate=sr, autoplay=False))

def get_separation(model, 
				   n_fft,
				   hop_size,
				   length,
				   norm_factor=1,
				   spec_reverse_transform=False,
				   ref_mic_idx=0):
	output = model.separated(mic_index=ref_mic_idx)    # (n_src, F, T) or (n_mics, n_src, F, T)
	separated_src_freq = output.cpu()    # (n_src, F, T) or (n_mics, n_src, F, T)

	audio = to_audio(X=separated_src_freq, 
				  n_fft=n_fft,
				  hop_size=hop_size,
				  length=length,
				  norm_factor=norm_factor,
				  spec_reverse_transform=spec_reverse_transform
				  )
	# print('audio')
	# play_waveform(audio[0])
	# play_waveform(audio[1])
	return separated_src_freq, audio

def to_audio(X, n_fft,
			 hop_size,
			 length,
			 norm_factor=1,
			 spec_reverse_transform=False,
			 ref_mic_idx=None):
	"""
    Transform from spectral domain to temporal domain
    inputs:
        X: complex spectrogram (torch.Tensor) of shape (n_src, F, T) or (n_mics, n_src, F, T)
    outputs:
	"""
	device = X.device
	if spec_reverse_transform:
		X = spec_back(X)
	if norm_factor != 1:
		norm_factor = norm_factor.to(device)
	window = torch.hann_window(n_fft).to(device)
	if len(X.shape) == 4:
		# output : (n_mics, n_src, F, T)
		n_mics, n_src, F, T = X.shape
		separated_src = torch.zeros((n_mics, n_src, length))
		for m in range(n_mics):
			separated_src[m] = torch.istft(
				input=X[m],
				n_fft=n_fft,
				hop_length=hop_size,
				window=window,
				length=length,
				center=True
			)	# n_mics, n_src, n_samples
		if ref_mic_idx is not None:
			separated_src = separated_src[ref_mic_idx]
	else:
		# output : (n_src, F, T)
		separated_src = torch.istft(
			input=X,
			n_fft=n_fft,
			hop_length=hop_size,
			window=window,
			length=length,
			center=True
		)	# n_src, n_samples
		if ref_mic_idx is not None:
			assert len(X.shape) == 3, f'len(X.shape) : {len(X.shape)}'
			separated_src = separated_src[ref_mic_idx]

	return separated_src * norm_factor


################################################## SGMSE speech enhancement #######################################################

def get_multichannel_files(folder):
	"""
	Retourne une liste de listes, où chaque sous-liste correspond
	aux 6 canaux d'un même fichier audio CHiME3.
	
	Args:
		folder (str): répertoire racine contenant les fichiers .wav
	"""
	all_files = sorted(glob(os.path.join(folder, "*.wav")))
	
	# On regroupe par "basename" sans l'extension .CHx.wav
	groups = {}
	for f in all_files:
		basename = os.path.basename(f)
		# ex: "F01_22GC010A_STR.CH1.wav" → "F01_22GC010A_STR"
		key = basename.split(".CH")[0]
		groups.setdefault(key, []).append(f)
	
	# On trie les fichiers de chaque groupe pour avoir CH1..CH6
	grouped_files = []
	for key in sorted(groups.keys()):  # tri des groupes par ordre alphabétique
		files_sorted = sorted(groups[key], key=lambda x: int(x.split(".CH")[1].split(".")[0]))
		grouped_files.append(files_sorted)
	
	return grouped_files


def get_multichannel_multisrc_files(folder):
	all_files = sorted(glob(os.path.join(folder, "*.wav")))

	# Groupement par nom de fichier sans channel ni src
	groups = {}
	for f in all_files:
		basename = os.path.basename(f)
		# Exemple: "F01_22GC010A_STR.CH1.src2.wav"
		file_id = basename.split(".CH")[0]   # -> "F01_22GC010A_STR"
		groups.setdefault(file_id, []).append(f)

	grouped_files = []
	for file_id in sorted(groups.keys()):  # tri par fichier
		files = groups[file_id]

		# on regroupe par src
		src_groups = {}
		for f in files:
			basename = os.path.basename(f)
			src_part = basename.split(".src")
			if len(src_part) > 1:
				src_id = int(src_part[1].split(".")[0])  # srcX
			else:
				src_id = 0  # si pas de src explicite
			src_groups.setdefault(src_id, []).append(f)

		# pour chaque src, trier par channel
		src_list = []
		for src_id in sorted(src_groups.keys()):
			chans_sorted = sorted(
				src_groups[src_id],
				key=lambda x: int(os.path.basename(x).split(".CH")[1].split(".")[0])
			)
			src_list.append(chans_sorted)

		grouped_files.append(src_list)

	return grouped_files

def mean_std(data):
	data = data[~np.isnan(data)]
	mean = np.mean(data)
	std = np.std(data)
	return mean, std


################ Refine FastSGMSE ##################################

def spec_fwd(spec, 
			 spec_abs_exponent=0.5,
			 transform_type="exponent",
			 spec_factor=0.15,
			 ):
	if transform_type == "exponent":
		if spec_abs_exponent != 1:
			# only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
			# and introduced numerical error
			e = spec_abs_exponent
			spec = spec.abs()**e * torch.exp(1j * spec.angle())
		spec = spec * spec_factor
	elif transform_type == "log":
		spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
		spec = spec * spec_factor
	elif transform_type == "none":
		spec = spec
	return spec

def spec_back(spec,
			  spec_abs_exponent=0.5,
			  transform_type="exponent",
			  spec_factor=0.15
			  ):
	if transform_type == "exponent":
		spec = spec / spec_factor
		if spec_abs_exponent != 1:
			e = spec_abs_exponent
			spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
	elif transform_type == "log":
		spec = spec / spec_factor
		spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
	elif transform_type == "none":
		spec = spec
	return spec

def pad_spec(Y, mode="zero_pad"):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    if mode == "zero_pad":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0,0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0,0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y)
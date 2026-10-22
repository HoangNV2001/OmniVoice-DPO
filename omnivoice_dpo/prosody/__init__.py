"""Syllable-level prosody embeddings for OmniVoice.

A small encoder reads frame-level prosodic features (F0, voicing, energy, coarse spectrum) of the target audio, pools them
per syllable, quantizes them (FSQ) and adds the result to the text-token embeddings of that syllable. Trained jointly with
the masked-diffusion loss; a zero (dropped) code leaves the input exactly as in the base model.
"""

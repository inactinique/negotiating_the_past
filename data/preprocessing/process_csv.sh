#!/bin/zsh

# Nom du fichier d'entrée
input_file="prompts.csv"
# Nom du fichier de sortie pour la première colonne
output_file="processed_prompts.txt"
# Nom du fichier de sortie pour le JSON
# à n’activer que si la partie raw data vous intéresse 
# json_file="prompts.json"

# Vérification que le fichier existe
if [[ ! -f "$input_file" ]]; then
    echo "Erreur: Le fichier $input_file n'existe pas."
    exit 1
fi

# Traitement de la première colonne comme avant
cut -d',' -f1 "$input_file" | sed '1d' | sed 's/^/**** *type_prompt\n/' > "$output_file"

# Extraction de la deuxième colonne et formatage en JSON
# à n’activer que si la partie raw data vous intéresse 
# echo "[" > "$json_file"
# cut -d',' -f2 "$input_file" | sed '1d' | sed 's/$/,/' | sed '$s/,$//' >> "$json_file"
# echo "]" >> "$json_file"

echo "Traitement terminé."
echo "Première colonne enregistrée dans $output_file"
# echo "Données JSON enregistrées dans $json_file"
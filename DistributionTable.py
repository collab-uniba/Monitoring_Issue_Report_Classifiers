import os
import pandas as pd

# Base Path CSV
base_path = "CSV/"
# Path Mapped files
mapped_folder = os.path.join(base_path, "MAPPED/")

# Inizializza una lista per memorizzare le informazioni di ogni file
file_info_list = []

# Somme totali inizializzate a zero
total_bug_count = 0
total_enhancement_count = 0
total_question_count = 0

# Itera su ogni file nella cartella
for filename_input in os.listdir(mapped_folder):
    # Costruisci il percorso completo del file
    input_csv_filename = os.path.join(mapped_folder, filename_input)

    # Carica il CSV in un DataFrame
    df = pd.read_csv(input_csv_filename)

    # Converte la colonna 'date' in formato datetime considerando il fuso orario
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Estrai l'anno massimo dalla colonna 'date' escludendo i valori NaT (Not a Time)
    max_year = df['date'].dt.year.min(skipna=True)

    # Calcola le occorrenze delle label per l'intero file
    bug_count = (df['label'].str.lower() == 'bug').sum()
    enhancement_count = (df['label'].str.lower() == 'enhancement').sum()
    question_count = (df['label'].str.lower() == 'question').sum()

    # Aggiorna le somme totali
    total_bug_count += bug_count
    total_enhancement_count += enhancement_count
    total_question_count += question_count

    # Aggiungi le informazioni del file alla lista
    file_info_list.append({
        'Jira Name': os.path.splitext(filename_input)[0],
        'Year': int(max_year),
        'Bug': bug_count,
        'Enhancement': enhancement_count,
        'Question': question_count
    })

# Aggiungi una nuova riga alla lista con le somme totali
file_info_list.insert(0, {
    'Jira Name': 'Total',
    'Year': '',  # Puoi lasciarlo vuoto o mettere un valore appropriato
    'Bug': total_bug_count,
    'Enhancement': total_enhancement_count,
    'Question': total_question_count
})

# Crea un DataFrame dalle informazioni della lista
distribution_table = pd.DataFrame(file_info_list)

# Create the 'DISTRIBUTION/' directory if it doesn't exist
os.makedirs("DISTRIBUTION", exist_ok=True)
# Salva la tabella di distribuzione in un nuovo file CSV
distribution_table.to_csv('DISTRIBUTION/distribution_table.csv', index=False)

# Visualizza la tabella di distribuzione
print(distribution_table)
import os
import random
from collections import Counter
from sys import argv

def bilancia_dataset(cartella_input):
    file_list = [f for f in os.listdir(cartella_input) if f.endswith('.txt')]
    
    # 1. Mappatura: ogni file a quali e quanti numeri contiene
    # Esempio: info_file["file1.txt"] = Counter({'0': 5, '1': 2})
    info_file = {}
    conteggio_totale = Counter()

    for nome_file in file_list:
        path_completo = os.path.join(cartella_input, nome_file)
        with open(path_completo, 'r') as f:
            numeri_nel_file = []
            for line in f:
                parti = line.split()
                if parti:
                    numeri_nel_file.append(parti[0])
            
            conteggio_file = Counter(numeri_nel_file)
            info_file[nome_file] = conteggio_file
            conteggio_totale.update(conteggio_file)

    print("Distribuzione istanze iniziale:", dict(conteggio_totale))
    
    # 2. Definiamo il target
    # In un mondo ideale, ogni numero dovrebbe apparire quanto quello meno frequente
    target_istanze = min(conteggio_totale.values())
    print(f"Target istanze per ogni numero: {target_istanze}")
    print("-" * 30)

    # 3. Processo di eliminazione
    # Mischiamo i file per non essere deterministici
    nomi_file_random = list(info_file.keys())
    random.shuffle(nomi_file_random)

    eliminati = 0
    for nome_file in nomi_file_random:
        conteggio_questo_file = info_file[nome_file]
        
        # Decidiamo se eliminare il file:
        # Lo eliminiamo solo se TUTTE le classi contenute nel file superano il target
        puo_essere_eliminato = True
        for numero, freq in conteggio_questo_file.items():
            if conteggio_totale[numero] - freq < target_istanze:
                puo_essere_eliminato = False
                break
        
        if puo_essere_eliminato:
            # Aggiorniamo il conteggio globale e rimuoviamo il file
            for numero, freq in conteggio_questo_file.items():
                conteggio_totale[numero] -= freq
            
            os.remove(os.path.join(cartella_input, nome_file))
            eliminati += 1

    print("Distribuzione istanze finale:", dict(conteggio_totale))
    print(f"File eliminati: {eliminati}")

# Utilizzo:
bilancia_dataset(argv[1])
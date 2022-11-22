import json
import redis
from redis.commands.json.path import Path

nome_file_json = "nobelPrizes.json"
nome_database = 'premi_nobel'


# Apro la connessione col server
r = redis.Redis(db=0)

# Recupero il contenuto del file...
with open(nome_file_json) as data_file: 
    dati_json = json.load(data_file)

lista_json = list(dati_json)

# Questo for va messo solo in fase di debug
for i in range(len(lista_json)):
    lista_json[i]['idNobel'] = i

r.json().set(nome_database, Path.root_path(), dati_json)

# # ... e lo salvo sul database
# lista_json = list(dati_json)

# # Questo for va messo solo in fase di debug
# for i in range(len(lista_json)):
#     r.delete(nome_database+":"+str(i)) 

# for i in range(len(lista_json)):
#     r.json().set(nome_database+":"+str(i), Path.root_path(), lista_json[i])

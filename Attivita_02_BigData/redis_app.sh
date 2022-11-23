function chiedi_di_uscire {
  echo ""
  echo ""
  echo -n "Premi un tasto per continuare, q per uscire: "
  read input_letto
  if [[ $input_letto == 'q' ]]; then
    exit 0
  fi
  clear
}

# Controllo che il client riesca a comunicare col server
redis-cli PING 2> /dev/null > /dev/null
if (( $? == 1 ))
then  
    echo "Server non attivo! Attivarlo col comando"
    echo "redis-stack-server >/dev/null &"
    exit 1
fi

echo "Reset del database..."
python3 import_json.py

# 1. Stampo le chiavi
clear
echo "*****************************************************************"
echo "  Stampa di tutte le chiavi contenute nel database "
echo "*****************************************************************"
redis-cli --raw KEYS '*'
chiedi_di_uscire

# 2. Stampo alcuni valori
echo "*****************************************************************"
echo "  Stampa dei dati del premio Nobel numero 1 in formato JSON "
echo "*****************************************************************"
echo -e "$(redis-cli --raw JSON.GET premi_nobel:1 INDENT "\t" NEWLINE "\n" SPACE " ")"
chiedi_di_uscire


echo "*****************************************************************"
echo "  Anno del premio Nobel numero 1"
echo "*****************************************************************"
redis-cli JSON.GET premi_nobel:1 awardYear
chiedi_di_uscire


echo "*****************************************************************"
echo "  Vincitori del premio Nobel numero 20 "
echo "*****************************************************************"
redis-cli --raw JSON.GET premi_nobel:20 $..knownName
chiedi_di_uscire

# 3. Aggioramento di un valore
echo "*****************************************************************"
echo "  Aggiornamento dell'importo del premio 1"
echo "*****************************************************************"
echo -n "Valore prima dell'aggiornamento: "
redis-cli --raw JSON.GET premi_nobel:1 prizeAmount

redis-cli --raw JSON.SET premi_nobel:1 prizeAmount '150783' >/dev/null

echo -n "Valore dopo l'aggiornamento: "
redis-cli --raw JSON.GET premi_nobel:1 prizeAmount 
chiedi_di_uscire

# 4. Eliminazione di una chiave
echo "*****************************************************************"
echo "  Eliminazione dell'importo del premio 1"
echo "*****************************************************************"
echo -n "Coppia Chiave-Valore prima dell'eliminazione: "
#redis-cli --raw JSON.GET premi_nobel:1 prizeAmount
echo -e "$(redis-cli --raw JSON.GET premi_nobel:1 INDENT "\t" NEWLINE "\n" SPACE " ")" | grep "prizeAmount"

redis-cli --raw JSON.DEL premi_nobel:1 $.prizeAmount > /dev/null

echo "Premio Nobel dopo l'eliminazone:"
echo -e "$(redis-cli --raw JSON.GET premi_nobel:1 INDENT "\t" NEWLINE "\n" SPACE " ")"

exit 0
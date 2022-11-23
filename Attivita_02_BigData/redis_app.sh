function chiedi_di_uscire {
  echo -n "Premi un tasto per continuare, q per uscire: "
  read input_letto
  if [[ $input_letto == 'q' ]]; then
    exit 0
  fi

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
echo "Queste sono le chiavi contenute nel database: "
redis-cli --raw KEYS '*'
chiedi_di_uscire

# 2. Stampo i valori
echo "*******************************************************"
echo "Stampo i dati del premio Nobel 1 in formato JSON: "
echo -e "$(redis-cli --raw JSON.GET premi_nobel:1 INDENT "\t" NEWLINE "\n" SPACE " ")"
chiedi_di_uscire

# Arrivata qui

exit 0
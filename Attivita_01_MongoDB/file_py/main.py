from funzioni_notebook import \
    interrogazione1_cercare_nobel_in_categoria, \
    interrogazione2_cercare_nobel_con_data_anno, \
    interrogazione3_cercare_nobel_ordine_importo_data, \
    interrogazione4_max_min_avg, \
    interrogazione5_stampa_numero_nobel_vinti_per_categoria, \
    interrogazione6_cercare_vincitori_di_piu_nobel, \
    interrogazione1_comandi_non_visti_vincitore_random_ita, \
    interrogazione2_comandi_non_visti_anni_senza_nobel, \
    scrittura1_inserimento_nobel, \
    scrittura2_modifica_dati_vincitore


continuare = True
while (continuare):
    print("*****************************************************************************************")
    print(" 0. Uscire")
    print(" 1. Cercare tutti i vincitori dei premi Nobel di una categoria")
    print(" 2. Cercare un Nobel (con data e anno)")
    print(" 3. Stampare i primi Nobel in ordine (crescente o decrescente) di importo o data")
    print(" 4. Interrogazioni con max, min e avg")
    print(" 5. Quanti Nobel sono stati vinti per ogni categoria?")
    print(" 6. Persone/organizzazioni che hanno vinto piu di un Nobel")
    print(" 7. Stampare i dati di un vincitore random italiano")
    print(" 8. Stampare gli anni in cui almeno una categoria non e' stata assegnati")
    print(" 9. Inserire un premio nobel ed eventualmente il vincitore")
    print("10. Modifica i dati di un vincitore")
    opzione = input(" >> ")

    if opzione == "0":
        break
    elif opzione == "1":
        interrogazione1_cercare_nobel_in_categoria()
    elif opzione == "2":
        interrogazione2_cercare_nobel_con_data_anno()
    elif opzione == "3":
        interrogazione3_cercare_nobel_ordine_importo_data()
    elif opzione == "4":
        interrogazione4_max_min_avg()
    elif opzione == "5":
        interrogazione5_stampa_numero_nobel_vinti_per_categoria()
    elif opzione == "6":
        interrogazione6_cercare_vincitori_di_piu_nobel()
    elif opzione == "7":
        interrogazione1_comandi_non_visti_vincitore_random_ita()
    elif opzione == "8":
        interrogazione2_comandi_non_visti_anni_senza_nobel()
    elif opzione == "9":
        scrittura1_inserimento_nobel()
    elif opzione == "10":
        scrittura2_modifica_dati_vincitore()
    else: 
        print("Opzione non valida!!")
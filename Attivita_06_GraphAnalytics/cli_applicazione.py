#!/usr/bin/env python
# coding: utf-8

# <center>Progetto realizzato da Elena Curti (matr. 185431)
# 
# # Crimes Database
# </center>
# 
# ## Introduzione
# Lo scopo di questo progetto è di rispondere a dei research question attraverso le tecniche di graph analytics. <br>
# Verrà usato un database POLE (Person, Object, Location, Event) contenente informazioni sui dei reati (inventati) commessi a Greater Manchester, UK. <br>
# In particolare si vorrà rispondere principalmente ai seguenti questiti:
# - Chi sono gli agenti di polizia più importanti?
# - Esistono dei gruppi di persone pericolose? Quali sono?
# - Esiste un "collegamento" tra persone con precedenti per spaccio e altre persone pregiudicate?
# 
# Per rispondere a tali quesiti verranno usati tecniche e algorimi diversi, basandosi sui dati presenti nel database.
# 
# ## Requisiti
# E' necessaria l'installazione di Neo4j e del seguente pacchetto python:

# In[1]:

# !pip install py2neo


# ## Descrizione del database
# Il database usato è stato trovato al link: https://github.com/neo4j-graph-examples/pole. <br>
# Ha il seguente schema:

# ![schema](schema.png)

# Sono quindi presenti:
# - I crimini (nodo <i>Crime</i>). Le propietà memorizzate sono: "id", "date", "type", "last_outcome", "charge" e "note". Sono inoltre presenti:
#     - Le prove raccolte (<i>Object</i>), con proprietà "id", "type", "description". Tutti gli oggetti hanno come come type "Evidence".
#     - I veicoli coinvolti (<i>Vehicle</i>) con proprietà "make" (marca), "model", "year", "reg" (targa).
# - Le persone (<i>Person</i>) che hanno compiuto reati o loro conoscenti. Proprietà: "nhs_no" (univoco), "name", "surname" e "age". Sono inoltre presenti dati aggiuntivi in altri nodi:
#     - Numero di telefono (<i>Phone</i>), con propietà "phoneNo"
#     - Le chiamate o i messaggi scambiati (<i>PhoneCall</i>) con proprietà: "call_date", "call_type", "call_duration", "call_time". call_type contiene "SMS" o "CALL". 
#     - L'email (<i>Email</i>) con proprietà "email_address"
# - Le posizioni geografiche che rappresentano i luoghi dei crimini e le abitazioni delle persone. Sono memorizzati con tre nodi (dal luogo più preciso al più generale):
#     - <i>Location</i> con proprietà: "address", "postcode", "longitude", "latitude"
#     - <i>Postcode</i> con proprietà: "code"
#     - <i>Area con</i> proprietà: "areaCode"
# 
#     Il postcode del Regno Unito è formato da due sezioni: l'area e il postcode vero e proprio. Ad esempio nel postcode "M1 1AA", "M1" indica il codice dell'area e "M1 1AA" l'intero codice postale. Il postcode è in genere limitato a una strada o a pochi isolati. L'area copre una città o un quartiere.
# - I poliziotti (<i>Officer</i>) che hanno indagato un crimine. Propietà: "name", "surname", "badge_no", "rank". 
# 
# Per memorizzare le conoscenze tra due persone sono presenti le seguenti relazioni:
# - <i>FAMILY_REL</i>: le persone sono imparentate. E' presente la proprietà "rel_type" contenente "SIBLING" o "PARENT" 
# - <i>KNOWS_LW</i>: le due persone convivono (Lives With)
# - <i>KNOWS_PHONE</i>: tra le due persone è stata effettuata almeno una chiamata o un messaggio.
# - <i>KNOWS_SN</i>: le due persone si conoscono sui social network
# - <i>KNOWS</i>: conoscenza generica, è presente se è presente almeno una tra le precedenti relazioni
# 
# A parte <i>FAMILY_REL</i>, nessun'altra relazione presente nel database contiene delle proprietà.

# ## Operazioni iniziali
# Per il corretto funzionamento del progetto, occorre:
# - Scaricare il seguente file https://github.com/neo4j-graph-examples/pole/blob/main/data/pole-50.dump 
# - Creare in Neo4j un DBMS a partire dal file scaricato. Scegliere come nome <i>neo4j</i> e come password <i>password1234</i>
# - Scaricare nel DBMS i plugin "APOC" e "Graph Data Science Library"
# - Far partire il DBMS
# 
# Eseguire poi le celle di codice, seguendo l'ordine proposto.

# In[2]:
from IPython.core.display_functions import display
def chiedi_di_continuare():
  if input("Premi un tasto qualunque per continuare. Digitare [exit] per uscire... ") == "[exit]":
    exit()

from py2neo import Graph
graph = Graph("bolt://localhost:7687",  auth=("neo4j", "password1234"))
print("Connessione al database eseguita correttamente!")

print("\n\n*****************************************************************************************\n\n")
print("1. Chi sono gli agenti di polizia più importanti?")
# Per trovare gli agenti di polizia più importanti, uso la centrality analysis.

print("\n\n----------------\n\n")

print("1.1. In base al numero dei crimini investigati (cypher query) ")
# Cerco inizialmente gli agenti che hanno investigato su più crimini, usando una cypher query:

# In[3]:


def print_n_results(cq, n=100):
    """ Funzione usata per stampare al massimo n righe dei risultati della query. Senza tale funzione, la stampa verrebbe automaticamente troncata a 3 righe. """
    ris = graph.run(cq)
    ris.sample_size = n
    display(ris)

cq = """ 
MATCH (o:Officer)<-[:INVESTIGATED_BY]-(c:Crime) 
WITH o, count(c) as num_invest
RETURN o.badge_no as badge_number, o.name as nome, o.surname as cognome, o.rank as rango, num_invest
ORDER BY num_invest DESC
LIMIT 5 ;
"""
print_n_results(cq)

chiedi_di_continuare()
# Ai primi posti ci sono quindi Madelon DeSousa che ha indagato in 50 crimini, Cloe	Ings (47 crimini), Kania Notti (46 crimini), Worthy	Nettles (45 crimini) e Olenolin	Klehyn (44 crimini).
print("\n\n----------------\n\n")
print("1.2. In base al numero dei crimini (Page Rank e Betweenness Centrality)")
# Anche stavolta cerco gli stessi dati, usando però gli algoritmi di Page Rank e Betweenness Centrality forniti da gds.

# In[4]:


# Creo la proiezione
if graph.run("call gds.graph.exists('officers')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("officers")')

cq = """
CALL gds.graph.project(
  'officers',    
  ['Officer', 'Crime'],
  {INVESTIGATED_BY: {orientation: 'UNDIRECTED'} }
) ;
"""

display(graph.run(cq))


# In[5]:


# Memory estimation
display(graph.run(""" 
CALL gds.pageRank.write.estimate('officers', { writeProperty: 'pageRank' })
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN "Page Rank" as algoritmo, nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory

UNION ALL

CALL gds.betweenness.write.estimate('officers', { writeProperty: 'betweenness' })
YIELD  nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN "Betweenness" as algoritmo, nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory ; 
""" ))


# In[6]:


# Risultati
print("Page Rank: ")
cq=""" 
CALL gds.pageRank.stream('officers')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score AS pageRank
RETURN n.badge_no as badge_number, n.name as nome, n.surname as cognome, n.rank as rango,pageRank
ORDER BY pageRank DESC
LIMIT 5 ;
"""
print_n_results(cq)


print("Betweenness Centrality: ")
cq=""" 
CALL gds.betweenness.stream('officers')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score AS pageRank
RETURN n.badge_no as badge_number, n.name as nome, n.surname as cognome, n.rank as rango,pageRank
ORDER BY pageRank DESC
LIMIT 5 ;
"""
print_n_results(cq)

chiedi_di_continuare()

# I risultati ottenuti sono ovviamente gli stessi trovati precedentemente.
print("\n\n----------------\n\n")
print("1.3. In base al numero dei crimini indagati, con almeno 1 arresto (Degree centrality)")
# Per stabilire quali agenti siano i più importanti, andrò stavolta a contare i soli crimini che hanno portato all'arresto di almeno una persona. Userò questa volta l'algoritmo Degree centrality.

# In[7]:


# Creo la proiezione
if graph.run("call gds.graph.exists('officers-crime-con-persone')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("officers-crime-con-persone")')

cq=""" 
CALL gds.graph.project.cypher('officers-crime-con-persone',
  'MATCH (o:Officer|Crime)  RETURN DISTINCT id(o) as id',
  'MATCH (o:Officer)<--(c:Crime)<--(:Person) RETURN  id(o) AS source, id(c) as target, "INVESTIGATED_BY" as type',
{validateRelationships:false}
);""" 

display(graph.run(cq))


# In[8]:


# Memory estimation
display(graph.run(""" 
CALL gds.degree.write.estimate('officers-crime-con-persone', { writeProperty: 'betweenness' })
YIELD  nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory ; 
""" ))

# In[9]:


# Seleziono solo gli agenti che hanno arrestato una persona
cq=""" 
CALL gds.degree.stream('officers-crime-con-persone')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
RETURN n.badge_no as badge_number, n.name as nome, n.surname as cognome, n.rank as rango,score
ORDER BY score DESC
LIMIT 5 ;
"""
print_n_results(cq)

chiedi_di_continuare()
# Questa volta, gli agenti migliori sono: Devy Larive (con 3 investigazioni che hanno portato all'arresto di una o più persone), Von Death, Pauline Petrasso, Kara-lynn Ironside e Kris Teaze (tutte con 1 investigazione).

print("\n\n----------------\n\n")
print("1.4. In base al numero di persone arrestate (HITS)")
# Non è detto che ad un crimine abbia partecipato una sola persona. Un agente che ha investigato un solo crimine potrebbe aver portato all'arresto di più persone. Per stabilire gli agenti più importanti, quindi, cerco di capire quali agenti abbiano portato all'arresto di più persone. Userò stavolta l'algoritmo HITS.

# In[10]:


# Creo la proiezione
if graph.run("call gds.graph.exists('officers-arresti')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("officers-arresti")')

cq=""" 
CALL gds.graph.project.cypher('officers-arresti',
  'MATCH (o:Officer|Person)  RETURN DISTINCT id(o) as id',
  'MATCH (o:Officer)<--(c:Crime)<--(p:Person) RETURN distinct id(p) as source, id(o) as target',
{validateRelationships:false}
);""" 

display(graph.run(cq))


# In[11]:


# In base al numero di persone arrestate
cq=""" 
CALL gds.alpha.hits.stream('officers-arresti',  {hitsIterations: 1})
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS n, values as score
RETURN n.badge_no as badge_number, n.name as nome, n.surname as cognome, n.rank as rango,score
ORDER BY score DESC
LIMIT 5 ;
"""
print_n_results(cq)

chiedi_di_continuare()
# Al primo posto troviamo di nuovo Devy Larive con uno score maggiore. In seconda posizione a pari merito ci sono invece Gregorius Shakesby, Simmonds Greensall, Karlyn Calladin e Chet Vasic.

print("\n\n----------------\n\n")
print("1.5. Solo gli agenti di rango maggiore (Degree Centrality)")
# Questa volta cerco gli agenti di rango maggiore che hanno investigato su più crimini. Userò di nuovo il degree centrality. <br>
# Cerco inizialmente tutti i possibili ranghi:

# In[12]:


display(graph.run("MATCH (o:Officer) RETURN COLLECT(DISTINCT o.rank) AS tutti_i_ranghi"))


# Dopo una breve ricerca su Internet (https://en.wikipedia.org/wiki/Police_ranks_of_the_United_Kingdom#Great_Britain) si può capire che il rango maggiore è Chief Inspector. Vado quindi a selezionare solo tali agenti:

# In[13]:


if graph.run("call gds.graph.exists('officers-chief')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("officers-chief")')

cq = """ 
CALL gds.graph.project.cypher('officers-chief',
  'MATCH (o:Officer|Crime) WHERE o.rank="Chief Inspector" OR o.rank IS NULL RETURN DISTINCT id(o) as id',
  'MATCH (o:Officer)<--(c:Crime) RETURN  id(o) AS source, id(c) as target',
  {validateRelationships:false}
);
"""

display(graph.run(cq))


# In[14]:


# Memory estimation
display(graph.run(""" 
CALL gds.degree.write.estimate('officers-chief', { writeProperty: 'betweenness' })
YIELD  nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory ; 
""" ))


# In[15]:


# Agenti di polizia con il rango maggiore (Chief Inspector) che hanno indagato su più crimini
cq=""" 
CALL gds.degree.stream('officers-chief')
YIELD nodeId, score 
WITH gds.util.asNode(nodeId) AS n, score 
RETURN n.badge_no as badge_number, n.name as nome, n.surname as cognome,n.rank as rango, score
ORDER BY score DESC
LIMIT 5 ;
"""
print_n_results(cq)

chiedi_di_continuare()

# Possiamo quindi vedere che gli agenti di rango Chief Inspector che hanno investigato su più crimini sono: Kort Monelli, Urban Stave, Roberto Febry, Evey Rahlof e Dottie Syddie.<br>
# Si può notare anche che questi agenti non sono tra gli agenti stampati al punto 1.1 (dove non è stato tenuto conto del rango). Questo è probabilmente dovuto al fatto che i dati nel database sono casuali.

# ### Conclusioni
# Per rispondere alla domanda iniziale, i migliori agenti sono coloro che sono stati stampati ai punti precedenti.

print("\n\n*****************************************************************************************\n\n")
print("2. Esistono dei gruppi di persone pericolose? Quali sono?")
# Per trovare i gruppi di persone pericolose uso la community detection.

# In[16]:


def print_in_columns(my_list):
    """ Funzione che stampa una lista su due colonne"""
    columns = 2  #if len(my_list)>10 else 3
    len_max = str(max([len(i) for i in my_list]) + 4)
    for i in range(0, len(my_list), columns):
        print("  ", end="")
        [eval('print(f"{i:'+len_max+'}", end="")') for i in my_list[i:i+columns]]
        print("")

def print_id_gruppo(query_results):
  """ Funzione che stampa i gruppi di persone """
  for (communityId, gruppo) in query_results:
      print("---------- Gruppo "+str(communityId)+" ----------")
      # print(gruppo)
      print_in_columns(gruppo)
      print("")


print("\n\n----------------\n\n")
print("2.1. Ricerca in base alle conoscenze generiche dei criminali (Algoritmo di Louvain)")
# In questa prima versione, cerco i gruppi di persone che hanno preso parte ad un crimine e che sono legate dalla relazione generica "KNOWS", senza quindi dare un peso al tipo di conoscenza. Per trovare i gruppi uso l'algoritmo di Louvain.

# In[17]:


# Creo la proiezione
if graph.run("call gds.graph.exists('criminals-knows')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("criminals-knows")')

cq = """ 
  CALL gds.graph.project.cypher('criminals-knows',
  'MATCH (p:Person)-->(:Crime) RETURN id(p) as id',
  'MATCH (criminal:Person)-[:KNOWS]-(conoscente:Person) RETURN DISTINCT id(conoscente) as target, id(criminal) as source',
  {validateRelationships:false}
); 
"""
display(graph.run(cq))


# In[18]:


display(graph.run("""CALL gds.louvain.write.estimate('criminals-knows', { writeProperty: 'community' })
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory ;
"""))


# In[19]:


cq = """ 
CALL gds.louvain.stream('criminals-knows')
  YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
WITH communityId, apoc.coll.sort(collect(node.name + " " + node.surname + " (" + node.nhs_no+")")) AS gruppo
WHERE size(gruppo)>1  
RETURN *
"""
print_id_gruppo(graph.run(cq))

chiedi_di_continuare()
# L'algoritmo ci restituisce quattro gruppi. Analizzo il primo e cerco i conoscenti di Diana:

# In[20]:


print_n_results(""" 
MATCH (diana:Person {nhs_no:"900-41-3309"})-[:KNOWS]-(conoscenti_di_diana:Person)
RETURN conoscenti_di_diana.name, conoscenti_di_diana.surname, conoscenti_di_diana.nhs_no, EXISTS ( (conoscenti_di_diana)-->(:Crime) ) AS ha_precedenti 
""")


# Come si può vedere Kathleen e Jessica compaiono nel gruppo, mentre Kathy e Melissa no. Questo perché Kathleen e Jessica hanno dei precedenti penali mentre Kathy e Melissa no. <br> 
# Cerchiamo di capire ora come mai compaiono anche Raymond e Phillip nel gruppo:

# In[21]:


print_n_results(""" 
MATCH (raymond:Person {nhs_no:"879-22-8665"})-[:KNOWS]-(conoscenti_di_raymond:Person)
RETURN conoscenti_di_raymond.name, conoscenti_di_raymond.surname, conoscenti_di_raymond.nhs_no, EXISTS ( (conoscenti_di_raymond)-->(:Crime) ) AS ha_precedenti 
""")
chiedi_di_continuare()

# Si può notare quindi che Diana e Raymond hanno tra le conoscenze in comune Kathleen, che quindi "unisce" i due gruppi di conoscenti.

print("\n\n----------------\n\n")
print("2.2. Ricerca in base al tipo di conoscenze (Modularity Optimization)")
# Per trovare i gruppi darò stavolta un peso diverso alle diverse tipologie di conoscenze. Ho deciso di dare più importanza ai conviventi (KNOWS_LW); in secondo luogo ai parenti (FAMILY_REL), poi alle persone che si sono almeno chiamate o scambiate messaggi (KNOWS_PHONE) ed infine a coloro che si conoscono sui social network (KNOWS_SN). Userò questa volta la modularity optimization.

# In[22]:


## Creo il project
if graph.run("call gds.graph.exists('criminals-knows-pesate')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("criminals-knows-pesate")')

cq = """ 
  CALL gds.graph.project.cypher('criminals-knows-pesate',
   'MATCH (p:Person)-->(:Crime) RETURN id(p) as id',
   'MATCH (criminal:Person)-[]-(conoscente:Person)
    RETURN id(conoscente) as target,
    CASE
        WHEN EXISTS ( (conoscente)-[:KNOWS_LW]-(criminal) ) THEN 10
        WHEN EXISTS ( (conoscente)-[:FAMILY_REL]-(criminal) ) THEN 8
        WHEN EXISTS ( (conoscente)-[:KNOWS_PHONE]-(criminal) ) THEN 4
        WHEN EXISTS ( (conoscente)-[:KNOWS_SN]-(criminal) ) THEN 3
    END AS peso, id(criminal) as source',
  {validateRelationships:false}
); 
"""
display(graph.run(cq))


# In[23]:


cq = """ 
CALL gds.beta.modularityOptimization.stream('criminals-knows-pesate', {relationshipWeightProperty: 'peso' })
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
WITH communityId, apoc.coll.sort(collect(node.name + " " + node.surname + " (" + node.nhs_no+")")) AS gruppo
WHERE size(gruppo)>1  
RETURN *
"""
print_id_gruppo(graph.run(cq))

chiedi_di_continuare()
# L'unica differenza dal risultato precedente è che il gruppo 14 e 28 prima era unito. Questo perché probabilmente ci sono conoscenze più significative tra gli individui di un gruppo rispetto ai componendi dell'altro grupo.

print("\n\n----------------\n\n")
print(" 2.3. Aggiungo gli abitanti della stessa città (Weakly Connected Components)")

# Come si può vedere esistono persone senza un legame di conoscenza che però abitano vicine. 

# In[24]:


display(graph.run(""" 
MATCH (p1:Person)-->(:Location)-->(:Area)<--(:Location)<--(p2:Person) 
WHERE NOT EXISTS ( (p1)-[]-(p2))
RETURN COUNT(*) AS num_persone_che_non_si_conoscono_ma_che_abitano_vicine """))


# Aggiungo quindi questa possibile conoscenza, tra le conoscenze già presenti, con un peso basso. Userò stavolta l'algoritmo Weakly Connected Components.

# In[25]:


# Creo il project
if graph.run("call gds.graph.exists('criminals-knows-pesate-con-vicini')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("criminals-knows-pesate-con-vicini")')

cq = """ 
  CALL gds.graph.project.cypher('criminals-knows-pesate-con-vicini',
   'MATCH (p:Person)-->(:Crime) RETURN id(p) as id',
   'MATCH (criminal:Person),(conoscente:Person)
    WHERE criminal<>conoscente
    RETURN id(conoscente) as target,
    CASE
        WHEN EXISTS ( (conoscente)-[:KNOWS_LW]-(criminal) ) THEN 10
        WHEN EXISTS ( (conoscente)-[:FAMILY_REL]-(criminal) ) THEN 8
        WHEN EXISTS ( (conoscente)-[:KNOWS_PHONE]-(criminal) ) THEN 4
        WHEN EXISTS ( (conoscente)-[:KNOWS_SN]-(criminal) ) THEN 3
        WHEN EXISTS ( (criminal)-->(:Location)-->(:Area)<--(:Location)<--(conoscente) ) THEN 1
    END AS peso, id(criminal) as source',
  {validateRelationships:false}
); 
"""
display(graph.run(cq))


# In[26]:


display(graph.run(""" 
CALL gds.wcc.write.estimate('criminals-knows-pesate-con-vicini', { writeProperty: 'component' })
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
"""))

# In[27]:


cq = """ 
CALL gds.wcc.stream('criminals-knows-pesate-con-vicini', { relationshipWeightProperty: 'peso' })
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS node, componentId
WITH componentId, apoc.coll.sort(collect(node.name + " " + node.surname + " (" + node.nhs_no+")")) AS gruppo
WHERE size(gruppo)>1  
RETURN *
"""
print_id_gruppo(graph.run(cq))

chiedi_di_continuare()

# L'algoritmo restituisce un unico gruppo di 29 persone, contenente quindi un potenziale gruppo di persone pericolose.

print("\n\n----------------\n\n")
print("2.4. Aggiunta dei conoscenti dei criminali (Strongly Connected Components)")
# Proverò questa volta a formare i gruppi includendo anche i conoscenti dei criminali, che potrebbero non aver commesso reati, ma che potrebbero comunque essere considerarti  pericolosi. Non darò pesi alle conoscenze. Userò stavolta l'algoritmo Strongly Connected Components.

# In[28]:


# Creo il project
if graph.run("call gds.graph.exists('criminals-conoscenti')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("criminals-conoscenti")')

cq = """ 
  CALL gds.graph.project.cypher('criminals-conoscenti',
  ' MATCH (p:Person) 
    WHERE 
        EXISTS ( (p)-->(:Crime) )                             // Persone che hanno preso parte a un crimine
        OR EXISTS ( (p)-[]-(:Person)-[:PARTY_TO]->(:Crime) )  // Conoscenti dei criminali
    RETURN id(p) as id',
  
  'MATCH (criminal)-[:KNOWS]-(conoscente:Person) RETURN DISTINCT id(conoscente) as target, id(criminal) as source',
  {validateRelationships:false}
); 
"""
display(graph.run(cq))
print("Proiezione crata correttamente!")


# In[29]:


cq = """ 
CALL gds.alpha.scc.stream('criminals-conoscenti')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS node, componentId
WITH componentId, apoc.coll.sort(collect(node.name + " " + node.surname + " (" + node.nhs_no+")")) AS gruppo
WHERE size(gruppo)>1  
RETURN *
"""
print_id_gruppo(graph.run(cq))

chiedi_di_continuare()

# L'algoritmo restituisce un unico gruppo contenente le persone restituite precedentemente ed i loro conoscenti, indicando quindi la presenza di una relazione tra essi. <br>
# Questo fornisce quindi una risposta ancora più precisa alla seconda research question.

print("\n\n*************************************************************************************************************\n\n")
print("3. Esiste un collegamento tra persone con precedenti per spaccio e altre persone pregiudicate?")
# Cercherò stavolta i possibili collegamenti tra le persone con precedenti per spaccio e altre persone pregiudicate. Per trovare tali collegamenti userò gli algoritmi di Shortest Path. <br>
# 
# Prendo intanto due persone: una coinvolta in un crimine di tipo Drugs e l'altra pregiudicata ma che non abbia mai compiuto un crimine Drugs. Queste due persone inoltre non devono conoscersi direttamente, perché altrimenti il loro nome sarà sicuramente già presente nell'elenco al punto 2.

# In[30]:


cq = """ 
MATCH (p1:Person)-->(c1:Crime {type:"Drugs"}), (p2:Person)-->(c2:Crime)
WHERE 
    NOT EXISTS ( (p2)-->(:Crime {type:"Drugs"} )) 
    AND NOT EXISTS ( (p1)-[]-(p2)) 
RETURN p1.nhs_no AS spacciatore_nhs_no, p1.name AS spacciatore_name, p2.nhs_no AS pregiudicato_nhs_no, p2.name AS pregiudicato_name
LIMIT 1
"""
display(graph.run(cq))

data = graph.run(cq).data()[0]
spacciatore_nhs_no = data["spacciatore_nhs_no"]
pregiudicato_nhs_no = data["pregiudicato_nhs_no"]


# Creo ora una proiezione con pesi diversi, in base alla relazione che unisce due nodi. In questo modo riesco a trovare il collegamento più importante che unisce due persone. Tutte le relazioni in ordine di importanza sono:
# - CURRENT_ADDRESS, OCCURRED_AT, INVOLVED_IN, PARTY_TO, KNOWS_LW. In questo modo do più importanza alle persone conviventi, ai crimini, ai luoghi dei crimini e alle abitazioni delle persone.
# - HAS_POSTCODE, FAMILY_REL. In questo modo do un'importanza secondaria (ma comunque alta) ai parenti e alle persone che vivono nella stessa strada.
# - POSTCODE_IN_AREA, LOCATION_IN_AREA. In questo modo do un'importanza leggermente minore alle persone che vivono nella stessa città.
# - HAS_PHONE, KNOWS_PHONE, CALLER, CALLED. Do quindi meno importanza alle persone che si conoscono solo per uno scambio di chiamate o messaggi.
# - KNOWS_SN. Il legame meno importante è quello tra i social network.
# 
# Si fa notare che a differenza dei casi al punto 2, in questo caso più l'importanza di una relazione cresce, più il peso associato diminuisce. Questo perché stavolta vengono usati algoritmi di Shortest Path che cercano il percorso con peso minore.

# In[31]:


# Creo la proiezione
if graph.run("call gds.graph.exists('drugs-groups')").data()[0]["exists"]:
  graph.run('call gds.graph.drop("drugs-groups")')

cq = """ 
  CALL gds.graph.project.cypher(
    'drugs-groups',
    'MATCH (p) RETURN id(p) as id',
    'MATCH (a)-[r]-(b)
     RETURN id(a) as source, id(b) as target, 
      CASE

        WHEN EXISTS ( (a)-[:CURRENT_ADDRESS]-(b) ) THEN 1
        WHEN EXISTS ( (a)-[:OCCURRED_AT]-(b) ) THEN 1
        WHEN EXISTS ( (a)-[:INVOLVED_IN]-(b) ) THEN 1
        WHEN EXISTS ( (a)-[:PARTY_TO]-(b) ) THEN 1        
        WHEN EXISTS ( (a)-[:KNOWS_LW]-(b) ) THEN 1
        
        WHEN EXISTS ( (a)-[:FAMILY_REL]-(b) ) THEN 2
        WHEN EXISTS ( (a)-[:HAS_POSTCODE]-(b) ) THEN 2

        WHEN EXISTS ( (a)-[:POSTCODE_IN_AREA]-(b) ) THEN 3
        WHEN EXISTS ( (a)-[:LOCATION_IN_AREA]-(b) ) THEN 3
        
        WHEN EXISTS ( (a)-[:HAS_PHONE]-(b) ) THEN 4
        WHEN EXISTS ( (a)-[:KNOWS_PHONE]-(b) ) THEN 4
        WHEN EXISTS ( (a)-[:CALLER]-(b) ) THEN 4
        WHEN EXISTS ( (a)-[:CALLED]-(b) ) THEN 4

        WHEN EXISTS ( (a)-[:KNOWS_SN]-(b) ) THEN 5
        
        ELSE 10
      END AS peso, type(r) AS type',
    {validateRelationships:false}
);  
"""
display(graph.run(cq))


# In[32]:


# Memory estimation
print_n_results(""" 
MATCH (source:Person {nhs_no: '"""+spacciatore_nhs_no+"""'}), (target:Person {nhs_no: '"""+pregiudicato_nhs_no+"""'})
CALL gds.shortestPath.dijkstra.write.estimate('drugs-groups', {sourceNode: source, targetNode: target, writeRelationshipType: 'PATH'})
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN "Dijkstra senza pesi" as algoritmo, nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory

UNION ALL

MATCH (source:Person {nhs_no: '"""+spacciatore_nhs_no+"""'})
CALL gds.allShortestPaths.delta.write.estimate('drugs-groups', {sourceNode: source, relationshipWeightProperty: 'peso', writeRelationshipType: 'PATH' })
YIELD nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory
RETURN "All Shortest Paths con pesi" as algoritmo, nodeCount, relationshipCount, bytesMin, bytesMax, requiredMemory

""" )


# In[33]:


def print_results_shortest_path(cq):
    """ Funzione che stampa i risultati dello shortest path in modo piu' leggibile """
    lista_data = graph.run(cq).data()
    for data in lista_data:
        print("---------------------")
        print("Costo totale:", data["totalCost"])
        print("Costo ad ogni nodo:", data["costs"])
        print("Shortest Path:")

        ordine_nodi = data["ordine_nodi"]
        from pprint import pprint 
        for nodo in ordine_nodi:
            text = ""
            tipo_nodo = str(nodo.labels)[1:]
            diz = dict(nodo)
            if tipo_nodo == "Person":
                text = diz["name"]+ " " + diz["surname"] + " (" +diz["nhs_no"]+")"
            elif tipo_nodo == "Crime":
                text = "Reato del " + diz["date"] + ". Tipo: "+diz["type"]+". Esito: "+ diz["last_outcome"] 
            elif tipo_nodo == "Location" or tipo_nodo == "Area":
                text = tipo_nodo + " = " +  str(diz)[1:-1].replace("'", "", -1)
            else:
                text = nodo
            print("\t", text)
        

print("\n\n----------------\n\n")
print("3.1. Trovo il collegamento usando i pesi (All Shortest Path)")
# Cercherò innanziutto un possibile collegamento tra le persone in base ai pesi dati precedentemente. Userò l'algoritmo All Shortest Path.

# In[34]:


cq = """ 
MATCH (source:Person {nhs_no: '"""+spacciatore_nhs_no+"""'}), (target:Person {nhs_no: '"""+pregiudicato_nhs_no+"""'})
CALL gds.allShortestPaths.delta.stream( 'drugs-groups', {sourceNode: source, relationshipWeightProperty: 'peso'})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
WITH targetNode, totalCost, [nodeId IN nodeIds | gds.util.asNode(nodeId)] AS ordine_nodi, costs
WHERE targetNode = id(target)
RETURN totalCost, ordine_nodi, costs; 
"""
# print(cq)
print_results_shortest_path(cq)

chiedi_di_continuare()
# ![all_shortest_path_con_pesi](all_shortest_path_con_pesi.png)

print("\n\n----------------\n\n")
print("3.2. Trovo il collegamento senza usare i pesi (Dijkstra)")
# Proverò stavolta a cercare un collegamento tra Raymond e Stephanie senza usare i pesi, considerando quindi come distanza il numero di nodi. Userò l'algoritmo di Dijkstra.

# In[35]:


cq = """ 
MATCH (source:Person {nhs_no: '"""+spacciatore_nhs_no+"""'}), (target:Person {nhs_no: '"""+pregiudicato_nhs_no+"""'})
CALL gds.shortestPath.dijkstra.stream( 'drugs-groups', {sourceNode: source, targetNode: target})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
WITH totalCost, [nodeId IN nodeIds | gds.util.asNode(nodeId)] AS ordine_nodi, costs 
RETURN totalCost, ordine_nodi, costs; 
"""
print_results_shortest_path(cq)
    
chiedi_di_continuare()

# ![dijkstra_senza_pesi](dijkstra_senza_pesi.png)

# Non sembra esserci quindi un collegamento "importante" tra Raymond e Stephanie:
# - Il primo risultato mostra che un familiare di Raymond ha compiuto un reato nella stessa città in cui Stephanie ha compiuto un reato.
# - Il secondo invece mostra un collegamento "distante" tante persone.
# 

print("\n\n----------------\n\n")
print("3.3. Collegamento tra due persone generiche (Dijkstra)")
# Cercherò questa volta un collegmanto tra una generica persona con precedenti per droga e una generica persona con precedenti. Andrò poi a selezionare le persone che hanno il collegamento più corto:

# In[36]:


cq = """ 
MATCH (source:Person)-->(:Crime {type:"Drugs"}), (target:Person)
WHERE 
    EXISTS ( (target)-->(:Crime) ) 
    AND NOT EXISTS ( (target)-->(:Crime {type:"Drugs"})) 
    AND NOT EXISTS ( (source)-[]-(target)) 
WITH source, COLLECT(DISTINCT id(target)) AS lista_id_target

CALL gds.allShortestPaths.dijkstra.stream('drugs-groups', {
    sourceNode: source,
    relationshipWeightProperty: 'peso'
})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
WHERE targetNode IN lista_id_target
WITH totalCost, [nodeId IN nodeIds | gds.util.asNode(nodeId)] AS ordine_nodi, costs 
RETURN totalCost, ordine_nodi, costs
ORDER BY totalCost 
LIMIT 1
"""
print_results_shortest_path(cq)


# ![shortest_path_generico](./shortest_path_generico.png)

# Questa volta il collegamento è molto più importante: un parente di Raymond (condannato per droga) è il convivente di Jessica (pregiudicata).

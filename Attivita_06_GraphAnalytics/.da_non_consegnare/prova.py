# !pip install git+https://github.com/ybaktir/networkx-neo4j


import nxneo4j as nx
import pandas as pd 
import matplotlib as plt
query = "match (n:Person) return n limit 2"
df = pd.DataFrame(graph.run(query).data())

G=nx.DiGraph()   
G.add_nodes_from(list(set(list(df['child']) + list(df.loc['parent']))))

#Add edges

tuples = [tuple(x) for x in df.values] 
G.add_edges_from(tuples)
G.number_of_edges()

#Perform Graph Drawing
#A star network  (sort of)
nx.draw_networkx(G)
plt.show()

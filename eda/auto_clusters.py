import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def pendiente(x1,y1,x2,y2):
    return (y2 - y1) / (x2 - x1)

def kmeans_auto_clusters(datos, graph = False, verbose = False, prom_line = False, prop_limit = 0.1):
    """
    Kmeans auto clusters:
        Recibe:
        - datos: Un DataFrame de Datos para ejecutar K-means de sklearn.
        - graph: Un Booleano que define si se imprimirá el gráfico del codito.
        - verbose: Un Booleano que define si se imprimirán todos los resultados tomados en cuenta para la elección del número de clusters.
                   Ayudando claramente a comprender no solo como funciona el algoritmo, sino también como mejorar su performance de acuerdo a los datos que está recibiendo.
        - prom_line: Un Booleano que define si se trazará la línea promedio de las inercias.
        - prop_limit: Un Float que define la proporción límite por debajo de la cual se consederará el punto de inflexión.

    Este un algoritmo creado durante el curso de la Maestría en Ciencia de Datos de la Universidad de Sonora.
    Por Viowi Y. Cabrisas Amuedo.

    En Resumen que hace:
        - Calcula las pendientes de cada una de las rectas que se pueden trazar entre los puntos que contiene las siguientes coordenadas:
          (número de clusters; valor de la inercia), de un scatter plot.
        - Calcula las proporciones de cada una de las pendientes con respecto a la suma total de las pendientes analizadas hasta el momento.
        - Revisa si la última pendiente encontrada tiene una proporción por debajo de: prop_limit.
        - Finalmente establece el punto de inflexión un paso antes.
        - Devuelve el número de clusters óptimos y además ya una serie de numpy con las etiquetas del número de clusters óptimos.
    
    Ventajas:
        - Todas las clásicas del Algoritmo de K-Means.
        - Flexibilidad: Adaptable a sets de datos específicos.
        - Se ejecuta el algoritmo de K-Medias solo k + 1 veces, donde k sería el número de clusters óptimos, y no en grandes rangos de ejecución.
        - Economiza recursos, ya que de una vez devuelve una serie con las etiquetas según el k-óptimo número de clusters.
        - Open source y libre de uso. :-)

    Desventajas:
        - Las clásicas, heredadas del algoritmo de K-medias.
        - Las que estén por venir... aún se estaán haciendo pruebas, y depende de que tanto se conoce el set de datos en cuestión.

    Se acepta todo tipo de sugerencias, si ya estuvieran probadas mejor.
    Gracias de antemano, disfruten el código.

    Y citen por cortesía.
    Viowi Y.

    GitHub: https://github.com/viowiy/MediumPubs/tree/main/02%20-%20K-means%20Automatic%20clusters%20detection
    """
    labels_model = {}
    k_list = []
    inertias = []
    pendientes = []
    proporciones = []

    k = 1
    proporcion = 1 # Esta será la proporción de la inercia conque empezaremos...
    flag = 1

    while proporcion > prop_limit: # Proporción límite... por debajo de este valor se descartarán los clústers...
        model = KMeans(n_clusters = k, random_state = 42)
        model.fit(datos)
        inertias.append(model.inertia_)
        inercia_final = inertias[-1] / np.sum(inertias)
        k_list.append(k)
        # print(f"clusters...{k_list}")

        clusters_recomendados = k - 1
        labels_model[k] = model.labels_

        if k >= 2:
            pend = pendiente(k-1, inertias[k-2],k,inertias[k-1])
            pendientes.append(pend)
            proporcion = pend/sum(pendientes)

            if verbose == True:
                proporciones.append(proporcion)
                print(f"Clusters: {k} --- Pendiente actual: {pend} --- Lista: {pendientes} --- Lista de Proporciones: {proporciones}")
        
        k += 1

    if verbose == True:
        print(f"Total de clusters recomendados: {clusters_recomendados}")

    if graph == True:
        # Plot k vs inertias... Gráfico del codito...
        plt.clf()
        
        x_marker = clusters_recomendados
        y_marker = inertias[(clusters_recomendados - 1)]
        # Colocar el marcador en el punto especificado
        plt.scatter(x_marker, y_marker, color='red', marker='o', label='Clusters recomendados', zorder=100)
        
        plt.plot(k_list, inertias, '-o')
        
        if prom_line == True:
            prom = np.mean(inertias)        
            plt.axhline(y = prom, color = "red")

        plt.xlabel('Número de clusters (k)')
        plt.ylabel('Inercia')
        plt.xticks(k_list)
        plt.title("Número de clusters vs Inercia.")
        plt.show()

    return clusters_recomendados, labels_model[clusters_recomendados]

# Ejemplos de uso:
# Uso más común... para cualquier set de datos...
# clusters_recomendados, df_datos["pred"] = kmeans_auto_clusters(df_datos, verbose = False, graph = True, prom_line = True, prop_limit = 0.1)

# En casos en que los datos de cada grupo están muy conectados o cercanos y se dificulta decidir como dividir los grupos a simple vista...
# clusters_recomendados, df_datos["pred"] = kmeans_auto_clusters(df_datos, verbose = False, graph = True, prom_line = True, prop_limit = 0.3)

# Más fácil aún... para entender y ver cada salida...
# kmeans_auto_clusters(pca_df, graph = True, prom_line = True, prop_limit = 0.1, verbose = True)
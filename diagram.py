from diagrams import Cluster, Diagram
from diagrams.programming.blank import Blank
from diagrams.onprem.analytics import Tensorflow


# Diagrama general
with Diagram("Proceso de desarrollo", show=True, outformat="png"):
    with Cluster("Análisis exploratorio"):
        datasets = Blank("Datasets:\n\t estudio3.csv\n\t tabla-resumen.csv")
        curado = Blank("Definición de variables E/S y limpieza")
        generacion = Blank("Generación de datasets curados para entrenamiento")
        
        with Cluster("Datasets para entrenamiento"):
            dataset_1 = Blank("Dataset 1 segundo")
            dataset_5 = Blank("Dataset 5 segundos")
        
    with Cluster("Entrenamiento de modelos"):
        modelos = Blank("Topologías de modelos")
        entrenamiento = Blank("Entrenamiento de modelos\n")
        test = Blank("Comparación de modelos")

    with Cluster("Validación de modelos"):
        validacion_preliminar = Blank("Validación preliminar") 

    start >> svc_group >> db_primary >> end
    

with Diagram("Red neuronal base", show=True, outformat="png"):
    with Cluster("Red neuronal base"):
        entrada = Tensorflow("Entrada: 40 variables")
        capa_1 = Tensorflow("Capa 1: 20 neuronas")
        capa_2 = Tensorflow("Capa 2: 10 neuronas")
        salida = Tensorflow("Salida: 7 neuronas")
        
        entrada >> capa_1 >> capa_2 >> salida
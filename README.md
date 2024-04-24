# rnr-meleros

Estudios para uso de Redes Neuronal Recurrentes (En ingles Recurrent Neural Network - RNN) aplicada sobre lecturas de acelerometro tomadas de Osos Meleros para predecir el comportamiento en tiempo real a partir de los valores medidos. 

## ToDo's

- [X] Limpieza y analisis dataset inicial
- [X] Prueba con distintos modelos nn clasicas (ver notebooks/dev) 
- [ ] Refactor del proyecto
    - [X] Nueva estructura
    - [ ] Refactor modulos
        - [ ] data
        - [ ] visualizacion
        - [ ] nn_model
        - [ ] implementar logs
    - [ ] Nuevos notebooks de proceso
        - [ ] Limpieza de datos
        - [ ] Analisis de datos
        - [ ] Entrenamiento
        - [ ] Evaluacion
- [ ] Implementar modulos y notebook RNR
- [ ] Añadir soporte para DB

## Intro

Se planteo el problema como uno de clasificación, ya que se desea inferir el comportamiento del animal a partir de las lecturas de un Acelerometro de 3 ejes y del calculo instantaneo ODBA del animal. Se busca evaluar distintos tipos de topologias de redes neuronales para comparar su performance para este tipo de problemas. También se incluyen en la comparativa los algoritmos de clasificacion clasicos. 

El repositorio contiene el codigo con el cual se implementaron:

- Algoritmos de clasificacion:
    - Clustering
- Redes neuronales clasicas (Feedforward Neural Networks)
- Redes neuronales recurrentes (Recurrent Neural Netwoks)

## Estructura

El proyecto esta pensado para correr en un ambiente containerizado dentro de docker con una [imagen base de jupyter](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html), que internamente levanta el kernel de desarrollo con [mamba](https://mamba.readthedocs.io/en/latest/index.html) para mantener los paquetes y ambientes organizados. 

Uso: En la terminal, corregir source ~/.bashrc y despues mamba activate dev-tf. 


## Trabajos a futuro

- Usar los modelos mejor performantes para realizar la clasificacion en tiempo real y contrastar con datos de una camara en tiempo real.
- Validar los modelos con otro/s dispositivo/s de medicion (Datalogger de otras marcas, agregar giroscopo, etc.)
- Implementar los mejores modelos para este caso de uso embebidas en el dispositivo de medición para detectar y relevar los comportamientos directamente en el dispositivo de medicion. 


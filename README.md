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

## Servidor (ambiente)

### Jupyter server

El proyecto esta pensado para correr en un ambiente containerizado dentro de docker con una [imagen base de jupyter](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html), que internamente levanta el kernel de desarrollo con [mamba](https://mamba.readthedocs.io/en/latest/index.html) para mantener los paquetes y ambientes organizados. 

Uso: En la terminal, corregir source ~/.bashrc y despues mamba activate dev-tf. 

Comandos para buildear la imagen y lanzar el contenedor de desarrollo:

´´´
$ docker build --rm -t dev-env .
$ docker run -i -t -p 8888:8888 -v "${PWD}":/home/jovyan/meleros dev-env
´´´

### VSCode usando kernel del server 

Si usas vscode, para conectar el kernel hay que poner en el existing jupyter server:

http://127.0.0.1:8888/tree

Cuando pide el password, ingresar el token que aparece al final de la url para el login despues de inicializar el contenedor de docker.

Ejemplo: 
- URL login web: http://127.0.0.1:8888/lab?token=3f570f3b8be77afd4866819421c1cddf69399b60f0e3153c
- Token (pass para vscode): 3f570f3b8be77afd4866819421c1cddf69399b60f0e3153c

## Trabajos a futuro

- Usar los modelos mejor performantes para realizar la clasificacion en tiempo real y contrastar con datos de una camara en tiempo real.
- Validar los modelos con otro/s dispositivo/s de medicion (Datalogger de otras marcas, agregar giroscopo, etc.)
- Implementar los mejores modelos para este caso de uso embebidas en el dispositivo de medición para detectar y relevar los comportamientos directamente en el dispositivo de medicion. 

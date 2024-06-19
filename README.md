# rnr-meleros

Estudios para uso de Redes Neuronal Recurrentes (En ingles Recurrent Neural Network - RNN) aplicada sobre lecturas de acelerometro tomadas de Osos Meleros para predecir el comportamiento en tiempo real a partir de los valores medidos. 


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

Si usas vscode, para conectar el kernel hay que poner en _existing jupyter server_:

http://127.0.0.1:8888/tree

Cuando pide el password, ingresar el token que aparece al final de la url para el login despues de inicializar el contenedor de docker.

Ejemplo: 
- URL login web: http://127.0.0.1:8888/lab?token=3f570f3b8be77afd4866819421c1cddf69399b60f0e3153c
- Token (pass para vscode): 3f570f3b8be77afd4866819421c1cddf69399b60f0e3153c


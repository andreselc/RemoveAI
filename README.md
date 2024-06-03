# Proyecto 2: RemoveAI-Usando el modelo de Machine Learning RMBG-1.4

Para este proyecto, se usó el modelo RMGB-1.4, cuya función principal es remover los fondos de las imágenes, siguiendo un rango de categorías y tipo de imágenes. Este modelo est;a entrenado sobre un dataset que incluye imágenes de archivos generales, e-commerce, juegos y demás contenido publicitario, haciéndola útil para casos comerciales y contenidos empresariales.  Cabe destacar que fue desarrollado por BriaAI, compañía que desarrolla contenido visual generativo con IA, por razones comerciales. 

## RMBG-1.4 adaptado para videos:

Este código contiene este modelo de segmentación entrenado sobre un dataset determinado. Dicho modelo fue entrenado sobre unas 12000 imágenes aproximadamente, de alta calidad, resolución, manualmente etiquetadas y con permisos legales para ser usadas sin fines comerciales. 

Según BriaAI, el modelo funciona de mejor forma dependiendo de la categoría de la imagen, siguiend los siguentes datos:

![Porcentajes de éxito en categoría de imágenes usadas en el modelo](REMOVEAI/../images/categorias.png)

### Imágenes de ejemplo:
![Imágenes de ejemplo](REMOVEAI/../images/ejemplos.png)

Ahora bien, el proyecto solicita aplicar un modelo que le quite los fondos a varios video, para ello, se adaptó el código de la siguiente manera:

### Carga del modelo:
1. Primeramente se carga el modelo de segmentación de imágenes RMBG-1.4 utilizando la función from_pretrained de la biblioteca transformers. Posteriormente, se verifica si hay una GPU disponible y se asigna el modelo al dispositivo correspondiente (GPU si está disponible, de lo contrario, CPU), y por último, se muestra un mensaje indicando quela carga completada del modelo. 

-Snippet:

  ```bash
print("Cargando el modelo...")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Modelo cargado y enviado a:", device)
```   

### Funciones de preprocesamiento y postprocesamiento:
2. 



    


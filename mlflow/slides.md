---
marp: true
theme: gaia
paginate: true
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  section {
    font-size: 1.5rem;
  }
  code {
    font-size: 1.2rem;
  }
---

# MLOps: Fundamentos y Mejores Prácticas
## Del Desarrollo a Producción

---

# ¿Qué es MLOps?

MLOps es un conjunto de prácticas que buscan:

- Simplificar flujos de trabajo
- Automatizar despliegues de ML/DL
- Mantener modelos en producción de forma confiable
- Escalar iniciativas de ML de forma eficiente
- Alinear demandas del negocio con requerimientos regulatorios

---

# MLOps en Sistemas Embebidos
<div class="columns">

<div>

### Optimización de Modelos
- Cuantización de modelos
- Pruning y compresión
- TinyML frameworks
- Edge computing optimizations

### Hardware Constraints
- Memoria limitada
- Capacidad computacional
- Consumo energético
- Latencia real-time
</div>

<div>

### Pipeline Específico
- Cross-compilation
- Testing en hardware real
- Simulación previa
- Versionado de firmware

### Frameworks Compatibles
- TensorFlow Lite
- TinyML
- Edge Impulse
- Arduino TinyML Kit
</div>

</div>

---

# Pipeline MLOps para Embebidos
<div class="columns">

<div>

### Desarrollo
- Prototipado rápido
- Simulación hardware
- Test unitarios específicos
- Optimización recursos

### Despliegue
- OTA updates
- Rollback automático
- Monitoreo remoto
- Gestión de versiones
</div>

<div>

### Validación
- Testing en dispositivo
- Benchmarking
- Medición consumo
- Pruebas de estrés

### Monitorización
- Telemetría
- Debug remoto
- Alertas hardware
- Logs específicos
</div>

</div>

---


# Beneficios Clave de MLOps

<div class="columns">

<div>

### Incremento de Productividad
- Automatización de tareas repetitivas
- Despliegue y mantenimiento eficiente
- Alineación entre equipos

### Reproducibilidad
- Automatización de workflows
- Versionado de datos y modelos
- Feature store y snapshots

### Reducción de Costos
- Minimización de esfuerzos manuales
- Detección temprana de errores
</div>

<div>

### Monitorización
- Seguimiento sistemático
- Reentrenamiento continuo
- Alertas de data/model drift
- Insights de performance

### Sostenibilidad
- Reducción de tareas repetitivas
- Uso eficiente de recursos
- Automatización de procesos

</div>

</div>

---

# Ciclo de Vida MLOps

![height:300px](https://api.placeholder.com/svg?text=ML%20Lifecycle)

<div class="columns">

<div>

### Obtención de Datos
- Múltiples fuentes
- Datos numéricos, texto, video
- Validación y calidad

### Ingeniería de Features
- Procesamiento y transformación
- Limpieza de duplicados
- Refinamiento de características
</div>

<div>

### Entrenamiento y Testing
- Experimentación con frameworks ML
- Optimización y tuning
- Versionado de modelos y datos

### Despliegue
- Frecuencia de actualización
- Gestión de alertas
- Monitorización continua
</div>

</div>

---

# Roles Clave en MLOps

<div class="columns">

<div>

### Data Scientist
- Análisis exploratorio
- Desarrollo de modelos
- Experimentación y validación
- Optimización de hiperparámetros

### ML Engineer
- Diseño de pipelines
- Automatización de procesos
- Integración de sistemas
- Optimización de rendimiento
</div>

<div>

### DevOps Engineer
- Infraestructura y escalabilidad
- CI/CD para ML
- Monitorización y logging
- Seguridad y gobernanza
</div>

</div>

---

# Principios MLOps

<div class="columns">

<div>

### Versionado
- Scripts ML
- Datasets
- Modelos
- Control de cambios

### Testing
- Validación de datos
- Tests de features
- Tests de infraestructura
- Tests de modelos
</div>

<div>

### Automatización
- Pipeline ML
- CI/CD
- Reentrenamiento
- Monitorización

### Reproducibilidad
- Resultados consistentes
- Backup de datos
- Control de versiones
- Documentación
</div>

</div>

---

# MLflow en Acción

### Tracking
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")
```

### Registro de Modelos
```python
model_name = "ProductRecommender"
mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/model",
    model_name
)
```

---

# Mejores Prácticas MLOps

<div class="columns">

<div>

### Seguridad desde Día 1
- Protección de datos
- Prevención de accesos
- Evitar data poisoning
- Gestión de vulnerabilidades

### Cumplimiento Normativo
- Regulaciones de privacidad
- Normativas financieras
- Guías de compliance
</div>

<div>

### Gestión de Desastres
- Backups de datos
- Modelos de respaldo
- Reentrenamiento rápido
- Planes de contingencia

### Calidad de Código
- Tests unitarios
- Linters y formatters
- Revisión de código
- Documentación clara
</div>

</div>

---

# Monitorización en Producción

<div class="columns">

<div>

### Métricas Clave
- Model drift
- Data drift
- Métricas de performance
- Métricas de sistema
</div>

<div>

### Herramientas
- MLflow
- Prometheus
- Grafana
- Custom dashboards
</div>

</div>

---

# Conclusiones

1. MLOps es fundamental para ML en producción
2. Requiere colaboración entre roles
3. Automatización y monitorización son clave
4. La seguridad y compliance son prioritarios
5. El testing y la reproducibilidad aseguran calidad
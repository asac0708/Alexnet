# AlexNet CIFAR-10 

Implementación de la arquitectura **AlexNet** adaptada para clasificación de imágenes en el dataset **CIFAR-10**, usando TensorFlow/Keras.

**Autora:** Sofía Ávila  
**Curso:** Aprendizaje Profundo — Universidad del Rosario  

---

## Descripción

AlexNet (Krizhevsky et al., 2012) fue la arquitectura ganadora de ImageNet 2012. En este taller se adapta para trabajar con imágenes de 32×32 px del dataset CIFAR-10, aplicando upsampling a 64×64 para preservar la estructura convolucional original.

### Clases CIFAR-10
`airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck`

---

## Arquitectura

| Bloque | Capa | Filtros | Kernel | Stride |
|--------|------|---------|--------|--------|
| Input | UpSampling2D (32→64) | — | — | — |
| 1 | Conv2D + BN + MaxPool | 96 | 11×11 | 4 |
| 2 | Conv2D + BN + MaxPool | 256 | 5×5 | 1 |
| 3 | Conv2D × 3 + MaxPool | 384/384/256 | 3×3 | 1 |
| FC | Dense + Dropout (×2) | 4096 | — | — |
| Out | Dense Softmax | 10 | — | — |

> Se reemplaza Local Response Normalization (LRN) por **BatchNormalization**, más estable en la práctica.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Test Accuracy | ver `results/metrics.txt` |
| Test Loss | ver `results/metrics.txt` |
| Best Val Accuracy | ver `results/metrics.txt` |

Las métricas detalladas por clase se encuentran en `results/metrics.txt` y `results/metrics.json`.

---

## Estructura del Repositorio

```
alexnet-cifar10/
│
├── alexnet_cifar10.py     ← Notebook principal (Colab)
├── requirements.txt          ← Dependencias
├── README.md
│
└── results/
    ├── metrics.txt
    ├── training_curves.png
    ├── confusion_matrix.png
```
Por el peso del archivo del modelo (>100 MB) la plataforma no permite subirlo. Se entrega por la plataforma de E-aulas e igual al correr el archivo se generará un nuevo .keras
---

## Cómo ejecutar

### Google Colab
1. Abrir `alexnet_cifar10.py` en Colab
2. Activar GPU: `Runtime > Change runtime type > T4 GPU`
3. Ejecutar todas las celdas

---

## Dependencias principales

- Python 3.10+
- TensorFlow 2.15
- NumPy, Matplotlib, Seaborn, Scikit-learn

Ver lista completa en `requirements.txt`.

---

## Referencias

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks*. NeurIPS.
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report.

import os
import pydicom
import numpy as np
import pandas as pd
import cv2


class ProcesadorDICOM:
    def __init__(self, ruta_directorio):
        """
        Inicializa la clase con la ruta del directorio que contiene los archivos DICOM.
        """
        self.ruta_directorio = ruta_directorio
        self.archivos_dicom = []
        self.dataframe_metadatos = None

    def cargar_archivos(self):
        """
        4.1 Carga de archivos DICOM.
        Escanea el directorio y carga los archivos válidos manejando excepciones.
        """
        print("Iniciando la carga de archivos...")
        for archivo in os.listdir(self.ruta_directorio):
            ruta_completa = os.path.join(self.ruta_directorio, archivo)
            
            if os.path.isfile(ruta_completa):
                try:
                    ds = pydicom.dcmread(ruta_completa)
                    self.archivos_dicom.append((archivo, ds))
                except pydicom.errors.InvalidDicomError:
                    print(f"Archivo ignorado (No es DICOM válido): {archivo}")
                except Exception as e:
                    print(f"Error al leer el archivo {archivo}: {e}")
                    
        print(f"Se cargaron con éxito {len(self.archivos_dicom)} archivos DICOM.\n")

    def extraer_y_estructurar_datos(self):
        """
        4.2 Extracción de metadatos y 4.3 Estructuración.
        Extrae los tags solicitados y los almacena en un DataFrame de Pandas.
        También incluye el paso 4.4 (Cálculo de intensidad promedio con NumPy).
        """
        lista_datos = []

        for nombre_archivo, ds in self.archivos_dicom:
            datos_archivo = {
                "Archivo": nombre_archivo,
                "ID_Paciente": ds.get("PatientID", "N/A"),
                "Nombre_Paciente": ds.get("PatientName", "N/A"),
                "ID_Estudio": ds.get("StudyInstanceUID", "N/A"),
                "Descripcion_Estudio": ds.get("StudyDescription", "N/A"),
                "Fecha_Estudio": ds.get("StudyDate", "N/A"),
                "Modalidad": ds.get("Modality", "N/A"),
                "Filas": ds.get("Rows", "N/A"),
                "Columnas": ds.get("Columns", "N/A")
            }

            if "PixelData" in ds:
                matriz_pixeles = ds.pixel_array
                promedio = np.mean(matriz_pixeles)
                datos_archivo["Intensidad Promedio"] = round(promedio, 2)
            else:
                datos_archivo["Intensidad Promedio"] = "No Imagen"

            lista_datos.append(datos_archivo)

        # 4.3 Convertir la lista de diccionarios en un DataFrame de Pandas
        self.dataframe_metadatos = pd.DataFrame(lista_datos)
        print("Metadatos extraídos y estructurados en el DataFrame.")
        return self.dataframe_metadatos

    def procesar_imagenes_opencv(self, carpeta_salida="resultados_procesados"):
        """
        4.5 Procesamiento de imágenes con OpenCV.
        Normaliza, ecualiza, aplica Canny y guarda el resultado en formato PNG.
        """
        # Crear la carpeta de salida si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)

        print(f"\nIniciando procesamiento de imágenes. Los resultados se guardarán en '{carpeta_salida}'...")

        for nombre_archivo, ds in self.archivos_dicom:
            try:
                # Verificar si el archivo tiene matriz de píxeles
                if not hasattr(ds, "pixel_array"):
                    continue
                
                img = ds.pixel_array
                
                # 1. Normalización a 8 bits (0 - 255)
                # Restamos el mínimo y escalamos respecto al rango original de la imagen
                img_min = img.min()
                img_max = img.max()
                
                if img_max - img_min == 0:
                    # Evitar división por cero si la imagen es completamente plana
                    img_normalizada = np.zeros(img.shape, dtype=np.uint8)
                else:
                    img_normalizada = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

                # 2. Ecualización del histograma
                img_ecualizada = cv2.equalizeHist(img_normalizada)

                # 3. Detección de bordes con Canny
                # Usamos umbrales de 50 y 150 (estándar para resaltar estructuras óseas o tejidos blandos)
                img_bordes = cv2.Canny(img_ecualizada, threshold1=50, threshold2=150)

                # 4. Guardado de resultados (PNG)
                # Usamos el identificador del estudio o el nombre del archivo de origen para nombrarlos
                id_estudio = getattr(ds, "StudyInstanceUID", "SinID")
                # Limpiamos caracteres extraños por si acaso
                id_seguro = "".join(x for x in str(id_estudio) if x.isalnum())[:15] 
                
                nombre_base = f"{nombre_archivo}_{id_seguro}"
                
                cv2.imwrite(os.path.join(carpeta_salida, f"{nombre_base}_ecualizada.png"), img_ecualizada)
                cv2.imwrite(os.path.join(carpeta_salida, f"{nombre_base}_bordes.png"), img_bordes)
                
                print(f"Procesado con éxito: {nombre_archivo}")

            except Exception as e:
                # Manejo de excepciones para archivos sin píxeles o con formatos comprimidos no soportados
                print(f"No se pudo procesar la imagen de {nombre_archivo}: {e}")


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # IMPORTANTE: Reemplaza 'tu_carpeta_con_dicoms' por la ruta real de tu carpeta de datos
    ruta_datos = "tu_carpeta_con_dicoms" 
    
    # Crear una carpeta de prueba automática si ejecutas el código por primera vez
    if not os.path.exists(ruta_datos):
        os.makedirs(ruta_datos)
        print(f"Se ha creado la carpeta '{ruta_datos}'. Pon tus archivos .dcm allí y vuelve a ejecutar.")
    else:
        # Instanciar la clase
        procesador = ProcesadorDICOM(ruta_datos)
        
        # Ejecutar el flujo completo
        procesador.cargar_archivos()
        df = procesador.extraer_y_estructurar_datos()
        
        # Mostrar el DataFrame en consola
        print("\n--- DATAFRAME DE METADATOS ---")
        print(df.to_string())
        
        # Procesar y guardar imágenes
        procesador.procesar_imagenes_opencv()
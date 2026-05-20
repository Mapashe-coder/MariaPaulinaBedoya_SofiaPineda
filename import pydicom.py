import pydicom
from pydicom.data import get_testdata_files

# Esto busca los archivos de prueba que trae pydicom internamente
archivos_prueba = get_testdata_files("*.dcm")

# Te mostrará las rutas donde están guardados en tu computador para que los copies
for ruta in archivos_prueba[:3]: 
    print("Archivo de prueba encontrado en:", ruta)
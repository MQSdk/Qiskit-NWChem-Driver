#!/bin/bash

# Ruta base donde están las carpetas
BASE_DIR="../data/PW_LiH_data/3x3_aperiodic/NWChem"

# Lista de carpetas específicas
FOLDERS=("0covo" "1nocovo" "4nocovo" "8nocovo" "12nocovo" "18nocovo")

# Iterar sobre cada carpeta
for FOLDER_NAME in "${FOLDERS[@]}"; do
    FOLDER="$BASE_DIR/$FOLDER_NAME"
    # Verificar que la carpeta existe
    if [ -d "$FOLDER" ]; then
        echo "Procesando carpeta: $FOLDER"
        
        # Encontrar todos los archivos H1Li1-*.nw en la carpeta actual
        for INPUT_FILE in "$FOLDER"/H1Li1-*.nw; do
            # Verificar que el archivo existe
            if [ -f "$INPUT_FILE" ]; then
                # Extraer el nombre del archivo sin la ruta ni la extensión
                BASENAME=$(basename "$INPUT_FILE" .nw)
                
                # Guardar el archivo .out en la misma carpeta que el .nw
                OUTPUT_FILE="$FOLDER/${BASENAME}.out"
                
                echo "Ejecutando: $INPUT_FILE -> $OUTPUT_FILE"
                
                # Ejecutar NWChem con 4 procesadores (ajusta según tu sistema)
                mpirun -np 24 nwchem "$INPUT_FILE" > "$OUTPUT_FILE" 2>&1
                
                # Verificar si la ejecución fue exitosa
                if [ $? -eq 0 ]; then
                    echo "Completado: $INPUT_FILE"
                else
                    echo "Error en: $INPUT_FILE. Revisa $OUTPUT_FILE"
                fi
            fi
        done
    else
        echo "Carpeta no encontrada: $FOLDER"
    fi
done

echo "Todos los cálculos han finalizado."

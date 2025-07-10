#!/bin/bash
set -e

# Activar entorno
source venv/bin/activate

while true; do
  echo "$(date): Ejecutando validación..."
  python validate/validate.py
  echo "$(date): Dormir 10 horas..."
  sleep 36000
done
@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  PIPELINE PRINCIPAL - Reproduccion completa de resultados
echo ============================================================
echo.
echo  Este script ejecuta el pipeline de principio a fin y lanza
echo  el dashboard interactivo para explorar los resultados.
echo.
echo  Los aeropuertos micro se leen de config.AEROPUERTOS_MICRO.
echo.
echo  Requisitos:
echo    - Python 3.12+ con las dependencias de requirements.txt
echo    - Datos en datos/parquet/ y datos/flight_list.csv
echo.
set /p confirma="Continuar? (S/N): "
if /i not "%confirma%"=="S" (
    echo Cancelado.
    pause
    exit /b 0
)

cd /d "%~dp0\..\.."

echo.
echo ============================================================
echo  [LIMPIEZA] Borrando resultados anteriores
echo ============================================================

if exist "resultados\preparacion" rmdir /S /Q "resultados\preparacion"
if exist "resultados\macro" rmdir /S /Q "resultados\macro"
if exist "resultados\micro" rmdir /S /Q "resultados\micro"
if exist "resultados\logs" rmdir /S /Q "resultados\logs"

echo Resultados anteriores eliminados.

echo.
echo ============================================================
echo  [1/8] FUSION DE PARQUETS DIARIOS
echo ============================================================
python -m src.preparacion.fusionar
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [2/8] LIMPIEZA DE TRAYECTORIAS
echo ============================================================
python -m src.preparacion.limpieza
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [3/8] PROYECCION LCC
echo ============================================================
python -m src.preparacion.proyeccion
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [4/8] REMUESTREO ESPACIAL
echo ============================================================
python -m src.preparacion.remuestreo_espacial
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [5/8] DISTANCIAS MACRO
echo ============================================================
python -m src.macro.distancias_macro
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [6/8] CLUSTERING MACRO
echo ============================================================
python -m src.macro.clustering_macro
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [7/8] RECORTE MICRO (aeropuertos de config.py)
echo ============================================================
python -m src.micro.recorte_micro
if !ERRORLEVEL! NEQ 0 goto error

echo.
echo ============================================================
echo  [8/8] DISTANCIAS + CLUSTERING MICRO WED
echo ============================================================
for /f "delims=" %%A in ('python -c "from src import config; [print(a) for a in config.AEROPUERTOS_MICRO]"') do (
    echo.
    echo --- Distancias micro %%A WED ---
    python -m src.micro.distancias_micro %%A --esquema WED
    if !ERRORLEVEL! NEQ 0 goto error

    echo --- Clustering micro %%A WED ---
    python -m src.micro.clustering_micro %%A --esquema WED
    if !ERRORLEVEL! NEQ 0 goto error
)

echo.
echo ============================================================
echo  PIPELINE COMPLETADO CON EXITO
echo ============================================================
echo.
echo  Resultados en: resultados\
echo  Logs en:       resultados\logs\
echo.
echo  Lanzando dashboard interactivo...
echo  Abrir http://127.0.0.1:8050 en el navegador.
echo  Pulsar Ctrl+C para detener.
echo.

python -m src.dashboard.dashboard_general

pause
exit /b 0

:error
echo.
echo ============================================================
echo  ERROR EN EL ULTIMO PASO EJECUTADO
echo ============================================================
echo  Revisa el log correspondiente en resultados\logs\
pause
exit /b 1
@echo off
echo ================================================
echo Testing CSV-to-YAML App Docker Container
echo ================================================

set IMAGE_NAME=kerollosadel/csv-to-yaml-app
set TAG=latest
set CONTAINER_NAME=csv-yaml-test
set PORT=5000

echo.
echo Starting test container...
echo.

:: Stop any existing test container
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1

:: Run the container in detached mode
docker run -d ^
  --name %CONTAINER_NAME% ^
  -p %PORT%:%PORT% ^
  -e FLASK_ENV=development ^
  %IMAGE_NAME%:%TAG%

if %ERRORLEVEL% equ 0 (
    echo Container started successfully!
    echo.
    echo Waiting for application to be ready...
    timeout /t 5 /nobreak >nul
    
    echo.
    echo ================================================
    echo Application is running!
    echo ================================================
    echo.
    echo Access the application at: http://localhost:%PORT%
    echo.
    echo To view logs:
    echo   docker logs %CONTAINER_NAME%
    echo.
    echo To stop the test:
    echo   docker stop %CONTAINER_NAME%
    echo   docker rm %CONTAINER_NAME%
    echo.
    echo Press any key to stop the container and exit...
    pause >nul
    
    echo.
    echo Stopping container...
    docker stop %CONTAINER_NAME%
    docker rm %CONTAINER_NAME%
    echo Container stopped and removed.
) else (
    echo.
    echo ================================================
    echo ERROR: Failed to start container!
    echo ================================================
    echo.
    echo Check if Docker is running and the image exists:
    echo   docker images ^| findstr %IMAGE_NAME%
    echo.
    exit /b 1
)

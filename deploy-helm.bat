@echo off
echo ================================================
echo Deploying CSV-to-YAML App with Helm
echo ================================================

:: Set deployment variables
set RELEASE_NAME=csv-to-yaml-app
set NAMESPACE=default
set CHART_PATH=./Helm

echo.
echo Deployment Configuration:
echo   Release Name: %RELEASE_NAME%
echo   Namespace: %NAMESPACE%
echo   Chart Path: %CHART_PATH%
echo.

:: Check if release already exists
echo Checking if release exists...
helm list -n %NAMESPACE% | findstr %RELEASE_NAME% >nul 2>&1

if %ERRORLEVEL% equ 0 (
    echo Release %RELEASE_NAME% already exists. Upgrading...
    helm upgrade %RELEASE_NAME% %CHART_PATH% -n %NAMESPACE% --wait
) else (
    echo Installing new release %RELEASE_NAME%...
    helm install %RELEASE_NAME% %CHART_PATH% -n %NAMESPACE% --create-namespace --wait
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo ================================================
    echo Deployment successful!
    echo ================================================
    echo.
    echo To check the deployment status:
    echo   helm status %RELEASE_NAME% -n %NAMESPACE%
    echo.
    echo To get the application URL:
    echo   kubectl get svc -n %NAMESPACE%
    echo.
    echo To view logs:
    echo   kubectl logs -l app.kubernetes.io/name=Helm -n %NAMESPACE%
    echo.
    echo To uninstall:
    echo   helm uninstall %RELEASE_NAME% -n %NAMESPACE%
    echo.
) else (
    echo.
    echo ================================================
    echo ERROR: Deployment failed!
    echo ================================================
    exit /b 1
)

@echo off
echo Fixing CloudSim/iFogSim imports in all Java files...

REM Replace imports in LoggingUtil.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\util\LoggingUtil.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\util\LoggingUtil.java'"

REM Replace imports in DataProcessor.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\util\DataProcessor.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\util\DataProcessor.java'"

REM Replace imports in ConfigurationManager.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\util\ConfigurationManager.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\util\ConfigurationManager.java'"

REM Replace imports in FogNode.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\topology\FogNode.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\topology\FogNode.java'"

REM Replace imports in EdgeNode.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\topology\EdgeNode.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\topology\EdgeNode.java'"

REM Replace imports in SecureFogSimulation.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\SecureFogSimulation.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\SecureFogSimulation.java'"
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\SecureFogSimulation.java') -replace 'import org.cloudbus.cloudsim.core.CloudSim;', '' | Set-Content 'src\main\java\org\nci\fogedge\SecureFogSimulation.java'"

REM Replace imports in SecurityManager.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\security\SecurityManager.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\security\SecurityManager.java'"

REM Replace imports in SimulationResults.java
powershell -Command "(Get-Content 'src\main\java\org\nci\fogedge\model\SimulationResults.java') -replace 'import org.cloudbus.cloudsim.Log;', 'import java.util.logging.Logger;' | Set-Content 'src\main\java\org\nci\fogedge\model\SimulationResults.java'"

echo Import replacements completed.
echo Running Maven build...

call mvn clean compile

echo Done.

param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PythonCommand
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PypiIndexUrl = if ([string]::IsNullOrWhiteSpace($env:PYPI_INDEX_URL)) {
    "https://pypi.org/simple"
} else {
    $env:PYPI_INDEX_URL
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"
$PythonProbe = "import sys; print('.'.join(map(str, sys.version_info[:3]))); print(sys.executable)"

function Format-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @()
    )

    if ($Arguments.Count -gt 0) {
        return "$FilePath $($Arguments -join ' ')"
    }

    return $FilePath
}

function Test-ExternalCommandPresence {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    return [bool](Get-Command -Name $FilePath -ErrorAction SilentlyContinue) -or (Test-Path -LiteralPath $FilePath)
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @(),

        [switch]$AllowFailure
    )

    $rawOutput = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $output = @($rawOutput | ForEach-Object { $_.ToString().TrimEnd() })

    if (-not $AllowFailure -and $exitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $exitCode, (Format-ExternalCommand -FilePath $FilePath -Arguments $Arguments))
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

function Get-PythonRuntimeInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @()
    )

    $commandText = Format-ExternalCommand -FilePath $FilePath -Arguments $Arguments

    if (-not (Test-ExternalCommandPresence -FilePath $FilePath)) {
        return [pscustomobject]@{
            FilePath = $FilePath
            Arguments = $Arguments
            CommandText = $commandText
            Status = "command_missing"
        }
    }

    $probeResult = Invoke-ExternalCommand -FilePath $FilePath -Arguments ($Arguments + @("-c", $PythonProbe)) -AllowFailure
    if ($probeResult.ExitCode -ne 0) {
        $errorText = $null
        if ($probeResult.Output.Count -gt 0) {
            $errorText = ($probeResult.Output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -First 1)
        }

        return [pscustomobject]@{
            FilePath = $FilePath
            Arguments = $Arguments
            CommandText = $commandText
            Status = "launch_failed"
            ExitCode = $probeResult.ExitCode
            ErrorText = $errorText
        }
    }

    if ($probeResult.Output.Count -lt 2) {
        return [pscustomobject]@{
            FilePath = $FilePath
            Arguments = $Arguments
            CommandText = $commandText
            Status = "probe_failed"
        }
    }

    $versionText = $probeResult.Output[0].Trim()
    $executablePath = $probeResult.Output[1].Trim()
    $versionMatch = [regex]::Match($versionText, "^(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)$")
    if (-not $versionMatch.Success) {
        return [pscustomobject]@{
            FilePath = $FilePath
            Arguments = $Arguments
            CommandText = $commandText
            Status = "probe_failed"
        }
    }

    $major = [int]$versionMatch.Groups["major"].Value
    $minor = [int]$versionMatch.Groups["minor"].Value
    $patch = [int]$versionMatch.Groups["patch"].Value
    $isSupported = ($major -eq 3 -and $minor -eq 13)

    return [pscustomobject]@{
        FilePath = $FilePath
        Arguments = $Arguments
        CommandText = $commandText
        Status = if ($isSupported) { "supported" } else { "unsupported_version" }
        VersionText = $versionText
        ExecutablePath = $executablePath
        IsSupported = $isSupported
        Major = $major
        Minor = $minor
        Patch = $patch
    }
}

function Format-PythonRuntimeStatus {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$RuntimeInfo
    )

    switch ($RuntimeInfo.Status) {
        "supported" {
            return ("{0}: Python {1} ({2})" -f $RuntimeInfo.CommandText, $RuntimeInfo.VersionText, $RuntimeInfo.ExecutablePath)
        }
        "unsupported_version" {
            return ("{0}: Python {1} ({2}) [requires Python 3.13.x]" -f $RuntimeInfo.CommandText, $RuntimeInfo.VersionText, $RuntimeInfo.ExecutablePath)
        }
        "command_missing" {
            return ("{0}: command not found" -f $RuntimeInfo.CommandText)
        }
        "launch_failed" {
            if (-not [string]::IsNullOrWhiteSpace($RuntimeInfo.ErrorText)) {
                return ("{0}: failed to start ({1})" -f $RuntimeInfo.CommandText, $RuntimeInfo.ErrorText)
            }

            return ("{0}: failed to start (exit code {1})" -f $RuntimeInfo.CommandText, $RuntimeInfo.ExitCode)
        }
        default {
            return ("{0}: failed to probe interpreter" -f $RuntimeInfo.CommandText)
        }
    }
}

function Write-NoSupportedPythonMessageAndExit {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject[]]$RuntimeInfos,

        [string]$RequestedCommandText
    )

    $messageLines = @(
        "Python 3.13.x is required for the full environment setup."
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedCommandText)) {
        $messageLines += ""
        $messageLines += ("Requested interpreter: {0}" -f $RequestedCommandText)
    }

    if ($RuntimeInfos.Count -gt 0) {
        $messageLines += ""
        $messageLines += "Detected runtimes:"
        foreach ($runtimeInfo in $RuntimeInfos) {
            $messageLines += ("  - {0}" -f (Format-PythonRuntimeStatus -RuntimeInfo $runtimeInfo))
        }
    }

    $messageLines += ""
    $messageLines += "Install Python 3.13.x and rerun:"
    $messageLines += "  scripts\setup_full_env.cmd"
    $messageLines += ""
    $messageLines += "Or pass an explicit 3.13 interpreter:"
    $messageLines += "  scripts\setup_full_env.cmd C:\Path\To\Python313\python.exe"

    [Console]::Error.WriteLine(($messageLines -join [Environment]::NewLine))
    exit 1
}

try {
    $runtimeInfos = @()
    $selectedRuntime = $null

    if ($PythonCommand -and $PythonCommand.Count -gt 0) {
        $requestedRuntime = Get-PythonRuntimeInfo -FilePath $PythonCommand[0] -Arguments $PythonCommand[1..($PythonCommand.Count - 1)]
        $runtimeInfos += $requestedRuntime

        if ($requestedRuntime.Status -ne "supported") {
            Write-NoSupportedPythonMessageAndExit -RuntimeInfos $runtimeInfos -RequestedCommandText $requestedRuntime.CommandText
        }

        $selectedRuntime = $requestedRuntime
    } else {
        $defaultCandidates = @(
            @{ FilePath = "py"; Arguments = @("-3.13") },
            @{ FilePath = "python3.13"; Arguments = @() },
            @{ FilePath = "python"; Arguments = @() }
        )

        foreach ($candidate in $defaultCandidates) {
            $runtimeInfo = Get-PythonRuntimeInfo -FilePath $candidate.FilePath -Arguments $candidate.Arguments
            $runtimeInfos += $runtimeInfo

            if ($runtimeInfo.Status -eq "supported") {
                $selectedRuntime = $runtimeInfo
                break
            }
        }

        if ($null -eq $selectedRuntime) {
            Write-NoSupportedPythonMessageAndExit -RuntimeInfos $runtimeInfos
        }
    }

    $env:PIP_NO_CACHE_DIR = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

    Invoke-ExternalCommand -FilePath $selectedRuntime.FilePath -Arguments ($selectedRuntime.Arguments + @("-m", "venv", $VenvDir))
    Invoke-ExternalCommand -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip<26", "setuptools<81", "wheel")
    Invoke-ExternalCommand -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.10.0+cpu")
    Invoke-ExternalCommand -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--index-url", $PypiIndexUrl, "-r", $RequirementsFile)
    Invoke-ExternalCommand -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--index-url", $PypiIndexUrl, "--no-deps", "LightAutoML==0.4.1")

    Write-Host "Environment is ready."
    Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}

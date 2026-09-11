# Termops — PowerShell terminal hook
# Source this in your PowerShell profile to auto-capture command errors.
#
# Install:  termops hook install
# Status:   termops hook status
# Uninstall: termops hook uninstall

$__termops_home = if ($env:TERMOPS_HOME) { $env:TERMOPS_HOME } else { Join-Path $HOME ".termops" }
$__termops_hook_file = Join-Path $__termops_home "hook_status.txt"
$Global:__termops_last_fingerprint = ""

# PS 7.3+: surface native command failures as ErrorRecords so their stderr
# text is capturable via $Error[0] instead of vanishing into the console.
if ($PSVersionTable.PSVersion -ge [version]"7.3") {
    $global:PSNativeCommandUseErrorActionPreference = $true
}

# Preserve the existing prompt so we can delegate after doing our capture.
$__termops_original_prompt = $function:prompt

function global:prompt {
    # Capture FIRST, before anything in here can clobber $? / $LASTEXITCODE.
    $last_success = $?
    $exit_code = $global:LASTEXITCODE
    $last_error = $global:Error[0]

    $failed = ($exit_code -is [int] -and $exit_code -ge 1) -or (-not $last_success)
    if ($failed -and (Test-Path $__termops_hook_file) -and (Get-Content $__termops_hook_file -Raw).Trim() -eq "enabled") {
        $last_cmd = ""
        try { $last_cmd = (Get-History -Count 1).CommandLine } catch {}
        $text = ""
        if ($last_error) { $text = ($last_error | Out-String).Trim() }
        if (-not $text) { $text = "Command '$last_cmd' exited with code $exit_code" }
        if ($text.Length -gt 2000) { $text = $text.Substring(0, 2000) }

        # Prompt redraws (resize etc.) must not resubmit the same failure.
        $fingerprint = "$last_cmd|$exit_code|$text"
        if ($fingerprint -ne $Global:__termops_last_fingerprint) {
            $Global:__termops_last_fingerprint = $fingerprint
            # Fire-and-forget: never block the prompt on the daemon round-trip.
            $arg_text = $text -replace '"', '\"'
            $arg_cmd = $last_cmd -replace '"', '\"'
            Start-Process -FilePath "termops" -WindowStyle Hidden `
                -ArgumentList "analyze --text `"$arg_text`" --source ps-hook --command `"$arg_cmd`" --exit-code $exit_code"
        }
    }

    if ($__termops_original_prompt) {
        & $__termops_original_prompt
    } else {
        "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
    }
}
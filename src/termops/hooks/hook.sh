# Termops — Bash/Zsh terminal hook
# Source this in your .bashrc or .zshrc to auto-capture command errors.
#
# Install:  termops hook install
# Status:   termops hook status
# Uninstall: termops hook uninstall

__termops_home="${TERMOPS_HOME:-$HOME/.termops}"
__termops_hook_file="$__termops_home/hook_status.txt"
__termops_stderr_log="$__termops_home/last_stderr.log"
__termops_last_exit=0
__termops_last_command=""

# Tee a copy of this shell's stderr into a rolling file so the *real* error
# text (not just the exit code) can be submitted. fd 9 preserves the original
# stderr; stdbuf -oL (GNU coreutils) keeps the tee line-buffered, otherwise
# the copy flushes in blocks and the tail read in precmd can race ahead.
if [ -d "$__termops_home" ]; then
    exec 9>&2
    if command -v stdbuf >/dev/null 2>&1; then
        exec 2> >(stdbuf -oL tee -a "$__termops_stderr_log" >&9)
    else
        exec 2> >(tee -a "$__termops_stderr_log" >&9)
    fi
fi

__termops_precmd() {
    __termops_last_exit=$?   # must stay the first line

    # Keep the rolling capture bounded (~64 KiB); tee -a + truncate is safe.
    if [ -f "$__termops_stderr_log" ] && [ "$(wc -c < "$__termops_stderr_log" 2>/dev/null || echo 0)" -gt 65536 ]; then
        : >| "$__termops_stderr_log"
    fi

    [ "$__termops_last_exit" -lt 1 ] && return
    [ -f "$__termops_hook_file" ] || return
    [ "$(cat "$__termops_hook_file")" = "enabled" ] || return

    case "$(basename "$SHELL")" in
        zsh)  __termops_last_command=$(fc -ln -1 2>/dev/null | sed 's/^ *//') ;;
        *)    __termops_last_command=$(history 1 | sed 's/^ *[0-9]\+ *//') ;;
    esac

    text=""
    [ -s "$__termops_stderr_log" ] && text=$(tail -c 4000 "$__termops_stderr_log")
    [ -z "$text" ] && text="Command '$__termops_last_command' exited with code $__termops_last_exit"

    # Background + detached: never stall the prompt on the daemon round-trip.
    ( termops analyze --text "$text" --source "sh-hook" \
        --command "$__termops_last_command" --exit-code "$__termops_last_exit" \
        >/dev/null 2>&1 & )
}

case "$(basename "$SHELL")" in
    zsh)
        autoload -Uz add-zsh-hook
        add-zsh-hook precmd __termops_precmd
        ;;
    bash)
        PROMPT_COMMAND="__termops_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
        ;;
esac
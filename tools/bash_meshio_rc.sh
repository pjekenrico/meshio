_meshio_complete() {
    local cur prev commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD - 1]}"

    # Define available subcommands
    commands="ascii binary compress convert decompress info"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- ${cur}))
        return 0
    fi

    # Autocomplete files for subcommands
    case "${prev}" in
    ascii | binary | compress | convert | decompress | info)
        COMPREPLY=($(compgen -f -- "${cur}"))
        return 0
        ;;
    esac
}

complete -F _meshio_complete meshio

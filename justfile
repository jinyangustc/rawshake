[private]
default:
    @just -g --list

sync target:
    rsync -azvhP . {{ target }}:/ws/$(basename $PWD) --exclude='/.git' --filter=':- .gitignore' --delete

grep PAT:
    uv run src/rs_geo/geophone.py 2>&1 | grep --line-buffered "{{ PAT }}"

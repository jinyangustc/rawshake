[private]
default:
    @just -f {{ justfile() }} --list

# rsync code to shake for on-device testing
sync target:
    rsync -azvhP . {{ target }}:/ws/$(basename $PWD) --exclude='/.git' --filter=':- .gitignore' --delete

# Run the geophone code and grep for a pattern
grep PAT:
    uv run src/rs_geo/geophone.py 2>&1 | grep --line-buffered "{{ PAT }}"

# Create a release
tag:
    #!/usr/bin/env bash
    version="v$(sed -n "s/^__version__ = '\([^']*\)'/\1/p" src/rawshake/__init__.py)"
    if git rev-parse "$version" &>/dev/null; then
        echo "Tag $version already exists"
    else
        git tag -a "$version" -m "$version"
        printf 'Tag %s created.\n' "$version"
        printf 'To publish the release, run:\n\n  git push origin %s\n\n' "$version"
    fi

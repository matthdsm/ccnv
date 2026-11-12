FROM ghcr.io/prefix-dev/pixi:0.84.1 AS build

WORKDIR /app
COPY pyproject.toml pixi.lock Cargo.toml Cargo.lock README.md LICENSE.md ./
COPY crates ./crates
COPY src ./src

RUN pixi install --locked -e prod \
    && rm -rf /app/.pixi/envs/prod/lib/rustlib /app/.pixi/envs/prod/bin/rust* /app/.pixi/envs/prod/bin/cargo* \
    && pixi shell-hook -e prod -s bash > /shell-hook \
    && printf '#!/bin/bash\n' > /app/entrypoint.sh \
    && cat /shell-hook >> /app/entrypoint.sh \
    && printf '\nexec "$@"\n' >> /app/entrypoint.sh

FROM ubuntu:26.04 AS production

WORKDIR /app
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY --from=build /app/src /app/src
COPY --from=build /app/pyproject.toml /app/pyproject.toml

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["ccnv"]

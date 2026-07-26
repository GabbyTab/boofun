#!/usr/bin/env bash
# Regenerate the SageMath cross-validation fixtures with full provenance.
#
# Runs scripts/generate_sage_fixtures.py inside the pinned SageMath Docker
# image and records the resolved image digest in the fixture metadata.
#
# Usage (from the repository root):
#   ./scripts/generate_sage_fixtures.sh [TAG]
# TAG defaults to the pinned version below. Bumping the pin is a deliberate
# act: update SAGE_TAG here, regenerate, and commit the fixture diff.
set -euo pipefail

SAGE_TAG="${1:-10.9}"
IMAGE="sagemath/sagemath:${SAGE_TAG}"
OUTPUT="tests/cross_validation/fixtures/sagemath.json"

# The sagemath/sagemath images only publish linux/amd64; on arm64 hosts
# (Apple Silicon) this runs under emulation, which is fine — the fixture
# values are exact integers and platform-independent.
docker pull --platform linux/amd64 "${IMAGE}"
DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${IMAGE}")"
echo "Using ${DIGEST}"

mkdir -p "$(dirname "${OUTPUT}")"
docker run --rm --platform linux/amd64 -v "${PWD}:/work" -w /work \
    -e SAGE_FIXTURE_IMAGE="${DIGEST}" \
    "${IMAGE}" \
    sage -python scripts/generate_sage_fixtures.py "${OUTPUT}"

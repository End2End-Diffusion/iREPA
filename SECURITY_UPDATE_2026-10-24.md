# Security dependency update — 2026-10-24

Update: certifi 2026.10.19.

The eMudhra incident requires updating application-bundled CA stores as well as
the operating system. Frozen environments must be recreated from the reviewed
updated lock; refreshing only a global tool does not update an existing venv
or container. Rebuild affected container images with fresh base images and
package metadata before reuse. Existing experiment snapshots remain historical
evidence; do not reuse their network clients without remediation.

This change preserves model code, data, experimental results and unrelated
package versions. It does not certify historical experiments, rebuild deployed
images, or authorize a deployment.

## Evidence and verification

The previous pinned certifi wheel was inspected without executing its code.
Loading its PEM bundle into Python's TLS context included the affected G3 root
(SHA-256 `86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b`).
The corresponding absence assertion fails on the previous bundle and passes
on 2026.10.19. The replacement wheel SHA-256 was checked against PyPI metadata:
`42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b`.
The sdist hash is
`035c1e85572b141bf1c3a3eff0e33a8eb74917522f2559075ff8d2fea780a1ae`.
This verifies the trust-store change; it is not an exploitation or GPU test.

## Primary sources

- [Mozilla incident response](https://blog.mozilla.org/security/2026/10/20/protecting-firefox-users-following-the-compromise-of-emudhra-intermediate-certificates/)
- [Python security guidance](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)
- [uv security release](https://github.com/astral-sh/uv/releases/tag/0.13.4)
- [Node.js security release](https://nodejs.org/en/blog/vulnerability/october-2026-security-releases)

See the pull request for repository-specific test results and limitations.

## Local validation

Validated the old/new certifi bundles without executing downloaded package code; the affected-root absence assertion fails before and passes after the update. Only the certifi requirement changes. GPU training was not rerun.

"""Regenerate the public-synthetic Pinares FEM fixed-price proxy fixtures."""

from __future__ import annotations

from finite_element_options.validation.pinares_fixed_price_proxy import (
    PINARES_FEM_PROXY_PROBLEM_SPEC_PATH,
    PINARES_FEM_PROXY_RESULT_EXPORT_PATH,
    PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH,
    PINARES_QPS_FIXTURE_PATH,
    run_public_pinares_fixed_price_proxy_fixture,
)


def main() -> None:
    """Run the deterministic fixture and refresh all public JSON exports."""

    report = run_public_pinares_fixed_price_proxy_fixture(refresh_exports=True)
    if not report.converged:
        raise SystemExit("Pinares FEM fixed-price proxy fixture failed tolerance gates")
    for path in (
        PINARES_FEM_PROXY_PROBLEM_SPEC_PATH,
        PINARES_FEM_PROXY_RESULT_EXPORT_PATH,
        PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH,
        PINARES_QPS_FIXTURE_PATH,
    ):
        print(path.relative_to(path.parents[3]))


if __name__ == "__main__":
    main()

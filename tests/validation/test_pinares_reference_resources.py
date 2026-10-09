"""Public Pinares reference bytes must be readable by ordinary wheel consumers."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from finite_element_options.validation import pinares_fixed_price_proxy as proxy

# Literal digests of the five original reviewed checkout snapshots at e946f38.
# These expectations do not depend on the package's resource or hash helpers.
REFERENCE_DIGESTS = {
    "PINARES_FEM_PROXY_PROBLEM_SPEC_PATH": "f7e48be4f88c572f6c4f11c0a3fdb12741dfaeebfe71a2da1762202096e6a092",
    "PINARES_FEM_PROXY_RESULT_EXPORT_PATH": "984e01a10d17693400b269a1fd935851abda34e253caeb2b36fd217d037698c1",
    "PINARES_FEM_PROVIDER_EVIDENCE_MANIFEST_PATH": "96c5d506eb3540e19a95a2679cd2c1645276ef936560d95e64f54ad7f87463ba",
    "PINARES_FEM_PROXY_UNSUPPORTED_SPEC_PATH": "02bd2bd3440dd14a76ab22b2732c6f673e239a014846da2bc918ba032bfbc1d3",
    "PINARES_QPS_FIXTURE_PATH": "f7e48be4f88c572f6c4f11c0a3fdb12741dfaeebfe71a2da1762202096e6a092",
}


class PinaresReferenceResources(unittest.TestCase):
    def test_original_public_reference_bytes_read_from_unrelated_directory(
        self,
    ) -> None:
        if sys.flags.isolated:
            self.assertTrue(
                Path(proxy.__file__).resolve().is_relative_to(Path(sys.prefix))
            )
            self.assertIsNone(importlib.util.find_spec("src"))
            distribution = metadata.distribution("finite-element-options")
            direct = json.loads(distribution.read_text("direct_url.json"))
            self.assertFalse(direct.get("dir_info", {}).get("editable", False))

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as unrelated:
            try:
                os.chdir(unrelated)
                for name, expected in REFERENCE_DIGESTS.items():
                    with self.subTest(reference=name):
                        resource = getattr(proxy, name)
                        self.assertTrue(
                            resource.is_file(),
                            f"installed public reference is missing: {name}: {resource}",
                        )
                        contents = resource.read_bytes()
                        self.assertEqual(sha256(contents).hexdigest(), expected)
                        self.assertIsInstance(json.loads(contents), dict)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

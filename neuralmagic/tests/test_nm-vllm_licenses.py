import re
from importlib.metadata import metadata
from pathlib import Path
from typing import Tuple

from pytest import mark, param

import vllm

THIRD_PARTY_LICENSE_FILES = [
    ("NOTICE", r"Neural Magic vLLM.*\n.*Neuralmagic, Inc", ""),
    ("LICENSE.apache", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.awq", "MIT HAN Lab", ""),
    ("LICENSE.fastertransformer", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.gptq", "MIT License\n*.*turboderp", ""),
    ("LICENSE.marlin", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.punica", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.squeezellm", "MIT License\n*.*SqueezeAILab", ""),
    ("LICENSE.tensorrtllm", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.vllm", r"Apache License\n\s*Version 2.0", ""),
    (
        "LICENSE",
        r".*NEURAL MAGIC COMMUNITY LICENSE AGREEMENT.*",
        "",
    ),
    (
        "METADATA",
        ".*License: Neural Magic Community License",
        "",
    ),
]


class TestNMThirdPartyLicenseFiles:
    """
    These tests verify that the proper files for licensing purposes exist and
    [generally] have the expected content.
    """

    @staticmethod
    def _package_name_version() -> Tuple[str, str]:
        """
        provides the package name as pip understands it, and the package version
        :return: (name, version)
        """
        site_package = vllm.__path__[0]
        package_version = vllm.__version__
        return site_package, package_version

    @staticmethod
    def _build_dist_info_path() -> Path:
        """
        builds a Path to the nm-vllm dist-info site package directory
        """
        (
            package_name,
            package_version,
        ) = TestNMThirdPartyLicenseFiles._package_name_version()
        package_name = package_name.replace("vllm", "nm_vllm")
        return Path(f"{package_name}-{package_version}.dist-info")

    def check_file_exists_and_content(self, file_name: str,
                                      content_regex: str):
        """
        shared function to check license files
        :param file_name: the file to check.
        :param content_regex: the regular expression to search the file content
        """
        # since we want to ensure that the files are actually available to the
        # user, this test function specifically looks for the files, rather than
        # accessing dist-info metadata for the package
        dist_info = self._build_dist_info_path()
        file_path = dist_info / file_name

        assert (file_path.exists()
                ), f"failed to find the expected license info {file_path}"
        license_text = file_path.read_text("utf-8")
        assert re.search(content_regex, license_text), (
            f"license file {file_path} does not have expected content matching "
            f"{content_regex}")

    @mark.parametrize(
        ("file_name", "content_regex"),
        [param(lf[0], lf[1], marks=lf[2]) for lf in THIRD_PARTY_LICENSE_FILES],
    )
    def test_common_license_file_presence_content(self, file_name: str,
                                                  content_regex: str):
        """
        Check Neural Magic license files that are common to the community and
        enterprise packages
        """
        self.check_file_exists_and_content(file_name, content_regex)

    def test_expected_files_included(self, request):
        """
        verifies that the list of license files in the directory matches the
        list provided in the METADATA file included with the distribution.
        """
        # collect the list of files in the dist_info directory
        dist_info = self._build_dist_info_path()
        dist_info_license_list = [p.name for p in dist_info.glob("*.license")]
        dist_info_license_list.extend(
            [p.name for p in dist_info.glob("LICENSE*")])
        dist_info_license_list.extend(
            [p.name for p in dist_info.glob("NOTICE")])

        # collect the list of files that METADATA expects to be available
        vllm_metadata = metadata("nm-vllm")
        all_metadata_licenses = vllm_metadata.get_all("License-File")
        metadata_license_list = [
            license.replace("licenses/", "")
            for license in all_metadata_licenses
        ]

        if set(metadata_license_list) != set(dist_info_license_list):
            # Check that all of METADATA's files are in the directory
            metadata_licenses_not_in_dir = set(
                metadata_license_list).difference(set(dist_info_license_list))
            assert not metadata_licenses_not_in_dir, (
                "not all third party license files from METADATA are found in "
                "the package dist_info directory.\n"
                f"{metadata_licenses_not_in_dir}")

            # check if there are files in dist_info that are not listed in the
            # METADATA
            dist_info_licenses_not_in_metadata = set(
                dist_info_license_list).difference(set(metadata_license_list))
            assert not dist_info_licenses_not_in_metadata, (
                "additional license files are listed in package dist_info "
                "directory, not listed in METADATA.\n"
                f"{dist_info_licenses_not_in_metadata}")

        # check that other tests are verifying all the files listed in
        # METADATA we only need to check that the files listed in METADATA
        # are a subset of those listed in the files we test.
        tested_license_files = [
            lf[0] for lf in THIRD_PARTY_LICENSE_FILES if lf[0] != "METADATA"
        ]
        assert set(metadata_license_list).issubset(
            set(tested_license_files)
        ), ("packaged third party license files match the list in METADATA in "
            "the package dist_info. we need to update THIRD_PARTY_LICENSE_FILES"
            " to match so that test_common_license_file_presence_content will "
            "verify all license files. unaccounted for:\n"
            f"{set(tested_license_files).symmetric_difference(metadata_license_list)}"
            )

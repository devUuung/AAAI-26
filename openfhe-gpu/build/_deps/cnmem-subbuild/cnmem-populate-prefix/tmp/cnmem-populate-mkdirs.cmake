# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/opt/openfhe-gpu/build/_deps/cnmem-src")
  file(MAKE_DIRECTORY "/opt/openfhe-gpu/build/_deps/cnmem-src")
endif()
file(MAKE_DIRECTORY
  "/opt/openfhe-gpu/build/_deps/cnmem-build"
  "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix"
  "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/tmp"
  "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/src/cnmem-populate-stamp"
  "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/src"
  "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/src/cnmem-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/src/cnmem-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/opt/openfhe-gpu/build/_deps/cnmem-subbuild/cnmem-populate-prefix/src/cnmem-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

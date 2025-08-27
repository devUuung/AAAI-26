file(REMOVE_RECURSE
  "../../lib/libOPENFHEpke_static.a"
  "../../lib/libOPENFHEpke_static.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/OPENFHEpke_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

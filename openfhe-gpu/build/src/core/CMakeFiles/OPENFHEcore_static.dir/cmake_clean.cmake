file(REMOVE_RECURSE
  "../../lib/libOPENFHEcore_static.a"
  "../../lib/libOPENFHEcore_static.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CXX)
  include(CMakeFiles/OPENFHEcore_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

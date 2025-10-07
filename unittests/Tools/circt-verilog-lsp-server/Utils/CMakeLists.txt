add_circt_unittest(CIRCTVerilogLSPTests
  DebouncerTest.cpp
  ChangeBufferTest.cpp
  DocChangeBucketTest.cpp
  BucketRegistryTest.cpp
)

target_link_libraries(CIRCTVerilogLSPTests
  PRIVATE
  CIRCTVerilogLspUtilsHeaders
)

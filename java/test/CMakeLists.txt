
find_package(Java REQUIRED)
include(UseJava)

set(JAVA_TESTS
test_trtc.java
)

add_jar(JThrustRTCTests SOURCES ${JAVA_TESTS} INCLUDE_JARS JThrustRTC)
install_jar(JThrustRTCTests test_java)

